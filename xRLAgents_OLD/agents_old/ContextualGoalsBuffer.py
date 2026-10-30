import torch 
import numpy
from ..training.ValuesLogger           import *


class ContextualGoalsBuffer(): 
    def __init__(self, batch_size, buffer_size, height, width, add_threshold, reach_threshold,  alpha = 0.99, temperature = 10.0):
        self.batch_size     = batch_size
        self.buffer_size    = buffer_size  

       
        self.add_threshold   = add_threshold
        self.reach_threshold = reach_threshold

        self.alpha = alpha
        self.temperature = temperature


        self.states_mu      = torch.zeros((buffer_size, 1, height, width), dtype=torch.float32)
        self.states_var     = torch.ones((buffer_size, 1, height, width), dtype=torch.float32)

        self.scores         = -(10**6)*torch.ones((buffer_size, ), dtype=torch.float32)
        self.steps          = torch.zeros((buffer_size, ), dtype=int)

        self.goal_ids       = torch.zeros((buffer_size, ), dtype=int)

        self.curr_ptr = 0

        self.steps_cnt = 0
        self.log_goals_buffer = ValuesLogger("goals_buffer", False)

    

    def get_log(self):
        return self.log_goals_buffer
    
    def get_count(self):
        return self.curr_ptr
    

    # randomly select one goal from buffer
    # selection probability is given by scores values and num of steps to reach
    # higher score leads to more promissing goal
    # more far away the goal is, more promissing goal
    def create_new_goal(self, env_idx):
        normalised = self._get_goals_probs(self.temperature)
        
        normalised = normalised.round(10)
        diff = 1.0 - numpy.sum(normalised)

        # use largest index to absorb difference
        max_index = numpy.argmax(normalised)
        normalised[max_index]+= diff

        goal_idx = numpy.random.choice(normalised.shape[0], p=normalised)

        self.goal_ids[env_idx] = goal_idx

    
    def step(self, states, scores, steps):

        state_curr = states[:, 0].unsqueeze(1)

        # initialisation
        if  self.curr_ptr == 0:
            # goal idx 0 is dummy empty goal
            # add on posiiton 1, and set current pointer to empty
            self.states_mu[1]     = state_curr.copy()
            self.states_var[1]    = 1.0
            self.curr_ptr = 2

        # find closest
        dist = torch.cdist(self.states_mu.flatten(1), state_curr.flatten(1))
        closests_ids = torch.argmin(dist, dim=0)

        # extract context
        context = self.states_mu[closests_ids]

        # normalise state with respect to context
        states_norm = (states - self.states_mu[closests_ids])/(self.states_var[closests_ids]**0.5 + 1e-6)
        states_norm = torch.clip(states_norm, -4.0, 4.0)

        # update statistics
        self.states_mu[closests_ids]  = self.alpha*self.states_mu[closests_ids]  + (1.0 - self.alpha)*state_curr
        self.states_var[closests_ids] = self.alpha*self.states_var[closests_ids] + (1.0 - self.alpha)*((state_curr - self.states_mu[closests_ids])**2)



        # likelihoods weighted with variance
        likelihoods = torch.exp(-0.5 * ((state_curr - self.states_mu[closests_ids]) ** 2) / (self.states_var[closests_ids] + 1e-2))
        likelihoods = torch.mean(likelihoods, dim=-1)

        # decission if need to add new goal
        candidates = torch.where(likelihoods < self.add_threshold)[0]

        # create new goal
        goal_added = False
        for n in candidates:
            if self.curr_ptr < self.buffer_size:

                self.states_mu[self.curr_ptr]  = state_curr[n].detach().copy()
                self.states_var[self.curr_ptr] = 1.0

                self.scores[self.curr_ptr] = scores[n]
                self.steps[self.curr_ptr]  = steps[n]

                self.curr_ptr+= 1

                goal_added = True
                print("\nnew goal added ", n, self.curr_ptr, scores[n], steps[n], likelihoods[n])

                break
        

        # decission if goal reached
        goal_reached = torch.zeros(self.batch_size, dtype=bool)
        steps_reward = torch.zeros(self.batch_size, dtype=bool)

        candidates = torch.where(likelihoods > self.reach_threshold)[0]
        for n in candidates:
            goal_id = self.goal_ids[n]
            # check if reached only given goal, and non initial or first goal
            if goal_id > 1 and closests_ids[n] == goal_id:
                # goal reaching reward
                goal_reached[n] = True

                # reward for less steps to reach goal
                if steps[n] < self.steps[goal_id]:
                    self.steps[goal_id] = steps[n]
                    steps_reward[n] = True


                print("\ngoal reached ", n, goal_id, self.scores[goal_id], self.steps[goal_id])

                self.goal_ids[n] = 0

        goals = self.states_mu[self.goal_ids]
        
        
        # create output state
        result_state = torch.concatenate([states_norm, context, goals], dim=1)

        # update logs
        if self.steps_cnt%128 == 0:
            self._update_log(likelihoods)
        self.steps_cnt+= 1


        return result_state, self.goal_ids, goal_added, goal_reached, steps_reward

  
    def save(self, prefix):
        numpy.save(prefix + "raw.npy", self.states_raw)
        numpy.save(prefix + "processed_mu.npy", self.states_processed_mu)
        numpy.save(prefix + "processed_var.npy", self.states_processed_var)
        numpy.save(prefix + "scores.npy", self.scores)
        numpy.save(prefix + "steps.npy", self.steps)

    def load(self, prefix):
        self.states_raw             = numpy.load(prefix + "raw.npy")
        self.states_processed_mu    = numpy.load(prefix + "processed_mu.npy")
        self.states_processed_var   = numpy.load(prefix + "processed_var.npy")
        self.scores                 = numpy.load(prefix + "scores.npy")
        self.steps                  = numpy.load(prefix + "steps.npy")

        # count number of real stored states
        self.curr_ptr = 0
        for n in range(self.steps.shape[0]):
            self.curr_ptr+= 1
            if self.states_raw[n].sum() < 10e-6:
                break

    def _get_goals_probs(self, temperature):
        tmp = (1.0 + self.scores)*(1.0 + numpy.log(self.steps + 1.0))
        tmp = numpy.array(tmp, dtype=numpy.float64)

        # normalise to sum equal to 1
        tmp = numpy.exp((tmp - numpy.max(tmp))/temperature)
        sum = numpy.sum(tmp)

        result = tmp/sum

        return result

    def _update_log(self, likelihoods):
        # round to multiply by 32
        count = ((self.curr_ptr-1+32)//32)*32

        p = self._get_goals_probs(self.temperature) 

        self.log_goals_buffer.add("likelihoods_mean", round(likelihoods.mean(), 6), 1.0)
        self.log_goals_buffer.add("likelihoods_std",  round(likelihoods.std(), 6), 1.0)

        for n in range(count):
            self.log_goals_buffer.add("p" + str(n), round(p[n], 5), 1.0)
            self.log_goals_buffer.add("sc" + str(n), self.scores[n], 1.0)
            self.log_goals_buffer.add("st" + str(n), self.steps[n], 1.0)