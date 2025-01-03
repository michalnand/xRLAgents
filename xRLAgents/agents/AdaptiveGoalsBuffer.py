import torch 
import numpy


class AdaptiveGoalsBuffer(): 
    def __init__(self, batch_size, buffer_size, height, width, add_threshold, reach_threshold,  alpha = 0.99):
        self.batch_size     = batch_size
        self.buffer_size    = buffer_size  


        state_processed       = self._features_func(numpy.zeros((1, height, width)))

        self.states_raw           = numpy.zeros((buffer_size, height, width), dtype=numpy.float32)
        self.states_processed_mu  = numpy.zeros((buffer_size, ) + state_processed.shape[1:], dtype=numpy.float32)
        self.states_processed_var = numpy.ones((buffer_size, ) + state_processed.shape[1:], dtype=numpy.float32)


        self.scores = -(10**6)*numpy.ones((buffer_size, ), dtype=numpy.float32)
        self.steps  = numpy.zeros((buffer_size, ), dtype=int)

        self.add_threshold    = add_threshold
        self.reach_threshold  = reach_threshold
        self.alpha      = alpha

        self.curr_ptr   = 0
        

    def step(self, states, goal_ids, scores, steps):
        # take only first frame
        states_tmp       = states[:, 0]
        states_processed = self._features_func(states_tmp)

        # initialisation
        if  self.curr_ptr == 0:
            self.states_raw[0]              = states_tmp[0].copy()
            self.states_processed_mu[0]     = states_processed[0].copy()
            self.states_processed_var[0]    = 1.0
            self.curr_ptr+= 1

        #print("states_processed_mu ", self.states_processed_mu.shape, states_processed.shape)
        # each by each distances
        # shape : (buffer_size, batch_size)
        d = numpy.expand_dims(self.states_processed_mu, 1) - numpy.expand_dims(states_processed, 0)
        d = (d**2).mean(axis=-1)   
        #print("d = ", d.shape) 

        # shape : (batch_size, )
        # closest goal IDs and its distance
        closests_ids = numpy.argmin(d, axis=0)
        d_min        = numpy.min(d, axis=0)

        #print("closests_ids = ", closests_ids.shape, d_min.shape) 

        # update goals buffer statistics
        self.states_processed_mu[closests_ids]  = self.alpha*self.states_processed_mu[closests_ids]  + (1.0 - self.alpha)*states_processed
        self.states_processed_var[closests_ids] = self.alpha*self.states_processed_var[closests_ids] + (1.0 - self.alpha)*((states_processed - self.states_processed_mu[closests_ids])**2)

        #print("mu var = ", self.states_processed_mu.shape, self.states_processed_var.shape) 

        # new goal add
        # compute new state likelihoods
        likelihoods = numpy.exp(-0.5 * ((states_processed - self.states_processed_mu[closests_ids]) ** 2) / (self.states_processed_var[closests_ids] + 1e-2))
        likelihoods = numpy.mean(likelihoods, axis=-1)

        #print("likelihoods = ", likelihoods.shape) 



        # goal reaching
        candidates = numpy.where(likelihoods < self.reach_threshold)[0]

        print(candidates)
        print(likelihoods[candidates])
        print("\n\n")


        goal_reached = numpy.zeros(self.batch_size, dtype=bool)
        steps_reward = numpy.zeros(self.batch_size, dtype=bool)
        for n in candidates:
            closest_id = closests_ids[n]

            # check if reached only given goal
            if closest_id == goal_ids[n]:
                # goal reaching reward
                goal_reached[n] = True

                # reward for less steps to reach goal
                if steps[n] < self.steps[closest_id]:
                    self.steps[closest_id] = steps[n]
                    steps_reward[n] = True

                if goal_ids[n] != 0:
                    print("goal reached ", n, closest_id, d_min[n], self.scores[n], self.steps[n], steps_reward[n])
            
            # some goal reached, update it's values, only if reached with higher score
            elif scores[n] > self.scores[closest_id]:
                self.states_raw[closest_id]          = states_tmp[n].copy()
                self.states_processed_mu[closest_id] = states_processed[n].copy()
                self.scores[closest_id]              = scores[n]
                self.steps[closest_id]               = steps[n]

                print("goal updated ", n, closest_id, d_min[n], self.scores[closest_id], self.steps[closest_id])





        # add new goal states
        goal_added = False

        candidates = numpy.where(likelihoods < self.add_threshold)[0]


        #print("candidates = ", len(candidates)) 

        for n in candidates:
            if self.curr_ptr < self.buffer_size:

                self.states_raw[self.curr_ptr]           = states_tmp[n].copy()
                self.states_processed_mu[self.curr_ptr]  = states_processed[n].copy()
                self.states_processed_var[self.curr_ptr] = 1.0

                self.scores[self.curr_ptr] = scores[n]
                self.steps[self.curr_ptr]  = steps[n]

                self.curr_ptr+= 1

                goal_added = True
                print("new goal added ", n, self.curr_ptr, d_min[n], scores[n], steps[n], likelihoods[n])

                break
        
        return goal_reached, steps_reward, goal_added

    def get_count(self):
        return self.curr_ptr
      

    # randomly select one goal from buffer
    # selection probability is given by scores values and num of steps to reach
    # higher score leads to more promissing goal
    # more far away the goal is, more promissing goal
    def get_goal(self, temperature = 1.0):
        tmp = (1.0 + self.scores)*(1.0 + self.steps)
        tmp = numpy.array(tmp, dtype=numpy.float64)

        # normalise to sum equal to 1
        tmp = numpy.exp((tmp - numpy.max(tmp))/temperature)
        sum = numpy.sum(tmp)

        normalised = tmp/sum
        normalised = normalised.round(10)
        diff = 1.0 - numpy.sum(normalised)

        # use largest index to absorb difference
        max_index = numpy.argmax(normalised)
        normalised[max_index]+= diff

        goal_idx = numpy.random.choice(tmp.shape[0], p=normalised)

        return goal_idx, numpy.expand_dims(self.states_raw[goal_idx], 0)


    def _features_func(self, x):
        x = torch.from_numpy(x)
        z = torch.nn.functional.avg_pool2d(x, (8, 8), 8)
        z = z.flatten(1)

        return z.detach().cpu().numpy()
    
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

