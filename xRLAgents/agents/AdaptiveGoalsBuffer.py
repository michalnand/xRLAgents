import torch 
import numpy
      
class AdaptiveGoalsBuffer(): 
    def __init__(self, batch_size, buffer_size, height, width, threshold, alpha = 0.01):
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.mu  = 0.0
        self.var = 10e-6

        self.states_raw       = numpy.zeros((buffer_size, height, width), dtype=numpy.float32)
        state_processed       = self._features_func(numpy.zeros((1, height, width)))
        self.states_processed = numpy.zeros((buffer_size, ) + state_processed.shape, dtype=numpy.float32)

        self.scores = -(10**6)*numpy.ones((buffer_size, ), dtype=numpy.float32)
        self.steps  = numpy.zeros((buffer_size, ), dtype=int)

        self.threshold  = threshold
        self.alpha      = alpha

        self.curr_ptr   = 0

    # states    : current states, shape (batch_size, channels, height, width)
    # goal_idx  : ids of current goals to reach, shape (batch_size, )
    def step(self, states, goal_ids, scores, steps):
        # take only first frame
        states_tmp       = states[:, 0]
        states_processed = self._features_func(states_tmp)

        # initialisation
        if  self.curr_ptr == 0:
            self.states_raw[:]       = states_tmp[0].copy()
            self.states_processed[:] = states_processed[0].copy()
            self.curr_ptr+= 1

        # each by each distances
        # shape : (buffer_size, batch_size)
        d = numpy.expand_dims(self.states_processed, 1) - numpy.expand_dims(states_processed, 0)
        d = (d**2).mean(axis=-1)

        # shape : (batch_size, )
        # closest goal IDs and its distance
        closests_ids = numpy.argmin(d, axis=0)[0]
        d_min        = numpy.min(d, axis=0)

        print("closests_ids = ", closests_ids)
        print("d_min = ", d_min)


        # estimate mean and var using EMA
        self.mu  = (1.0 - self.alpha)*self.mu  + self.alpha*d_min.mean()
        self.var = (1.0 - self.alpha)*self.var + self.alpha*d_min.var()
        
        # compute adaptive threshold for goal reaching
        threshold  = self.mu + 0.1*self.threshold * (self.var ** 0.5)

        # goal reaching reward
        candidates   = numpy.where(d_min < threshold)[0]

        goal_reached = numpy.zeros(self.batch_size, dtype=bool)
        steps_reward = numpy.zeros(self.batch_size, dtype=bool)
        for n in candidates:
            closest_id = closests_ids[n]

            print(closest_id)

            # check if reached only given goal
            if closest_id == goal_ids[n]:
                goal_reached[n] = True

                # reward for less steps to reach goal
                if steps[n] < self.steps[closest_id]:
                    self.steps[closest_id] = steps[n]
                    steps_reward[n] = True

                print("goal reached ", n, d_min[n], self.scores[n], self.steps[n], steps_reward)
            
            # some goal reached, update it's values, only if reached with higher score
            elif scores[n] > self.scores[closest_id]:
                self.states_raw[closest_id]       = states_tmp[n].copy()
                self.states_processed[closest_id] = states_processed[n].copy()
                self.scores[closest_id]           = scores[n]
                self.steps[closest_id]            = steps[n]

                print("goal updated ", n, d_min[n], self.scores[n], self.steps[n])

        # new goal adding
        # compute adaptive threshold
        threshold  = self.mu + self.threshold * (self.var ** 0.5)

        candidates = numpy.where(d_min < threshold)[0]
        
        # add new goal states
        goal_added = False
        for n in candidates:
            if self.curr_ptr < self.buffer_size:

                self.states_raw[self.curr_ptr]       = states_tmp[n].copy()
                self.states_processed[self.curr_ptr] = states_processed[n].copy()

                self.scores[n] = scores[n]
                self.steps[n]  = steps[n]

                self.curr_ptr+= 1

                goal_added = True
                print("new goal added ", n, d_min[n], self.scores[n], self.steps[n])

                break

        return goal_reached, steps_reward, goal_added


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


    def _features_func(self, states, kernel_size = 4):
        
        x = torch.from_numpy(states).float()
        x = torch.nn.functional.avg_pool2d(x, (kernel_size, kernel_size), kernel_size)

        result = x.flatten(1)

        return result.detach().cpu().numpy()
    

    def save(self, prefix):
        numpy.save(prefix + "raw.npy", self.states_raw)
        numpy.save(prefix + "processed.npy", self.states_processed)
        numpy.save(prefix + "scores.npy", self.scores)
        numpy.save(prefix + "steps.npy", self.steps)

    def load(self, prefix):
        self.states_raw       = numpy.load(prefix + "raw.npy")
        self.states_processed = numpy.load(prefix + "processed.npy", self.states_processed)
        self.scores           = numpy.load(prefix + "scores.npy", self.scores)
        self.steps            = numpy.load(prefix + "steps.npy", self.steps)

        # count number of real stored states
        self.curr_ptr = 0
        for n in range(self.steps.shape[0]):
            self.curr_ptr+= 1
            if self.states_raw[n].sum() < 10e-6:
                break

