import torch 
import numpy



class FeaturesExtractor:
    def __init__(self, n_size = 8, n_bins = 8, device="cpu"):
        self.n_size     = n_size    
        self.n_bins     = n_bins

        self.device     = device

        self.sobel_x = torch.tensor([[[
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]]], dtype=torch.float32).to(self.device)

        self.sobel_y = torch.tensor([[[
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]]], dtype=torch.float32).to(self.device)
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):        
        x = x.unsqueeze(1).float().to(self.device)

        gradient_x =  torch.nn.functional.conv2d(x, self.sobel_x, padding=1)
        gradient_y =  torch.nn.functional.conv2d(x, self.sobel_y, padding=1)
        
        magnitude   = torch.sqrt(gradient_x**2 + gradient_y**2)
        magnitude   = magnitude.squeeze(1)
        orientation = torch.atan2(gradient_y, gradient_x)

        # convert orientation to degrees and map to [0, 180]
        orientation = torch.rad2deg(orientation)%180
        orientation = ((orientation*self.n_bins)//180).long().squeeze(1)
        
        
        # compute historgram
        result = torch.zeros((x.shape[0], self.n_bins, x.shape[2], x.shape[3]), device=self.device)

        for b in range(self.n_bins):
            result[:, b, :, :]+= magnitude*(orientation[:, :, :] == b)

        # grouping cells
        result = torch.nn.functional.avg_pool2d(result, self.n_size, stride=self.n_size)
        result = result/(result.sum(axis=1, keepdim=True) + 1e-6)

        x_tmp = torch.nn.functional.avg_pool2d(x, self.n_size, stride=self.n_size)

        result = torch.concatenate([x_tmp, result], dim=1)

        return result
      
class AdaptiveGoalsBuffer(): 
    def __init__(self, batch_size, buffer_size, height, width, threshold):
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.features_extractor = FeaturesExtractor(8, 4, "cuda")

        #self.mu  = 1.0
        #self.var = 1.0

        self.states_raw       = numpy.zeros((buffer_size, height, width), dtype=numpy.float32)
        state_processed       = self._features_func(numpy.zeros((1, height, width)))
        self.states_processed = numpy.zeros((buffer_size, ) + state_processed.shape, dtype=numpy.float32)

        self.scores = -(10**6)*numpy.ones((buffer_size, ), dtype=numpy.float32)
        self.steps  = numpy.zeros((buffer_size, ), dtype=int)

        self.threshold  = threshold

        self.curr_ptr   = 0

    # states    : current states, shape (batch_size, channels, height, width)
    # goal_idx  : ids of current goals to reach, shape (batch_size, )
    def step(self, states, goal_ids, scores, steps):
        # take only first frame
        states_tmp       = states[:, 0]
        states_processed = self._features_func(states_tmp)

        # initialisation
        if  self.curr_ptr == 0:
            self.states_raw[0]       = states_tmp[0].copy()
            self.states_processed[0] = states_processed[0].copy()
            self.curr_ptr+= 1

        # each by each distances
        # shape : (buffer_size, batch_size)
        d = numpy.expand_dims(self.states_processed, 1) - numpy.expand_dims(states_processed, 0)
        d = (d**2).mean(axis=-1)    

        # shape : (batch_size, )
        # closest goal IDs and its distance
        closests_ids = numpy.argmin(d, axis=0)[0]
        d_min        = numpy.min(d, axis=0)[0]


        # estimate mean and var using EMA
        #self.mu  = (1.0 - self.alpha)*self.mu  + self.alpha*d_min.mean()
        #self.var = (1.0 - self.alpha)*self.var + self.alpha*d_min.var()
        
        # compute adaptive threshold for goal reaching
        #threshold  = self.mu + 0.1*self.threshold * (self.var**0.5)

        # goal reaching
        candidates   = numpy.where(d_min < 0.1*self.threshold)[0]

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
                self.states_raw[closest_id]       = states_tmp[n].copy()
                self.states_processed[closest_id] = states_processed[n].copy()
                self.scores[closest_id]           = scores[n]
                self.steps[closest_id]            = steps[n]

                print("goal updated ", n, closest_id, d_min[n], self.scores[closest_id], self.steps[closest_id])

        # new goal adding
        # compute adaptive threshold for new goal add
        #threshold  = self.mu + self.threshold * (self.var**0.5)

        candidates = numpy.where(d_min > self.threshold)[0]

        # add new goal states
        goal_added = False

        for n in candidates:
            if self.curr_ptr < self.buffer_size:

                self.states_raw[self.curr_ptr]       = states_tmp[n].copy()
                self.states_processed[self.curr_ptr] = states_processed[n].copy()

                self.scores[self.curr_ptr] = scores[n]
                self.steps[self.curr_ptr]  = steps[n]

                self.curr_ptr+= 1

                goal_added = True
                print("new goal added ", n, self.curr_ptr, d_min[n], scores[n], steps[n])

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


    def _features_func(self, states):
        x = torch.from_numpy(states).float()
        z = self.features_extractor(x)  
        
        #z = torch.nn.functional.avg_pool2d(x, (8, 8), 8)
        z = z.flatten(1)

        return z.detach().cpu().numpy()
    

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

