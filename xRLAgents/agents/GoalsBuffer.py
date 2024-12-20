import torch 
import numpy

import cv2
  
class GoalsBuffer(): 
    def __init__(self, height, width, buffer_size, downsample = 8):

        self.downsample       = downsample

        self.states_raw       = numpy.zeros((buffer_size, height, width), dtype=numpy.float32)
        self.states_processed = numpy.zeros((buffer_size, height//downsample, width//downsample), dtype=numpy.float32)

        self.scores = -(10**6)*numpy.ones((buffer_size, ), dtype=numpy.float32)
        self.steps  = numpy.zeros((buffer_size, ), dtype=int)

        self.curr_ptr = 1

    # randomly select one goal from buffer
    # selection probability is given by scores vlaues
    def get_goal(self, temperature = 1.0):
        # normalise to sum equal to 1
        tmp = numpy.array(self.scores, dtype=numpy.float64)
        tmp = numpy.exp((tmp - numpy.max(tmp))/temperature)
        sum = numpy.sum(tmp)

        normalised = tmp/sum
        normalised = normalised.round(10)
        diff = 1.0 - numpy.sum(normalised)

        # use largets index to absorb difference
        max_index = numpy.argmax(normalised)
        normalised[max_index]+= diff

        goal_idx = numpy.random.choice(tmp.shape[0], p=normalised)

        return goal_idx, numpy.expand_dims(self.states_raw[goal_idx], 0)


    def step(self, goal_idx, state, steps, score, threshold = 0.002):
        state_tmp = state[0]
        state_processed = self._preprocess_frame(state_tmp)

        d = self.states_processed - state_processed
        d = (d**2).mean(axis=(1, 2))

        closest_idx = numpy.argmin(d)
        

        reach_reward = False
        steps_reward = False
        goal_added   = False

        # check if current goal idx reached
        if d[goal_idx] < threshold and score >= self.scores[goal_idx]:
            
            reach_reward = True
            
            # less steps to reach goal
            if steps < self.steps[goal_idx]:
                self.steps[goal_idx] = steps
                steps_reward = True

            #print("\n\ngoal reached ", goal_idx, reach_reward, steps_reward)
        
        # update closest goal if needed
        if d[closest_idx] < threshold:
            # update content and scores if higher
            if score > self.scores[closest_idx]:
                #print("\n\nscore updated for ", closest_idx, self.scores[closest_idx], score)

                self.states_raw[closest_idx]       = state_tmp.copy()
                self.states_processed[closest_idx] = state_processed.copy()
                self.scores[closest_idx]           = score
                self.steps[closest_idx]            = steps
       
        # add new goal
        elif d[closest_idx] > 1.5*threshold and self.curr_ptr < self.states_raw.shape[0]:
            #print("\n\nnew goal added ", d.mean(), d[closest_idx], score)

            self.states_raw[self.curr_ptr]       = state_tmp.copy()
            self.states_processed[self.curr_ptr] = state_processed.copy()
            self.scores[self.curr_ptr]           = score
            self.steps[self.curr_ptr]            = steps

            self.curr_ptr+= 1

            goal_added = True
 

        return reach_reward, steps_reward, goal_added

   
    def get_count(self):
        return self.curr_ptr


    def save(self, prefix):
        numpy.save(prefix + "raw.npy", self.states_raw)
        numpy.save(prefix + "processed.npy", self.states_processed)
        numpy.save(prefix + "scores.npy", self.scores)
        numpy.save(prefix + "steps.npy", self.steps)

    def _preprocess_frame(self, frame, levels_count=8):
        height = self.states_processed.shape[1]
        width  = self.states_processed.shape[2]

        # downsample 
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # discretise    
        quantized = (resized*levels_count).astype(numpy.uint8) 
        quantized = numpy.array(quantized, dtype=numpy.float32)/levels_count 
        return quantized