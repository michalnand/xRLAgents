import torch 
import numpy

import cv2
  
class GoalsBuffer(): 
    def __init__(self, height, width, buffer_size, downsample = 8):

        self.downsample       = downsample

        self.states_raw       = numpy.zeros((buffer_size, height, width), dtype=numpy.float32)
        self.states_processed = numpy.zeros((buffer_size, height//downsample, width//downsample), dtype=numpy.float32)

        self.scores = numpy.zeros((buffer_size, ), dtype=numpy.float32)
        self.steps  = numpy.zeros((buffer_size, ), dtype=int)

        self.scores[0] = 10**-4

        self.curr_ptr = 1

    # randomly select one goal from buffer
    # selection probability is given by scores vlaues
    def get_goal(self):
        # normalise to sum equal to 1
        tmp = numpy.array(self.scores, dtype=numpy.float64)
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

        state_processed = self._preprocess_frame(state[0])

        d = self.states_processed - state_processed
        d = (d**2).mean(axis=(1, 2))

        closest_idx = numpy.argmin(d)
        

        reach_reward = False
        steps_reward = False
        goal_added   = False

        # check if current goal idx reached
        if d[goal_idx] < threshold and score >= self.scores[goal_idx]:
            
            # fewer steps to reach goal
            if steps < self.steps[goal_idx]:
                self.steps[goal_idx] = steps
                steps_reward = True
            
            # update content and scores if higher
            if score > self.scores[goal_idx]:
                print("score updated for ", goal_idx, self.scores[goal_idx], score)

                self.states_raw[goal_idx]       = state.copy()
                self.states_processed[goal_idx] = state_processed.copy()
                self.scores[goal_idx]           = score
                self.steps[goal_idx]            = steps

            reach_reward = True
        
        # check if need add new goal
        if d[closest_idx] > threshold and self.curr_ptr < self.states_raw.shape[0]:
            print("new goal added ", d.mean(), d[closest_idx])
            self.states_raw[self.curr_ptr]       = state[0].copy()
            self.states_processed[self.curr_ptr] = state_processed.copy()
            self.scores[self.curr_ptr]           = score
            self.steps[self.curr_ptr]            = steps

            self.curr_ptr+= 1

            goal_added = True
 

        return reach_reward, steps_reward, goal_added

   
    def get_count(self):
        return self.curr_ptr

    def _preprocess_frame(self, frame, bits=8):
        height = self.states_processed.shape[1]
        width  = self.states_processed.shape[2]

        # downsample 
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        
        # discretise
        binned = (resized*bits).astype(numpy.uint8)  
        return numpy.array(binned, dtype=numpy.float32)/bits