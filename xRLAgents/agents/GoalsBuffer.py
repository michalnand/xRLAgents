import torch 
import numpy

  
class GoalsBuffer(): 
    def __init__(self, shape, buffer_size):

        self.curr_ptr = 1
        self.buffer = numpy.zeros((buffer_size, ) + shape, dtype=numpy.float32)

        self.scores = numpy.zeros((buffer_size, ), dtype=numpy.float32)
        self.steps  = numpy.zeros((buffer_size, ), dtype=int)

        self.scores[0] = 10**-4

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

        return goal_idx, self.buffer[goal_idx]

    # check if current goal has been reached and update all values
    def update(self, goal_idx, x, steps, score, threshold = 0.005):
        d = (self.buffer[goal_idx] - x)
        d = (d**2).mean()

        reach_reward = False
        steps_reward = False

        if d < threshold :
            reach_reward = True

            if steps < self.steps[goal_idx]:
                self.steps[goal_idx] = steps
                steps_reward = True

            if score > self.scores[goal_idx]:
                self.scores[goal_idx] = score

        return reach_reward, steps_reward
    

    def add(self, x, steps, score, threshold = 0.005):
        d = self.buffer - x
        d = (d**2).mean(axis=(1, 2, 3))

        closest = numpy.argmin(d)

        d_closest = d[closest]

        max_score = numpy.max(self.scores)  

        if d_closest > 2*threshold and self.curr_ptr < self.buffer.shape[0] and score > max_score:
            self.buffer[self.curr_ptr] = x
            self.steps[self.curr_ptr]  = steps
            self.scores[self.curr_ptr] = score

            self.curr_ptr+= 1   

            print("\n\n")
            print("new goal added ", d_closest, self.curr_ptr)
            print("\n\n")
            
            return True
        
        return False
    

    def get_count(self):
        return self.curr_ptr