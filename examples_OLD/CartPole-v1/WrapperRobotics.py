import gymnasium as gym
import numpy


class WrapperRobotics(gym.Wrapper):
    def __init__(self, env, max_steps = 1000):
        gym.Wrapper.__init__(self, env)

        self.max_steps = max_steps
        self.steps     = 0
       
    def reset(self, seed = None, options = None):
        self.steps = 0
        return self.env.reset()

    def step(self, action): 
       
        state, reward, done, truncated, info = self.env.step(action)

        self.steps+= 1 
        if self.steps >= self.max_steps:
            done = True

        return state, reward, done, truncated, info


