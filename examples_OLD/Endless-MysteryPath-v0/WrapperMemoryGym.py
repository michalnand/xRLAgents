import memory_gym
import gymnasium as gym
import numpy
import cv2

import time

class WrapperMemoryGym(gym.Wrapper):
    def __init__(self, env, max_episode_steps = 4096):
        gym.Wrapper.__init__(self, env)
        self.obs_size          = 64
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3, self.obs_size, self.obs_size), dtype=numpy.float32)

        self.actions_list = []
        if isinstance(env.action_space, gym.spaces.MultiDiscrete): 
            for a in env.action_space:
                self.actions_list.append(a.n)
        else:
            self.actions_list.append(env.action_space.n)


        self.action_space = gym.spaces.Discrete(numpy.prod(self.actions_list))

        self.max_episode_steps = max_episode_steps
        self.steps             = 0


    def reset(self, seed = None, options = None):
        self.steps = 0

        obs, info = self.env.reset()

        self.state = self._wrap_state(obs)
        return self.state, None

    def step(self, action):
        if len(self.actions_list) > 1:
            a_tmp = self._wrap_action(action)
        else:
            a_tmp = action

        state, reward, done, truncated, info = self.env.step(a_tmp)
        self.state = self._wrap_state(state)

        self.steps+= 1
        if self.steps >= self.max_episode_steps:
            done = True

        return self.state, reward, done, truncated, info
    
    def render(self, size = 256):
        im = numpy.swapaxes(self.state, 2, 0)
        im = cv2.resize(im, (size, size), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("memory gym", im)
        cv2.waitKey(1)

    def _wrap_state(self, x):
        if isinstance(x, dict):
            x = x["visual_observation"]
            
        x = cv2.resize(x, (self.obs_size, self.obs_size), interpolation = cv2.INTER_CUBIC)
        x = numpy.swapaxes(x, 0, 2)
        x = numpy.array(x/255.0, dtype=numpy.float32)
        return x
    

    def _wrap_action(self, action):
        result = []
        for n in range(len(self.actions_list)):
            result.append(action%self.actions_list[n])
            action//= self.actions_list[n]

        return result
    


if __name__ == "__main__":

    #env = gym.make("SearingSpotlights-v0")
    #env = gym.make("Endless-MortarMayhem-v0")
    env = gym.make("Endless-MysteryPath-v0")
    
    
    env = WrapperMemoryGym(env)

    state, _ = env.reset()

    #print(">>> ", state.shape)

    episodes   = 0.0
    reward_all = 0.0
    reward_sum = 0.0
    while True:
        #print(env.action_space.sample())
        action = numpy.random.randint(0, env.action_space.n)
        obs, reward, done, truncation, info = env.step(action)
        
        #env.render()
      
        reward_sum+= reward
      
        if done:
            episodes+= 1
            reward_all+= reward_sum

            env.reset()
            print("episode reward ", reward_sum, reward_all/episodes)
            reward_sum = 0
            
        
        #time.sleep(0.1)
    
  