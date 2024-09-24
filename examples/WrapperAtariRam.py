import gymnasium as gym
import numpy
from PIL import Image


'''
import cv2

class VideoRecorder(gym.Wrapper):
    def __init__(self, env, file_name = "video.avi"):
        super(VideoRecorder, self).__init__(env)

        self.height  = env.observation_space.shape[0]
        self.width   = env.observation_space.shape[1]

        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        self.writer = cv2.VideoWriter(file_name, fourcc, 50.0, (self.width, self.height)) 
        self.frame_counter = 0 

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        
        if self.frame_counter%2 == 0:
            im_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)

            resized = cv2.resize(im_bgr, (self.width, self.height), interpolation = cv2.INTER_AREA)

            self.writer.write(resized) 

        self.frame_counter+= 1

        return state, reward, done, truncated, info

    def reset(self, seed = None, options = None):
        return self.env.reset()
'''


class NopOpsEnv(gym.Wrapper):
    def __init__(self, env=None, max_count=30):
        super(NopOpsEnv, self).__init__(env)
        self.max_count = max_count

    def reset(self, seed = None, options = None):
        self.env.reset()

        noops = numpy.random.randint(1, self.max_count + 1)
         
        for _ in range(noops):
            obs, _, done, _, _ = self.env.step(0)

            if done:
                obs = self.env.reset()
           
        return obs, None

    def step(self, action):
        return self.env.step(action)

class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if numpy.random.uniform() < self.p:
            action = self.last_action

        self.last_action = action
        return self.env.step(action)

    def reset(self, seed = None, options = None):
        self.last_action = 0
        return self.env.reset()
 

class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.successive_frame = numpy.zeros((2,) + self.env.observation_space.shape, dtype=numpy.uint8)

    def reset(self, seed = None, options = None):
        return self.env.reset()

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, truncated, info = self.env.step(action)
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, truncated, info



class RamEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(RamEnv, self).__init__(env)

        self.observation_space = gym.spaces.Box(low=0.0, high=255.0, shape=(128, ), dtype=numpy.float32)

    def observation(self, state):
        observation_ram = self.env.unwrapped.ale.getRAM()
        observation_ram = numpy.array(observation_ram)

        #print(observation_ram)

        return observation_ram


class MaxSteps(gym.Wrapper):
    def __init__(self, env, max_steps):
        gym.Wrapper.__init__(self, env)
        self.max_steps = max_steps
        self.steps     = 0

    def reset(self, seed = None, options = None):
        self.steps     = 0
        return self.env.reset()

    def step(self, action):
       
        state, reward, done, truncated, info = self.env.step(action)
        
        self.steps+= 1
        if self.steps >= self.max_steps:
            done = True

        return state, reward, done, truncated, info



class Rewards(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
       

    def reset(self, seed = None, options = None):
        return self.env.reset()

    def step(self, action):
       
        state, reward, done, truncated, info = self.env.step(action)

        info["raw_reward"] = reward

        if reward > 0:
            reward = 1.0
        else:
            reward = 0.0
            
        return state, reward, done, truncated, info




def WrapperAtariRam(env, max_steps = 4500):
    env = NopOpsEnv(env)
    env = StickyActionEnv(env)
    env = RepeatActionEnv(env) 
    env = RamEnv(env)
    env = MaxSteps(env, max_steps)
    env = Rewards(env)
     
    return env
