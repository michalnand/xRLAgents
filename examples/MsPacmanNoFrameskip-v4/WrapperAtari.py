import gymnasium as gym
import numpy
from PIL import Image

import cv2


class VideoRecorder(gym.Wrapper):
    def __init__(self, env, file_name = "records/video/video.mp4"):
        super(VideoRecorder, self).__init__(env)

        self.height  = env.observation_space.shape[0]
        self.width   = env.observation_space.shape[1]

        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        self.writer = cv2.VideoWriter(file_name, fourcc, 50.0, (self.width, self.height)) 
        self.frame_counter = 0 

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        
        #if self.frame_counter%2 == 0:
        im_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)

        resized = cv2.resize(im_bgr, (self.width, self.height), interpolation = cv2.INTER_AREA)

        self.writer.write(resized) 

        self.frame_counter+= 1

        if self.frame_counter%32 == 0:
            filename = "records/images/" + str(self.frame_counter//32) + ".png"
            cv2.imwrite(filename, im_bgr)

        return state, reward, done, truncated, info

    def reset(self, seed = None, options = None):
        return self.env.reset()



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


class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, height = 96, width = 96, frame_stacking = 4):
        super(ResizeEnv, self).__init__(env)
        self.height = height
        self.width  = width
        self.frame_stacking = frame_stacking

        state_shape = (self.frame_stacking, self.height, self.width)
        self.dtype  = numpy.float32

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=self.dtype)
        self.state = numpy.zeros(state_shape, dtype=self.dtype)

    def observation(self, state):
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize((self.width, self.height))

        self.state    = numpy.roll(self.state, 1, axis=0)
        self.state[0] = (numpy.array(img).astype(self.dtype)/255.0).copy()
        

        return self.state 


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



class StatesRecord(gym.Wrapper):
    def __init__(self, env, path = "records/states/"):
        gym.Wrapper.__init__(self, env)

        self.path = path
        self.cnt  = 0
       

    def reset(self, seed = None, options = None):
        return self.env.reset()

    def step(self, action):
       
        state, reward, done, truncated, info = self.env.step(action)

        file_name = self.path + str(self.cnt) + ".png"
        img = numpy.array(state[0]*255, dtype=numpy.uint8)
        cv2.imwrite(file_name, img)
        self.cnt+= 1

     
        return state, reward, done, truncated, info



     

def WrapperAtari(env, height = 96, width = 96, frame_stacking = 4, max_steps = 4500):
    env = VideoRecorder(env)
    env = NopOpsEnv(env)
    env = StickyActionEnv(env)
    env = RepeatActionEnv(env) 
    env = ResizeEnv(env, height, width, frame_stacking)
    env = MaxSteps(env, max_steps)
    env = Rewards(env)
    #env = StatesRecord(env)
     
    return env
