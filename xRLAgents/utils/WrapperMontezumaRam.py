import gymnasium as gym
import numpy
from PIL import Image

import cv2


class RamVideoRecorder(gym.Wrapper):
    def __init__(self, env, file_name = "records/video/video.mp4"):
        super(RamVideoRecorder, self).__init__(env)

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




class RamStateEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        state_shape = (128, 8)
        self.dtype  = numpy.float32

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=self.dtype)
        
    def reset(self, seed = None, options = None):
        state, info = self.env.reset()
        return self._get_ram_state(state), info

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        return self._get_ram_state(state), reward, done, truncated, info

    def _get_ram_state(self, state):
        ram = numpy.array(state, dtype=numpy.uint8)

        # Convert to binary representation (shape: (n, 8))
        binary_arr = numpy.unpackbits(ram[:, numpy.newaxis], axis=1)
        binary_arr = numpy.array(binary_arr*1.0, dtype=self.dtype)

        return binary_arr
    
    '''
    def _get_ram_state(self, state):
        ram = self.env.unwrapped.ale.getRAM()
        ram = numpy.array(ram, dtype=numpy.uint8)

        # Convert to binary representation (shape: (n, 8))
        binary_arr = numpy.unpackbits(ram[:, numpy.newaxis], axis=1)
        binary_arr = numpy.array(binary_arr*1.0, dtype=self.dtype)

        return binary_arr
    '''



class RamNopOpsEnv(gym.Wrapper):
    def __init__(self, env=None, max_count=30):
        super(RamNopOpsEnv, self).__init__(env)
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


class RamStickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(RamStickyActionEnv, self).__init__(env)
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
 


class RamRepeatActionEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self, seed = None, options = None):
        return self.env.reset()

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, truncated, info = self.env.step(action)
            reward += r
            if done:
                break

        return state, reward, done, truncated, info





class RamMaxSteps(gym.Wrapper):
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



class RamRewards(gym.Wrapper):
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




class RamExploredRoomsEnv(gym.Wrapper):
    '''
    room_address for games : 
    montezuma revenge : 3
    pitfall           : 1
    '''
    def __init__(self, env, room_address = 3):
        gym.Wrapper.__init__(self, env)
        self.room_address = room_address

        self.explored_rooms         = {}
        self.explored_rooms_episode = {}

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        room_id = self._get_current_room_id()

        if room_id not in self.explored_rooms:
            self.explored_rooms[room_id] = 1
        else:
            self.explored_rooms[room_id]+= 1

        if room_id not in self.explored_rooms_episode:
            self.explored_rooms_episode[room_id] = 1
        else:
            self.explored_rooms_episode[room_id]+= 1

        info["room_id"]                 = room_id
        info["explored_rooms"]          = len(self.explored_rooms)
        info["explored_rooms_episode"]  = len(self.explored_rooms_episode)

        #print("room_id = ", room_id, len(self.explored_rooms))

        return obs, reward, done, truncated, info
    
    def reset(self, seed = None, options = None):
        self.explored_rooms_episode = {}
        return self.env.reset()

    def _get_current_room_id(self):
        ram = self.env.unwrapped.ale.getRAM()
        return int(ram[self.room_address])



     

def WrapperMontezumaRam(env, max_steps = 4500):
    #env = VideoRecorder(env)
    env = RamStateEnv()
    env = RamNopOpsEnv(env)
    env = RamStickyActionEnv(env)
    env = RamRepeatActionEnv(env) 
    env = RamMaxSteps(env, max_steps)
    env = RamRewards(env)
    env = RamExploredRoomsEnv(env, room_address = 3)     

    return env



def WrapperPitfallRam(env, max_steps = 4500):
    #env = VideoRecorder(env)
    env = RamStateEnv()
    env = RamNopOpsEnv(env)
    env = RamStickyActionEnv(env)
    env = RamRepeatActionEnv(env) 
    env = RamMaxSteps(env, max_steps)
    env = RamRewards(env)
    env = RamExploredRoomsEnv(env, room_address = 1)     

    return env
