import gymnasium as gym
import numpy
from PIL import Image





# after reset, wait random num of steps, for agent robustness
class NopOpsWrapper(gym.Wrapper):
    def __init__(self, env=None, max_count=30):
        super(NopOpsWrapper, self).__init__(env)
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
    


class SkipFramesWrapper(gym.Wrapper):
    def __init__(self, env=None, skip_count=4):
        super(SkipFramesWrapper, self).__init__(env)
        self.skip_count = skip_count

    def step(self, action):
        state_result = 0
        for n in range(self.skip_count):
            state, reward, done, truncated, info = self.env.step(action)

            state_result+= state

            if done:
                break

        state_result = state_result//self.skip_count

        return state_result, reward, done, truncated, info

# resize, grayscale and framestacking
class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, height = 96, width = 96, frame_stacking = 4):
        super(ResizeWrapper, self).__init__(env)
        self.height = height
        self.width  = width
        self.frame_stacking = frame_stacking

        state_shape = (self.frame_stacking, self.height, self.width)

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=numpy.float32)
        self.state = numpy.zeros(state_shape, dtype=numpy.float32)

    def observation(self, state):
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize((self.width, self.height))

        self.state    = numpy.roll(self.state, 1, axis=0)
        self.state[0] = (numpy.array(img).astype(self.dtype)/255.0).copy()
        
        return self.state 


# limit maximum steps for episode
class MaxStepsWrapper(gym.Wrapper):
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


# clip rewards to 0 or 1
class RewardsWrapper(gym.Wrapper):
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



# count explored rooms in montezuma rewenge, using ram memory room location
class ExploredRoomsWrapper(gym.Wrapper):
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




def WrapperMontezumaNew(env, height = 96, width = 96, frame_stacking = 4, max_steps = 4500):
    env = NopOpsWrapper(env)
    env = SkipFramesWrapper(env)
    env = ResizeWrapper(env, height, width, frame_stacking)

    env = MaxStepsWrapper(env, max_steps)
    env = RewardsWrapper(env)
    env = ExploredRoomsWrapper(env, room_address = 3)     

    return env



def WrapperMontezumaBNew(env, height = 96, width = 128, frame_stacking = 4, max_steps = 4500):
    env = NopOpsWrapper(env)
    env = SkipFramesWrapper(env)
    env = ResizeWrapper(env, height, width, frame_stacking)

    env = MaxStepsWrapper(env, max_steps)
    env = RewardsWrapper(env)
    env = ExploredRoomsWrapper(env, room_address = 3)     

    return env
