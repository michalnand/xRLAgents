import envpool
import numpy
import time

from ..training.ValuesLogger           import *

class VectorizedAtariWrapper:
    def __init__(self, env_name = "MontezumaRevenge-v5", num_envs=128, room_id_adr = 3):

        self.num_envs    = num_envs
        self.room_id_adr = room_id_adr
        
        # 1. Initialize hyper-optimized C++ backend
        self.env = envpool.make_gymnasium(
            env_name,
            num_envs=num_envs,
            img_width=96,
            img_height=96,
            stack_num=4,
            frame_skip=4,
            noop_max=30,
            repeat_action_probability=0.25,
            reward_clip=False,
            max_episode_steps=4500,
            episodic_life=False,
        )

        # The overall batch observation shape: (128, 4, 96, 96)
        self.obs_shape = self.env.observation_space.shape

        # The total number of discrete actions available (e.g., 18 for Atari)
        self.action_dim = int(self.env.action_space.n)

        self.env_log = ValuesLogger("env")
        self.steps = 0

        self._init_vars()


    def _init_vars(self):
        self.episode_score      = numpy.zeros(self.num_envs)
        self.episode_score_curr = numpy.zeros(self.num_envs)

        # lifelong visited rooms IDs
        self.room_id_all = {}

        self.room_id_curr = [] 
        for n in range(self.num_envs):
            self.room_id_curr.append({})

        self.explored_rooms = [] 
        for n in range(self.num_envs):
            self.explored_rooms.append({})

        self.explored_rooms_episode = numpy.zeros(self.num_envs)


        self.infos = []
        for n in range(self.num_envs):
            self.infos.append({})


    def __len__(self):
        return self.num_envs

    def reset(self):
        obs, info = self.env.reset()
        # EnvPool returns array layout: (num_envs, channels, height, width) -> (128, 4, 96, 96)
        # Convert directly to PyTorch tensor on GPU
        obs = numpy.array(obs / 255.0, dtype=numpy.float32)
        return obs

    def step(self, actions):
        # Step all 128 environments instantly in C++
        obs, rewards, terminated, truncated, infos = self.env.step(actions)

        obs = numpy.array(obs/255.0, dtype=numpy.float32)
    
        rewards_clip = numpy.clip(rewards, 0.0, 1.0)

        dones = numpy.array(terminated | truncated, dtype=numpy.bool)
       
        # score holded in self.episode_score
        self._update_score(rewards, dones)


        # udpate lifelong self.explored_rooms_max and episodic self.explored_rooms_episode
        if self.room_id_adr != -1 and (self.steps%10) == 0:
            self._update_rooms_counter(dones, infos["ram"])


      
      
        # udpate logs
        self.env_log.add("reward_episode_mean", self.episode_score.mean(), 1.0)
        self.env_log.add("reward_episode_std",  self.episode_score.std(), 1.0)
        self.env_log.add("reward_episode_min",  self.episode_score.min(), 1.0)
        self.env_log.add("reward_episode_max",  self.episode_score.max(), 1.0)


        self.env_log.add("explored_rooms", len(self.room_id_all), 1.0)
        self.env_log.add("explored_rooms_episode_mean", self.explored_rooms_episode.mean(), 1.0)
        self.env_log.add("explored_rooms_episode_std", self.explored_rooms_episode.std(), 1.0)
        self.env_log.add("explored_rooms_episode_min", self.explored_rooms_episode.min(), 1.0)
        self.env_log.add("explored_rooms_episode_max", self.explored_rooms_episode.max(), 1.0)

      
        self.steps+= 1

        return obs, rewards_clip, dones, self.infos
    

    def get_logs(self):
        return [self.env_log]
    

    def _update_score(self, rewards, dones):
        # update episode raw score info
        self.episode_score_curr+= rewards   

        for n in range(self.num_envs):
            if dones[n]:        
                self.episode_score[n] = float(self.episode_score_curr[n])
                self.episode_score_curr[n] = 0

                self.infos[n]["episode_score"] = self.episode_score[n]
               

    def _update_rooms_counter(self, dones, ram_buffer):

      
        
        # Extract address 3 (montezuma revenge)  across all parallel envs
        room_ids    = ram_buffer[:, self.room_id_adr].astype(int)


        for n in range(self.num_envs):
            room_id = room_ids[n]

            # global lifelong stats
            self.room_id_all[room_id] = 1

            # episodic room IDs
            self.room_id_curr[n][room_id] = 1
            
            if dones[n]:
                self.explored_rooms[n] = dict(self.room_id_curr[n])
                self.explored_rooms_episode[n] = len(dict(self.room_id_curr[n]))

                self.room_id_curr[n] = {}

            self.infos[n]["room_id"] = room_id



        
# --- Quick Verification Run ---
if __name__ == "__main__":
    envs = VectorizedAtariWrapper("MontezumaRevenge-v5", 128)

    
    obs = envs.reset()
    print("Initial Batch Tensor Shape:", obs.shape) # Output: torch.Size([128, 4, 96, 96])

    # Sample loop
    fps = 0.0
    for n in range(500):
        time_start = time.time()
        random_actions = numpy.random.randint(0, 18, size=128) # Atari has 18 discrete actions
        obs, rewards, dones, infos = envs.step(random_actions)

        time_stop = time.time()

        fps = 0.9*fps + 0.1*1.0/(time_stop - time_start)
        
        # Verify room tracking across environments works asynchronously 

        if (n%20) == 0:
            print("fps ", round(fps, 2))
            print("all explored rooms ", len(envs.room_id_all))
            print("curr explored rooms ", envs.explored_rooms_episode.max(), envs.explored_rooms_episode.mean())
            print("obs ", obs.shape)
            print("rewards ", rewards.shape)
            print("dones ", dones.shape)
            print("\n\n")
    