import envpool
import numpy
import time

from ..training.ValuesLogger           import *

class VectorizedAtariWrapper:
    def __init__(self, env_name = "MontezumaRevenge-v5", num_envs=128, room_id_adr = 3):

        self.num_envs = num_envs
        self.room_id_adr = room_id_adr
        self._init_infos()
        
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

        self.episode_score_curr = numpy.zeros(self.num_envs)
        self.episode_score      = numpy.zeros(self.num_envs)

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
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        obs = numpy.array(obs/255.0, dtype=numpy.float32)
    
        rewards_clip = numpy.clip(rewards, 0.0, 1.0)

        dones = numpy.array(terminated | truncated, dtype=numpy.bool)
        
        # Re-implementing your Custom Info Trackers (ExploredRooms / Custom Logic)
        # EnvPool provides 'info' as a dictionary containing batched arrays.
        # Inside Montezuma, RAM address 3 contains the room ID.
        # We extract this from EnvPool's built-in batched RAM tracking.
        ram_buffer  = info["ram"] # Shape: (128, 128) -> All 128 bytes of Atari RAM for 128 envs
        room_ids    = ram_buffer[:, self.room_id_adr].astype(int) # Extract address 3  across all parallel envs


        infos = []

        self.episode_score_curr+= rewards   

        explored_rooms_max = 0

        for n in range(self.num_envs):
            if dones[n]:        
                self.episode_score[n] = float(self.episode_score_curr[n])
                self.episode_score_curr[n] = 0


            room_id = room_ids[n]
            if room_id not in self.explored_rooms[n]:
                self.explored_rooms[n][room_id] = 1
            else:
                self.explored_rooms[n][room_id]+= 1

            explored_rooms_count = len(self.explored_rooms[n])

            if explored_rooms_count > explored_rooms_max:
                explored_rooms_max = int(explored_rooms_count)
            
            info = {}
            info["room_id"] = int(room_id)
            info["explored_rooms"]      = explored_rooms_count
            info["episode_score"]       = self.episode_score[n]
            info["episode_score_max"]   = numpy.max(self.episode_score)

            infos.append(info)




        self.env_log.add("reward_episode_mean", self.episode_score.mean(), 1.0)
        self.env_log.add("reward_episode_std",  self.episode_score.std(), 1.0)
        self.env_log.add("reward_episode_min",  self.episode_score.min(), 1.0)
        self.env_log.add("reward_episode_max",  self.episode_score.max(), 1.0)

        self.env_log.add("explored_rooms", explored_rooms_max, 1.0)
      

        return obs, rewards_clip, dones, infos
    

    def get_logs(self):
        return [self.env_log]
    

    def _init_infos(self):
        self.explored_rooms = []
        for n in range(self.num_envs):
            self.explored_rooms.append({})



        
# --- Quick Verification Run ---
if __name__ == "__main__":
    pipeline = VectorizedRLPipeline(num_envs=128)

    
    obs, info = pipeline.reset()
    print("Initial Batch Tensor Shape:", obs.shape) # Output: torch.Size([128, 4, 96, 96])

    # Sample loop
    fps = 0.0
    for n in range(500):
        time_start = time.time()
        random_actions = numpy.random.randint(0, 18, size=128) # Atari has 18 discrete actions
        obs, rewards, dones, infos = pipeline.step(random_actions)

        time_stop = time.time()

        fps = 0.9*fps + 0.1*1.0/(time_stop - time_start)
        
        # Verify room tracking across environments works asynchronously 

        if (n%20) == 0:
            print("fps ", round(fps, 2))
            print(f"Current Room ID for Env #0: {infos[0]['room_id']}")
            print("obs ", obs.shape)
            print("rewards ", rewards.shape)
            print("dones ", dones.shape)
            print("\n\n")
    