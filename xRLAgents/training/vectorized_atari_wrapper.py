import envpool
import numpy
import time


class VectorizedAtariWrapper:
    def __init__(self, env_name = "MontezumaRevenge-v5", num_envs=128):

        self.num_envs    = num_envs
        
        # 1. Initialize hyper-optimized C++ backend
        # 4 stacked grayscale frames, size 96 x 96 pixels
        self.envs = envpool.make_gymnasium(
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
        self.observation_shape = self.envs.observation_space.shape

        # The total number of discrete actions available (e.g., 18 for Atari)
        self.num_actions = int(self.envs.action_space.n)

        self._init_logs_vars()


    def _init_logs_vars(self):
        self.episodes_count      = numpy.zeros(self.num_envs)

        self.episode_score      = numpy.zeros(self.num_envs)
        self.episode_score_curr = numpy.zeros(self.num_envs)

        self.episode_reward      = numpy.zeros(self.num_envs)
        self.episode_reward_curr = numpy.zeros(self.num_envs)

        self.episode_steps      = numpy.zeros(self.num_envs)
        self.episode_steps_curr = numpy.zeros(self.num_envs)



    def __len__(self):
        return self.num_envs
    
    def __getitem__(self, idx):
        return self.envs[idx]


    def reset_all(self):
        obs, infos = self.envs.reset()
        # EnvPool returns array layout: (num_envs, channels, height, width) -> (128, 4, 96, 96)
        # Convert directly to PyTorch tensor on GPU
        obs = numpy.array(obs / 255.0, dtype=numpy.float32)
        return obs, infos

    def step(self, actions):
        # Step all 128 environments instantly in C++
        obs, rewards, terminated, truncated, infos = self.envs.step(actions)

        obs = numpy.array(obs/255.0, dtype=numpy.float32)
    
        rewards_clip = numpy.clip(rewards, 0.0, 1.0)

        dones = numpy.array(terminated | truncated, dtype=numpy.bool)
       
        # raw score update
        self.episode_score_curr+= rewards

        # clipped score (what agent sees)
        self.episode_reward_curr+= rewards_clip

        # increment episode steps
        self.episode_steps_curr+= 1


        
        dones_ids = numpy.where(dones)[0]
        for i in dones_ids:
            self.episodes_count[i]+= 1

            self.episode_score[i]  = numpy.array(self.episode_score_curr[i])
            self.episode_reward[i] = numpy.array(self.episode_reward_curr[i])
            self.episode_steps[i]  = numpy.array(self.episode_steps_curr[i])

            self.episode_score_curr[i]  = 0
            self.episode_reward_curr[i] = 0
            self.episode_steps_curr[i]  = 0

      
        return obs, rewards_clip, terminated, truncated, infos

    
    def get_metrics(self, dp = 5):
        result = {}

        result["episodes_count"]    = round(self.episodes_count.mean(), dp)

        # raw rewards
        result["episode_score_mean"] = round(self.episode_score.mean(), dp)
        result["episode_score_std"]  = round(self.episode_score.std(), dp)
        result["episode_score_min"]  = round(self.episode_score.min(), dp)
        result["episode_score_max"]  = round(self.episode_score.max(), dp)

        # agent rewards 
        result["episode_reward_mean"] = round(self.episode_reward.mean(), dp)
        result["episode_reward_std"]  = round(self.episode_reward.std(), dp)
        result["episode_reward_min"]  = round(self.episode_reward.min(), dp)
        result["episode_reward_max"]  = round(self.episode_reward.max(), dp)

        # episode length
        result["episode_steps_mean"] = round(self.episode_steps.mean(), dp)
        result["episode_steps_std"]  = round(self.episode_steps.std(), dp)
        result["episode_steps_min"]  = round(self.episode_steps.min(), dp)
        result["episode_steps_max"]  = round(self.episode_steps.max(), dp)

        return result


    def get_default_log_flags(self):
        return {"env" : ["episode_score_mean", "episode_reward_mean", "episode_steps_mean"]}
    
       