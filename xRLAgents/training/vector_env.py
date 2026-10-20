import numpy


class VectorEnv:
    def __init__(self, num_envs, EnvClass, **kwargs):

        self.num_envs = num_envs

        self.envs = []
        for n in range(self.num_envs):
            env = EnvClass(**kwargs)
            self.envs.append(env)   

        self.observation_shape = self.envs[0].observation_space.shape
        self.action_shape      = self.envs[0].action_space.shape 

    def __len__(self):
        return self.num_envs
    
    def __getitem__(self, idx):
        return self.envs[idx]


    def step(self, actions):
        result_obs          = []
        result_reward       = []
        result_terminated   = []
        result_truncated    = []
        result_info         = []

        for n in range(self.num_envs):
            obs, reward, terminated, truncated, info = self.envs[n].step(actions[n])
            result_obs.append(obs)
            result_reward.append(reward)
            result_terminated.append(terminated)
            result_truncated.append(truncated)
            result_info.append(info)

        result_obs          = numpy.array(result_obs)
        result_reward       = numpy.array(result_reward)
        result_terminated   = numpy.array(result_terminated)
        result_truncated    = numpy.array(result_truncated)

        return result_obs, result_reward, result_terminated, result_truncated, result_info

    def reset(self, env_id):
        return self.envs[env_id].reset()

    def get_metrics(self, dp = 5):
        logs = []
        for n in range(self.num_envs):
            logs.append(self.envs[n].get_metrics())

        return self._compute_metrics_summary(logs, dp)
        


    def _compute_metrics_summary(self, log_list, dp):
        if not log_list:
            return {}
        
        keys = log_list[0].keys()
        summary = {}
        
        for key in keys:
            values = [d[key] for d in log_list if key in d]
            
            # Check if values are numeric
            if values and all(isinstance(v, (int, float)) for v in values):
                summary[f"{key}_mean"] = round(numpy.mean(values).item(), dp)
                summary[f"{key}_std"]  = round(numpy.std(values).item(), dp)
                summary[f"{key}_p68"]  = round(numpy.percentile(values, 68).item(), dp)
                summary[f"{key}_p95"]  = round(numpy.percentile(values, 95).item(), dp)
                summary[f"{key}_min"]  = round(numpy.min(values).item(), dp)
                summary[f"{key}_max"]  = round(numpy.max(values).item(), dp)
                summary[f"{key}_median"]  = round(numpy.median(values).item(), dp)
               
        return summary
        