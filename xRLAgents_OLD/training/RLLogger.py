from .RLStats        import *
from .ValuesLogger   import *

import json

class RLLogger:
    def __init__(self, n_envs, result_path = "./"):
        self.rl_stats   = RLStats(n_envs)
        self.rl_logger  = ValuesLogger("rl")

        self.result_path = result_path
        f = open(self.result_path + "summary.jsonl", "w")
        f.close()

        self.output_files = {}

        
    def update(self, iteration, rewards, dones, agent_logs = [], envs_logs = [], update_log = True):
        # add into log
        steps_per_second, episodes, reward_episode_mean, reward_episode_std, reward_episode_max = self.rl_stats.add(iteration, rewards, dones)
        
        # monitored variables
        self.rl_logger.add("iterations", iteration)
        self.rl_logger.add("steps_per_second", steps_per_second, 0.2)   
        self.rl_logger.add("episodes", episodes)
        self.rl_logger.add("reward_episode_mean", reward_episode_mean)
        self.rl_logger.add("reward_episode_std", reward_episode_std)
        self.rl_logger.add("reward_episode_max", reward_episode_max)


        # summary log
        summary_log = self._concatenate_log([self.rl_logger] + agent_logs + envs_logs)
        summary_str = json.dumps(summary_log)
        
     
        if update_log:
            # append separated logs to files
            # create file if not exitsts
            self._add_to_files([self.rl_logger] + agent_logs + envs_logs)

            # append summary log
            f = open(self.result_path + "summary.jsonl", "a+")
            f.write(summary_str+"\n")
            f.close()

        return summary_str
    

    def _concatenate_log(self, logs : list):
        result = {}
        for logger in logs:
            if logger.add_to_summary():
                name = logger.get_name()
                result[name] = logger.values

        return result

  
    def _add_to_files(self, logs_list : list):
        for logger in logs_list:

            logger_name  = logger.get_name()

            # create empty file not exists
            if logger_name not in self.output_files:
                file_name = self.result_path + logger_name + ".jsonl"
                self.output_files[logger_name] = file_name
            
                f = open(file_name, "w")
                f.close()  
            
                print("creating log file ", file_name) 

            # append to file
            f_name = self.output_files[logger_name]

            if len(logger.values) > 0:  
                f = open(f_name, "a+")
                result_str = json.dumps(logger.values)
                f.write(result_str + "\n")
                f.close() 
