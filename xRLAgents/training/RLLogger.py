from .RLStats        import *
from .ValuesLogger   import *


class RLLogger:
    def __init__(self, n_envs, result_path = "./"):
        self.rl_stats   = RLStats(n_envs)
        self.rl_logger  = ValuesLogger("rl")

        self.result_path = result_path
        f = open(self.result_path + "summary.log", "w")
        f.close()

        self.output_files = {}

        
    def update(self, iteration, rewards, dones, agent_logs = None, envs_logs = None, update_log = True):
        # add into log
        dt, episodes, reward_episode_mean, reward_episode_std, reward_episode_max = self.rl_stats.add(iteration, rewards, dones)
        
        # monitored variables
        self.rl_logger.add("iterations", iteration, 1.0)
        self.rl_logger.add("dt", dt, 0.2)   
        self.rl_logger.add("episodes", episodes, 1.0)
        self.rl_logger.add("reward_episode_mean", reward_episode_mean, 1.0)
        self.rl_logger.add("reward_episode_std", reward_episode_std, 1.0)
        self.rl_logger.add("reward_episode_max", reward_episode_max, 1.0)

       
        # create files for log
        self._create_log_files(self.rl_logger, agent_logs, envs_logs)

        # obtain summary string
        result_str = self._get_summary_string(self.rl_logger, agent_logs, envs_logs)

        if update_log:
            # append to files
            self._add_to_files(self.rl_logger, agent_logs, envs_logs)

            
            f = open(self.result_path + "summary.log", "a+")
            f.write(result_str+"\n")
            f.close()

        return result_str
    

    def _create_log_files(self, rl_logger, agent_logs, envs_logs):

        if rl_logger.get_name() not in self.output_files:
            logger_name       = rl_logger.get_name()
            file_name         = self.result_path + logger_name + ".log"
            self.output_files[logger_name] = file_name

            f = open(file_name, "w")
            f.close()

            print("creating rl log file ", file_name) 

        if agent_logs is not None:
            for log in agent_logs:
                logger_name  = log.get_name()
                if logger_name not in self.output_files:
                    file_name = self.result_path + logger_name + ".log"
                    self.output_files[logger_name] = file_name

                    f = open(file_name, "w")
                    f.close()  

                    print("creating agent log file ", file_name) 

        if envs_logs is not None:
            for log in envs_logs:
                logger_name = log.get_name()
                if logger_name not in self.output_files:
                    file_name = self.result_path + logger_name + ".log"
                    self.output_files[logger_name] = file_name

                    f = open(file_name, "w")
                    f.close()  

                    print("creating env log file ", file_name) 


    def _add_to_files(self, rl_logger, agent_logs, envs_logs):

        self._add_log_to_file(rl_logger)

        if agent_logs is not None:
            for log in agent_logs:
                self._add_log_to_file(log)

        if envs_logs is not None:
            for log in envs_logs:
                self._add_log_to_file(log)

    def _add_log_to_file(self, logger):
        f_name = self.output_files[logger.get_name()]

        f = open(f_name, "a+")
        f.write(logger.get_str() + "\n")
        f.close()

        return logger.get_named_str()

    def _get_summary_string(self, rl_logger, agent_logs, envs_logs):
        result_str = ""
        result_str+= rl_logger.get_named_str()

        if agent_logs is not None:
            for log in agent_logs:
                if log.add_to_summary():
                    result_str+= log.get_named_str()

        if envs_logs is not None:
            for log in envs_logs:
                if log.add_to_summary():
                    result_str+= log.get_named_str()

        return result_str