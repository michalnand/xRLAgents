import os

from .RLLogger import *

class RLTrainer:
    def __init__(self, envs, agent, result_path):
        n_envs = len(envs)

        self.envs        = envs
        self.agent       = agent
        self.result_path = result_path

        self.log_result_path = self.result_path + "logs/"

        # create result path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        if not os.path.exists(self.log_result_path):
            os.makedirs(self.log_result_path)
            

        self.logger = RLLogger(n_envs, self.log_result_path)


    def run(self, n_steps, log_period = 128):
        # reset all envs
        states = self.envs.reset()

        # main training loop
        for n in range(n_steps):
            states_new, rewards, dones, infos = self.agent.step(states)
            
            # add to log
            if hasattr(self.agent, "get_logs"):
                agent_logs = self.agent.get_logs()
            else:
                agent_logs = []

            if hasattr(self.envs, "get_logs"):
                envs_logs = self.envs.get_logs()
            else:
                envs_logs = []

            update_log = (n%log_period) == 0
            result_str = self.logger.update(n, rewards, dones, agent_logs, envs_logs, update_log)

            if update_log:
                print(result_str)


            if (n%(n_steps//10)) == 0:
                self.agent.save()

                if hasattr(self.envs, "save"):
                    self.envs.save()
                    
                print("saving model at step ", n)

        self.agent.save()


        if hasattr(self.envs, "close"):
            self.envs.close()


