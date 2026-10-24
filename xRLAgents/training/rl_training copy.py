from .training_logger import *
from .params_logger   import *

import time
import psutil
import os

class RLTraining:
    def __init__(self, config):
        self.config = config

        self.result_path  = config.result_path
        self.log_interval = config.log_interval

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path, exist_ok=True)

        if not os.path.exists(self.result_path + "checkpoints/"):
            os.makedirs(self.result_path + "checkpoints/", exist_ok=True)


        # store hyperparameters, extract from config
        param_logger(self.config, self.result_path)

        default_metrics = {}
        default_metrics = default_metrics | {"run " : ["steps_per_second", "eta_min", "cpu_usage", "memory_usage"]}
        default_metrics = default_metrics | self.config.agent.get_default_log_flags()
        default_metrics = default_metrics | self.config.envs.get_default_log_flags()

        result_file_name = os.path.join(self.result_path, "default_metrics.json")
        with open(result_file_name, "w") as f:
            json.dump(default_metrics, f, indent=4)

        # main logs
        self.run_log      = TrainingLogger(self.result_path, "run")
        self.env_logger   = TrainingLogger(self.result_path, "env")
        self.agent_logger = TrainingLogger(self.result_path, "agent")
        


    def run(self):

        env_logger   = TrainingLogger(self.result_path, "env")
        agent_logger = TrainingLogger(self.result_path, "agent")

        
        if hasattr(self.config.envs, "reset_all"):
            states, _ = self.config.envs.reset_all()
        else:
            states = []
            for n in range(len(self.config.envs)):
                obs, _ = self.config.envs.reset(n)
                states.append(obs) 
    
            states = numpy.array(states)

        saving_intervals = self.config.n_training_steps//10

        time_prev  = 0
        time_now   = time.time()

        last_save_step = -1

        pid = os.getpid()
        python_process = psutil.Process(pid)
        num_cores = psutil.cpu_count()

        for n in range(self.config.n_training_steps):
            states, rewards, terminated, truncated, info = self.config.agent.step(states)
            dones = terminated | truncated

            # resent envs if done
            if hasattr(self.config.envs, "reset"):
                dones_ids = numpy.where(dones)[0]
                for i in dones_ids:
                    obs, _ = self.config.envs.reset(i)
                    states[i] = obs

            if (n%self.log_interval) == 0:

                time_prev = time_now
                time_now  = time.time()

                # environment log
                log = self._add_step(n, self.config.envs.get_metrics())
                env_logger.add_raw(log)
                env_logger.save(verbose=True)

                # agent log
                log = self._add_step(n, self.config.agent.get_metrics())
                agent_logger.add_raw(log)
                agent_logger.save(verbose=True)

                # technical log
                steps_per_second = self.log_interval/(time_now - time_prev)
                eta_min          = ((self.config.n_training_steps - n)/steps_per_second)/60

                self.run_log.add("step", n)
                self.run_log.add("steps_per_second", steps_per_second, 3)
                self.run_log.add("eta_min", eta_min, 3)
                self.run_log.add("last_save_step", last_save_step, 0)

                self.run_log.add("cpu_usage",     float(python_process.cpu_percent()/num_cores), 1)
                self.run_log.add("memory_usage",  float(python_process.memory_info().rss)/1000000000, 6)

                self.run_log.save(verbose=True)

            if (n%saving_intervals) == 0:
                self.config.agent.save(self.result_path)
                last_save_step = n
                print("saving agent ", n)

        self.config.agent.save(self.result_path)

    def _add_step(self, n, rest):
        result = {}
        result["step"] = n
        result = result | rest

        return result

