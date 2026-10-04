import os
import numpy
import json

class ActionsEDA:

    def __init__(self, n_envs, n_actions, log_path, n_bins = 256):
        self.curr_actions = []
        for n in range(n_envs):
            self.curr_actions.append([])

        self.curr_rewards = []
        for n in range(n_envs):
            self.curr_rewards.append([])


        self.n_bins = n_bins

        self.actions_counts = numpy.zeros((n_bins, n_actions), dtype=int)
        
        
        self.episode_rewards     = numpy.zeros((n_bins, ), dtype=numpy.float32)

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        self.log_file_name = log_path + "/actions_eda.jsonl"
        f = open(self.log_file_name, "w")
        f.close()  
        print("creating log file ", self.log_file_name) 


    # actions = numpy((n_envs, ), int) 
    # dones = numpy((n_envs, ), bool)
    def __call__(self, actions, dones, rewards):
        for n in range(len(actions)):
            self.curr_actions[n].append(actions[n])
            self.curr_rewards[n].append(rewards[n])

     

        # this is sparse operation
        done_idx = numpy.where(dones)[0]
        for n in done_idx:  
            episode_length = len(self.curr_actions[n])

            # udpate raw histogram counts
            for i in range(episode_length):
                # map episode legnth to fixed bin index
                idx = int((self.n_bins*i)/(max(episode_length, 1)))
                action = self.curr_actions[n][i]
                self.actions_counts[idx][action]+= 1

                reward = self.curr_rewards[n][i]
                self.episode_rewards[idx] = round(0.9*self.episode_rewards[idx] + 0.1*reward, 5)

            # clear curr actions buffer
            self.curr_actions[n]    = []
            self.curr_rewards[n]    = []

    def save_log(self, iteration):

        result = {}
        result["iteration"] = iteration
        result["histogram"] = self.actions_counts.tolist()
        result["rewards"]   = self.episode_rewards.tolist()


        f = open(self.log_file_name, "a+")
        result_str = json.dumps(result)
        f.write(result_str + "\n")
        f.close() 
