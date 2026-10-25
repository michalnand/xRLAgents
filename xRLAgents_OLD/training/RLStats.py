import numpy
import time


class RLStats:

    def __init__(self, n_envs):

        self.iterations = 0

        self.episodes            = numpy.zeros(n_envs, dtype=int)
        self.reward_episode_curr = numpy.zeros(n_envs)
        self.reward_episode      = numpy.zeros(n_envs)

        self.iterations_curr = 0
        self.iterations_prev = 0

        self.time_prev= time.time()
        self.time_now = time.time()
        self.steps_per_second = 100
        

    def add(self, iteration, rewards, dones):
        
        self.reward_episode_curr+= rewards

        dones_idx = numpy.where(dones)[0]
        for idx in dones_idx:
            self.episodes[idx]+= 1
            self.reward_episode[idx]      = self.reward_episode_curr[idx]
            self.reward_episode_curr[idx] = 0



        self.iterations_prev = self.iterations_curr
        self.iterations_curr = iteration

        self.time_prev = self.time_now
        self.time_now  = time.time()

        dit = self.iterations_curr - self.iterations_prev
        
        dt  = self.time_now - self.time_prev
        if dt <= 0.0:
            dt = 0.0001

        self.steps_per_second = round(dit/dt, 3)

        return self.steps_per_second, round(self.episodes.mean().item(), 2), round(self.reward_episode.mean().item(), 8), round(self.reward_episode.std().item(), 8), round(self.reward_episode.max().item(), 8)

  
    