import numpy
import multiprocessing
import gymnasium as gym

import time

from ..training.ValuesLogger           import *


def _env_process_main(id, n_envs, env_name, Wrapper, render_mode, child_conn):

    print("creating env ", id, n_envs, env_name)

    envs = []
    for n in range(n_envs):
        if isinstance(env_name, str):
            env = gym.make(env_name, render_mode=render_mode)
        else:
            env = env_name()

        if Wrapper is not None:
            env = Wrapper(env)

        envs.append(env)



    while True:
        val 	= child_conn.recv()

        command = val[0]
        env_id  = val[1]

        if command == "reset":
            state, info = envs[env_id].reset()
            child_conn.send((state, info))

        elif command == "step":

            actions = val[1]

            states      = [] 
            rewards     = [] 
            dones       = [] 
            truncated   = [] 
            info        = []

            for n in range(n_envs): 
                state_, reward_, done_, truncated_, info_ = envs[n].step(actions[n])
                states.append(state_)
                rewards.append(reward_)
                dones.append(done_)
                truncated.append(truncated_)
                info.append(info_)  

            child_conn.send((states, rewards, dones, truncated, infos))

        elif command == "render":
            envs[env_id].render() 

        else:
            print("command error : ", command)
            env.close()
            break

'''
    multiple environments wrapper
'''
class EnvsListParallel:
    def __init__(self, env_name, n_envs, render_mode = None, Wrapper = None, n_threads = 4):
        if isinstance(env_name, str):
            env = gym.make(env_name, render_mode=render_mode)
        else:
            env = env_name()

        if Wrapper is not None:
            env = Wrapper(env)

        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        
        env.close()

        
        self.n_envs     = n_envs
        self.n_threads  = n_threads
        self.envs_per_thread = self.n_envs//self.n_threads

        print("EnvsListParallel")
        print("env	        : ", env_name)
        print("wrapper      : ", Wrapper)
        print("n_envs       : ", self.n_envs)
        print("n_threads    : ", self.n_threads)
        print("\n\n")

        #environments and threads
        self.parent_conn	= []
        self.child_conn		= []
        self.workers		= []

        for i in range(self.n_threads):
            parent_conn, child_conn = multiprocessing.Pipe()

            worker = multiprocessing.Process(target=_env_process_main, args=(i, self.envs_per_thread, env_name, Wrapper, render_mode, child_conn))
            worker.daemon = True

            self.parent_conn.append(parent_conn)
            self.child_conn.append(child_conn)
            self.workers.append(worker) 

        for i in range(self.n_threads):
            self.workers[i].start()

        time.sleep(2)

        self.score_per_episode = numpy.zeros(self.n_envs)
        self.score_per_episode_curr = numpy.zeros(self.n_envs)
    
        self.env_log = ValuesLogger("env")

    def __len__(self):
        return self.n_envs

    def step(self, actions):
        # send actions to all threads and its environments
        for i in range(self.n_threads):
            actions_ = actions[(i+0)*self.envs_per_thread:(i+1)*self.envs_per_thread]
            self.parent_conn[i].send(["step", actions_])


        states  = []
        rewards = []
        dones   = [] 
        infos   = []
        
        # obtain envs responses
        for i in range(self.n_threads):
            state_, reward_, done_, _, info_ = self.parent_conn[i].recv()

            for j in range(self.envs_per_thread):
                states.append(state_[j])
                rewards.append(reward_[j])
                dones.append(done_[j])
                infos.append(info_[j])

        states  = numpy.stack(states)
        rewards = numpy.stack(rewards)
        dones   = numpy.stack(dones)

        for i in range(self.n_envs):            
            if "raw_reward" in infos[i]:
                self.score_per_episode_curr[i]+= infos[i]["raw_reward"]
            else:
                self.score_per_episode_curr[i]+= rewards[i]

        dones_idx = numpy.where(dones)[0]
        for idx in dones_idx:
            self.score_per_episode[idx] = self.score_per_episode_curr[idx]
            self.score_per_episode_curr[idx] = 0.0

        self.env_log.add("reward_episode_mean", self.score_per_episode.mean(), 1.0)
        self.env_log.add("reward_episode_std", self.score_per_episode.std(), 1.0)
        self.env_log.add("reward_episode_min", self.score_per_episode.min(), 1.0)
        self.env_log.add("reward_episode_max", self.score_per_episode.max(), 1.0)

        return states, rewards, dones, infos
    
    
   
   
    def reset(self, env_id):
        thread_id, thread_env = self._get_ids(env_id)

        self.parent_conn[thread_id].send(["reset", thread_env])
        return self.parent_conn[thread_id].recv() 
    
    def reset_all(self):
        states = []
        infos  = []
        for n in range(self.n_envs):
            state, info = self.reset(n)
        
            states.append(state)
            infos.append(info)

        states = numpy.stack(states)

        return states, infos
    
    def render(self, env_id):
        thread_id, thread_env = self._get_ids(env_id)
        self.parent_conn[thread_id].send(["render", thread_env])

    def close(self):
        for i in range(len(self.workers)):
            self.parent_conn[i].send(["end"])

        for i in range(len(self.workers)):
            self.workers[i].join()

    def get_logs(self):
        return [self.env_log]
    
    def _get_ids(self, env_id):
        return env_id//self.envs_per_thread, env_id%self.envs_per_thread