import numpy
import multiprocessing
import gymnasium as gym

from ..training.ValuesLogger           import *


def _env_process_main(id, env_name, Wrapper, render_mode, child_conn):

    print("creating env ", id, env_name)
    if isinstance(env_name, str):
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = env_name()

    if Wrapper is not None:
        env = Wrapper(env)



    while True:
        val 	= child_conn.recv()

        command = val[0]

        if command == "reset":
            state, info = env.reset()
            child_conn.send((state, info))

        elif command == "step":

            action = val[1]
            state, reward, done, truncated, info = env.step(action)

            child_conn.send((state, reward, done, truncated, info))

        elif command == "render":
            env.render() 

        elif command == "get":
            child_conn.send(env)

        else:
            env.close()
            break

'''
    multiple environments wrapper
'''
class EnvsListParallel:
    def __init__(self, env_name, n_envs, render_mode = None, Wrapper = None):

        if isinstance(env_name, str):
            env = gym.make(env_name, render_mode=render_mode)
        else:
            env = env_name()

        if Wrapper is not None:
            env = Wrapper(env)

        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        
        env.close()

        
        self.n_envs = n_envs
        
        #environments and threads
        self.parent_conn	= []
        self.child_conn		= []
        self.workers		= []

        for i in range(n_envs):
            parent_conn, child_conn = multiprocessing.Pipe()

            worker = multiprocessing.Process(target=_env_process_main, args=(i, env_name, Wrapper, render_mode, child_conn))
            worker.daemon = True

            self.parent_conn.append(parent_conn)
            self.child_conn.append(child_conn)
            self.workers.append(worker) 

        for i in range(n_envs):
            self.workers[i].start()

        self.score_per_episode = numpy.zeros(self.n_envs)
        self.score_per_episode_curr = numpy.zeros(self.n_envs)
    
        self.env_log = ValuesLogger("env")

    def __len__(self):
        return self.n_envs

    def step(self, actions):
        # send actions to environments
        for i in range(self.n_envs):
            action = actions[i]
            self.parent_conn[i].send(["step", action])


        states  = []
        rewards = []
        dones   = [] 
        infos   = []
        
        # obtain envs responses
        for i in range(self.n_envs):
            state, reward, done, _, info = self.parent_conn[i].recv()

            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

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
    
    
    '''
        reset all environments
    '''
    def reset_all(self):
        states = []
        infos  = [] 

        for i in range(self.n_envs):
            self.parent_conn[i].send(["reset"])

        for i in range(self.n_envs):
            state, info = self.parent_conn[i].recv()
           
            states.append(state)
            infos.append(info)

        states  = numpy.stack(states)

        return states, infos
    
    def reset(self, env_id):
        self.parent_conn[env_id].send(["reset"])
        state, info = self.parent_conn[env_id].recv()
        return state, info
    
    def render(self, env_id):
        self.parent_conn[env_id].send(["render"])

    def close(self):
        for i in range(len(self.workers)):
            self.parent_conn[i].send(["end"])

        for i in range(len(self.workers)):
            self.workers[i].join()

    def get_logs(self):
        return [self.env_log]

    