import numpy
import gymnasium as gym


'''
    multiple environments wrapper
'''
class EnvsList:
    def __init__(self, env_name, n_envs, render_mode = None, Wrapper = None):
        
        self.envs   = []

        for _ in range(n_envs):
            if isinstance(env_name, str):
                env = gym.make(env_name, render_mode=render_mode)
            else:
                env = env_name()

            if Wrapper is not None:
                env = Wrapper(env)

            self.envs.append(env)


        self.observation_space = self.envs[0].observation_space
        self.action_space      = self.envs[0].action_space

        print("EnvsList")
        print("env	        : ", env_name)
        print("wrapper      : ", Wrapper)
        print("n_envs       : ", n_envs)
        print("\n\n")
        

    def __len__(self):
        return len(self.envs)

    def step(self, actions):

        states  = []
        rewards = []
        dones   = [] 
        infos   = []

        for i in range(len(self.envs)):
            state, reward, done, _, info = self.envs[i].step(actions[i])

            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        states  = numpy.stack(states)
        rewards = numpy.stack(rewards)
        dones   = numpy.stack(dones)

      

        
        return states, rewards, dones, infos
    
    def reset(self, env_id):
        return self.envs[env_id].reset()
    
    def reset_all(self):
        states = []
        infos  = []
        for n in range(len(self.envs)):
            state, info = self.reset(n)
        
            states.append(state)
            infos.append(info)

        states = numpy.stack(states)

        return states, infos
    
    def render(self, env_id):
        return self.envs[env_id].render()

    def __getitem__(self, index):
        return self.envs[index]
