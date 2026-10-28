import torch

class TrajectoryBufferContinuous:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
      
        self.clear()   
    

    def add(self, state, actions_mu, actions_var, value, actions, rewards, dones):  
        self.states[self.ptr]       = state.detach().to("cpu").clone() 
        self.values[self.ptr]       = value.squeeze(1).detach().to("cpu").clone() 
        self.actions[self.ptr]      = torch.from_numpy(actions)
        self.actions_mu[self.ptr]   = actions_mu.detach().to("cpu").clone()
        self.actions_var[self.ptr]  = actions_var.detach().to("cpu").clone()
        
        self.reward[self.ptr]    = torch.from_numpy(rewards)
        self.dones[self.ptr]     = torch.from_numpy(dones).float()
        
        self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True
        return False 
 

    def clear(self):
        self.states      = torch.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=torch.float32)
        self.actions_mu  = torch.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=torch.float32)
        self.actions_var = torch.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=torch.float32)
        self.values      = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        
        self.actions     = torch.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=torch.float32)
        self.reward      = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.dones       = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.ptr = 0  
 
 
    def compute_returns(self, gamma, lam = 0.95):
        self.returns, self.advantages   = self._gae(self.reward, self.values, self.dones, gamma, lam)
        
        #reshape buffer for faster batch sampling
        self.states      = self.states.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.actions_mu  = self.actions_mu.reshape((self.buffer_size*self.envs_count, self.actions_size))
        self.actions_var = self.actions_var.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.values     = self.values.reshape((self.buffer_size*self.envs_count, ))        
     
        self.actions    = self.actions.reshape((self.buffer_size*self.envs_count, self.actions_size))
        
        self.reward     = self.reward.reshape((self.buffer_size*self.envs_count, ))
      
        self.dones      = self.dones.reshape((self.buffer_size*self.envs_count, ))

        self.returns    = self.returns.reshape((self.buffer_size*self.envs_count, ))
        self.advantages = self.advantages.reshape((self.buffer_size*self.envs_count, ))
   

    def sample_batch(self, batch_size, device):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))

        states          = self.states[indices].to(device)
        actions_mu      = self.actions_mu[indices].to(device)
        actions_var     = self.actions_var[indices].to(device)
        
        actions         = self.actions[indices].to(device)
        
        returns         = self.returns[indices].to(device)
        advantages      = self.advantages[indices].to(device)
       
        return states, actions_mu, actions_var, actions, returns, advantages

        
    def _gae(self, rewards, values, dones, gamma, lam):
        buffer_size = rewards.shape[0]
        envs_count  = rewards.shape[1]
        
        returns     = torch.zeros((buffer_size, envs_count), dtype=torch.float32)
        advantages  = torch.zeros((buffer_size, envs_count), dtype=torch.float32)

        last_gae    = torch.zeros((envs_count), dtype=torch.float32)
        
        for n in reversed(range(buffer_size-1)):
            delta           = rewards[n] + gamma*values[n+1]*(1.0 - dones[n]) - values[n]
            last_gae        = delta + gamma*lam*last_gae*(1.0 - dones[n])
            
            returns[n]      = last_gae + values[n]
            advantages[n]   = last_gae
 
        return returns, advantages
