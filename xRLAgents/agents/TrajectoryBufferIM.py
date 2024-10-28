import torch

class TrajectoryBufferIM:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
      
        self.clear()   
    

    def add(self, state, logits, values_ext, values_int, actions, rewards_ext, rewards_int, dones, episode_steps):  
        self.states[self.ptr]       = state.detach().to("cpu").clone() 
        self.logits[self.ptr]       = logits.detach().to("cpu").clone() 
        
        self.values_ext[self.ptr]   = values_ext.squeeze(1).detach().to("cpu").clone() 
        self.values_int[self.ptr]   = values_int.squeeze(1).detach().to("cpu").clone() 
        
        self.actions[self.ptr]      = torch.from_numpy(actions)
        
        self.rewards_ext[self.ptr]  = torch.from_numpy(rewards_ext)
        self.rewards_int[self.ptr]  = torch.from_numpy(rewards_int)

        self.dones[self.ptr]            = torch.from_numpy(dones).float()
        self.episode_steps[self.ptr]    = torch.from_numpy(episode_steps)
        
        self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True
        return False 
 

    def clear(self):
        self.states     = torch.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=torch.float32)
        self.logits     = torch.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=torch.float32)
        
        self.values_ext = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        
        self.values_int = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        

        self.actions    = torch.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        
        self.rewards_ext    = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.rewards_int    = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.dones      = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.episode_steps = torch.zeros((self.buffer_size, self.envs_count, ), dtype=int)

        self.ptr = 0  
 
    def compute_returns(self, gamma_ext, gamma_int, lam = 0.95):
        self.returns_ext, self.advantages_ext = self._gae(self.rewards_ext, self.values_ext, self.dones, gamma_ext, lam)
        self.returns_int, self.advantages_int = self._gae(self.rewards_int, self.values_int, self.dones, gamma_int, lam)
        
        
        #reshape buffer for faster batch sampling
        self.states     = self.states.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.logits     = self.logits.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.values_ext = self.values_ext.reshape((self.buffer_size*self.envs_count, ))
        self.values_int = self.values_int.reshape((self.buffer_size*self.envs_count, ))        
     
        self.actions    = self.actions.reshape((self.buffer_size*self.envs_count, ))
        

        self.rewards_ext = self.rewards_ext.reshape((self.buffer_size*self.envs_count, ))
        self.rewards_int = self.rewards_int.reshape((self.buffer_size*self.envs_count, ))

      
        self.dones         = self.dones.reshape((self.buffer_size*self.envs_count, ))
        self.episode_steps = self.episode_steps.reshape((self.buffer_size*self.envs_count, ))

        self.returns_ext      = self.returns_ext.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_ext   = self.advantages_ext.reshape((self.buffer_size*self.envs_count, ))

        self.returns_int      = self.returns_int.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_int   = self.advantages_int.reshape((self.buffer_size*self.envs_count, ))


    def sample_batch(self, batch_size, device):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))

        states          = (self.states[indices]).to(device)
        logits          = (self.logits[indices]).to(device)
        
        actions         = (self.actions[indices]).to(device)
         
        returns_ext     = (self.returns_ext[indices]).to(device)
        returns_int     = (self.returns_int[indices]).to(device)

        advantages_ext  = (self.advantages_ext[indices]).to(device)
        advantages_int  = (self.advantages_int[indices]).to(device)


        return states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int
    

    def sample_states(self, batch_size, max_distance, device):
        count        = self.buffer_size*self.envs_count

        indices_a   = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))

        dif         = torch.randint(0, max_distance + 1, (batch_size, ))
        indices_b   = torch.clip(indices_a + self.envs_count*dif, 0, count-1)


        states_a   = (self.states[indices_a]).to(device)
        steps_a    = (self.episode_steps[indices_a]).to(device)

        states_b   = (self.states[indices_b]).to(device)
        steps_b    = (self.episode_steps[indices_b]).to(device)

        return states_a, steps_a,  states_b, steps_b
    


    def sample_trajectory_states(self, trajectory_length, batch_size, device):
        indices = torch.randint(0, self.envs_count*(self.buffer_size - trajectory_length), size=(batch_size, ))
        states  = torch.zeros((trajectory_length, batch_size, ) + self.state_shape,  dtype=torch.float32, device=device)

        for n in range(trajectory_length):
            states[n]        = self.states[indices].to(device)
            indices+= self.envs_count 

        return states
    
     
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
