import torch

class TrajectoryBufferDiscreteIM:

    def __init__(self, buffer_size, state_shape, num_actions, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.num_actions    = num_actions
        self.envs_count     = envs_count
      
        self.clear()   
    

    def add(self, state, logits, values_ext, values_int, actions, rewards_ext, rewards_int, dones):   
        self.states[self.ptr]       = state.detach().to("cpu").clone() 
        
        self.logits[self.ptr]       = logits.detach().to("cpu").clone() 

        self.values_ext[self.ptr]   = values_ext.squeeze(1).detach().to("cpu").clone() 
        self.values_int[self.ptr]   = values_int.squeeze(1).detach().to("cpu").clone() 
        
        self.actions[self.ptr]      = torch.from_numpy(actions)

        self.rewards_ext[self.ptr]    = torch.from_numpy(rewards_ext)
        self.rewards_int[self.ptr]    = torch.from_numpy(rewards_int)

        self.dones[self.ptr]          = torch.from_numpy(dones).float()
        
        self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True
        return False 
 

    def clear(self):
        self.states      = torch.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=torch.float32)
        self.logits      = torch.zeros((self.buffer_size, self.envs_count, self.num_actions), dtype=torch.float32)
        self.values_ext  = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        
        self.values_int  = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        
        self.actions     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.long)
        self.rewards_ext = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.rewards_int = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.dones       = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.ptr = 0  
 
 
    def compute_returns(self, gamma_ext, gamma_int, lam = 0.95, last_value_ext=None, last_value_int=None, normalize=True):
        self.returns_ext, self.advantages_ext  = self._gae(self.rewards_ext, self.values_ext, self.dones, gamma_ext, lam, last_value_ext)
        self.returns_int, self.advantages_int  = self._gae(self.rewards_int, self.values_int, self.dones, gamma_int, lam, last_value_int)

        # normalize advantages over the whole buffer
        if normalize:
            self.advantages_ext = (self.advantages_ext - self.advantages_ext.mean()) / (self.advantages_ext.std() + 1e-8)
            self.advantages_int = (self.advantages_int - self.advantages_int.mean()) / (self.advantages_int.std() + 1e-8)

        #reshape buffer for faster batch sampling
        n = self.buffer_size*self.envs_count

        self.states             = self.states.reshape((n, ) + self.state_shape)
        self.logits             = self.logits.reshape((n, self.num_actions) )
        self.values_ext         = self.values_ext.reshape((n, ))
        self.values_int         = self.values_int.reshape((n, ))
        self.actions            = self.actions.reshape((n, ))
        self.rewards_ext        = self.rewards_ext.reshape((n, ))
        self.rewards_int        = self.rewards_int.reshape((n, ))
        self.dones              = self.dones.reshape((n, ))
        self.returns_ext        = self.returns_ext.reshape((n, ))
        self.returns_int        = self.returns_int.reshape((n, ))
        self.advantages_ext     = self.advantages_ext.reshape((n, ))
        self.advantages_int     = self.advantages_int.reshape((n, ))



    def sample_batch(self, batch_size, device):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))

        states          = self.states[indices].to(device)
        logits          = self.logits[indices].to(device)
        
        actions         = self.actions[indices].to(device)
        
        returns_ext         = self.returns_ext[indices].to(device)
        returns_int         = self.returns_int[indices].to(device)

        advantages_ext      = self.advantages_ext[indices].to(device)
        advantages_int      = self.advantages_int[indices].to(device)
       
        return states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int

    def sample_states(self, batch_size, device):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))
        states          = self.states[indices].to(device)

        return states   

    def sample_batch_pairs(self, batch_size, max_distance, device):
        indices         = torch.randint(0, self.envs_count*(self.buffer_size - max_distance), size=(batch_size, ))
        distances       = torch.randint(0, max_distance+1, size=(batch_size, ))
        indices_next    = indices + self.envs_count*distances

        states_now      = self.states[indices].to(device)
        states_next     = self.states[indices_next].to(device)

        distances       = distances.to(device)

        print(distances)

        return states_now, states_next, distances
               

    def _gae(self, rewards, values, dones, gamma, lam, last_value=None):
        buffer_size = rewards.shape[0]
        envs_count  = rewards.shape[1]
                

        returns      = torch.zeros_like(rewards)
        advantages   = torch.zeros_like(rewards)

        if last_value is None:
            last_value = torch.zeros(envs_count, dtype=values.dtype)
        else:
            last_value = last_value.detach().squeeze(1).to(values.device)

        last_gae = torch.zeros(envs_count, dtype=values.dtype)

        for t in reversed(range(buffer_size)):
            if t == buffer_size - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            non_terminal = 1.0 - dones[t]

            delta       = (rewards[t] + gamma * next_value * non_terminal - values[t])
            last_gae    = (delta + gamma * lam * non_terminal * last_gae)

            advantages[t]   = last_gae
            returns[t]      = advantages[t] + values[t]

        return returns, advantages