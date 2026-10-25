import torch

class TrajectoryBufferRNN:

    def __init__(self, buffer_size, state_shape, hidden_shape, actions_size, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.hidden_shape   = hidden_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
      
        self.clear()   
    

    def add(self, state, logits, value, actions, reward, dones, hidden_state):  
        self.states[self.ptr]    = state.detach().to("cpu").clone() 
        self.logits[self.ptr]    = logits.detach().to("cpu").clone() 
        self.values[self.ptr]    = value.squeeze(1).detach().to("cpu").clone() 
        self.actions[self.ptr]   = torch.from_numpy(actions)
        
        self.reward[self.ptr]    = torch.from_numpy(reward)
        self.dones[self.ptr]     = torch.from_numpy(dones).float()

        self.hidden_states[self.ptr] = hidden_state.detach().to("cpu").clone() 
        
        self.ptr = self.ptr + 1 

    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True
        return False 
 
    def clear(self):
        self.states     = torch.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=torch.float32)
        self.logits     = torch.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=torch.float32)
        self.values     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        
        self.actions    = torch.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        self.reward     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.dones      = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.hidden_states = torch.zeros((self.buffer_size, self.envs_count, ) + self.hidden_shape, dtype=torch.float32)

        self.ptr = 0  
 
    def compute_returns(self, gamma, lam = 0.95):
        self.returns, self.advantages   = self._gae(self.reward, self.values, self.dones, gamma, lam)
        
        #reshape buffer for faster batch sampling
        self.states     = self.states.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.logits     = self.logits.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.values     = self.values.reshape((self.buffer_size*self.envs_count, ))        
     
        self.actions    = self.actions.reshape((self.buffer_size*self.envs_count, ))
        
        self.reward     = self.reward.reshape((self.buffer_size*self.envs_count, ))
      
        self.dones      = self.dones.reshape((self.buffer_size*self.envs_count, ))

        self.returns    = self.returns.reshape((self.buffer_size*self.envs_count, ))
        self.advantages = self.advantages.reshape((self.buffer_size*self.envs_count, ))

        self.hidden_states = self.hidden_states.reshape((self.buffer_size*self.envs_count, ) + self.hidden_shape)
   

    def sample_batch(self, seq_length, batch_size, device):
        indices        = torch.randint(0, self.envs_count*(self.buffer_size - seq_length), size=(batch_size, ))
        
        states         = torch.zeros((seq_length, batch_size, ) + self.state_shape,  dtype=torch.float32, device=device)
        hidden_states  = torch.zeros((seq_length, batch_size, ) + self.hidden_shape, dtype=torch.float32, device=device)

        logits         = torch.zeros((seq_length, batch_size, self.actions_size), dtype=torch.float32, device=device)
        actions        = torch.zeros((seq_length, batch_size), dtype=int, device=device)

        returns        = torch.zeros((seq_length, batch_size), dtype=torch.float32, device=device)
        advantages     = torch.zeros((seq_length, batch_size), dtype=torch.float32, device=device)

        for n in range(seq_length):
            states[n]        = self.states[indices].to(device)
            hidden_states[n] = self.hidden_states[indices].to(device)
            
            logits[n]     = self.logits[indices].to(device)
            
            actions[n]    = self.actions[indices].to(device)
            
            returns[n]    = self.returns[indices].to(device)
            advantages[n] = self.advantages[indices].to(device)

            indices+= self.envs_count 

        return states, logits, actions, returns, advantages, hidden_states
    
    
     
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
