import torch

class TrajectoryBuffer:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
      
        self.clear()   
    

    def add(self, state, logits, value, actions, reward, dones, hidden_state = None):  
        self.states[self.ptr]    = state.detach().to("cpu").clone() 
        self.logits[self.ptr]    = logits.detach().to("cpu").clone() 
        self.values[self.ptr]    = value.squeeze(1).detach().to("cpu").clone() 
        self.actions[self.ptr]   = torch.from_numpy(actions)
        
        self.reward[self.ptr]    = torch.from_numpy(reward)
        self.dones[self.ptr]     = torch.from_numpy(dones).float()

        if hidden_state is not None:
            if self.hidden_states is None:
                self.hidden_shape = hidden_state.shape[1:]
                self.hidden_states = torch.zeros((self.buffer_size, ) + hidden_state.shape, dtype=torch.float32) 
            
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

        self.hidden_states = None

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

        if self.hidden_states is not None:
            self.hidden_states  = self.hidden_states.reshape((self.buffer_size*self.envs_count, ) + self.hidden_shape)
   

    def sample_batch(self, batch_size, device):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))

        states          = self.states[indices].to(device)
        logits          = self.logits[indices].to(device)
        
        actions         = self.actions[indices].to(device)
        
        returns         = self.returns[indices].to(device)
        advantages      = self.advantages[indices].to(device)

       
        return states, logits, actions, returns, advantages
    

    def sample_rnn_batch(self, batch_size, steps, device):
        indices         = torch.randint(0, self.envs_count*(self.buffer_size - steps), size=(batch_size, ))

        states          = []
        logits          = []
        actions         = []
        returns         = []
        advantages      = []
        hidden_states   = []

        for n in range(steps):
            idx = indices + n*self.envs_count

            states.append(self.states[idx].to(device))
            logits.append(self.logits[idx].to(device))
            actions.append(self.actions[idx].to(device))
            returns.append(self.returns[idx].to(device))
            advantages.append(self.advantages[idx].to(device))
            hidden_states.append(self.hidden_states[idx].to(device))

        states          = torch.stack(states, dim=0)
        logits          = torch.stack(logits, dim=0)
        actions         = torch.stack(actions, dim=0)
        returns         = torch.stack(returns, dim=0)
        advantages      = torch.stack(advantages, dim=0)
        hidden_states   = torch.stack(hidden_states, dim=0)

        return states, logits, actions, returns, advantages, hidden_states
    

    def sample_state_pairs(self, batch_size, device):

        count           = self.buffer_size*self.envs_count

        indices_now     = torch.randint(0, count, size=(batch_size, ))
        indices_next    = torch.clip(indices_now + self.envs_count, 0, count-1)

        states          = self.states[indices_now].to(device)
        states_next     = self.states[indices_next].to(device)

        actions         = (self.actions[indices_now]).to(device=device) 
        
        return states, states_next, actions
    
     
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
