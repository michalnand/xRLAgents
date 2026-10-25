import torch

class TrajectoryBufferDiscrete:

    def __init__(self, buffer_size, state_shape, num_actions, envs_count):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.num_actions    = num_actions
        self.envs_count     = envs_count
      
        self.clear()   
    

    def add(self, states, logits, values, actions, rewards, dones):  
        self.states[self.ptr]       = states.detach().to("cpu").clone() 
        self.logits[self.ptr]       = logits.detach().to("cpu").clone() 
        self.values[self.ptr]       = values.squeeze(1).detach().to("cpu").clone() 
        
        self.actions[self.ptr]      = torch.from_numpy(actions)   

        self.reward[self.ptr]       = torch.from_numpy(rewards)
        self.dones[self.ptr]        = torch.from_numpy(dones).float()
        
        self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True
        return False 
 

    def clear(self):
        self.states      = torch.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=torch.float32)
        self.logits      = torch.zeros((self.buffer_size, self.envs_count, self.num_actions), dtype=torch.float32)
        self.values      = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        
        self.actions     = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.long)
        self.reward      = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.dones       = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.ptr = 0  
 
 
    def compute_returns(self, gamma, lam = 0.95, last_value=None):
        self.returns, self.advantages   = self._gae(self.reward, self.values, self.dones, gamma, lam, last_value)
        
        # normalize advantages over the whole buffer
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        #reshape buffer for faster batch sampling
        n = self.buffer_size*self.envs_count

        self.states      = self.states.reshape((n, ) + self.state_shape)
        self.logits      = self.logits.reshape((n, self.num_actions) )
        self.values      = self.values.reshape((n, ))
        self.actions     = self.actions.reshape((n, ))
        self.reward      = self.reward.reshape((n, ))
        self.dones       = self.dones.reshape((n, ))
        self.returns     = self.returns.reshape((n, ))
        self.advantages  = self.advantages.reshape((n, ))



    def sample_batch(self, batch_size, device):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))

        states          = self.states[indices].to(device)
        logits          = self.logits[indices].to(device)
        
        actions         = self.actions[indices].to(device)
        
        returns         = self.returns[indices].to(device)
        advantages      = self.advantages[indices].to(device)
       
        return states, logits, actions, returns, advantages


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