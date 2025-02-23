import torch

class TrajectoryBufferIMContext:

    def __init__(self, buffer_size, state_shape, context_shape, actions_size, envs_count, dtype = torch.float32):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.context_shape  = context_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count

        self.dtype = dtype
      
        self.clear()   

        print("TrajectoryBufferIM")
        print("states : ", self.states.shape)
        print("context: ", self.context.shape)
        print("logits : ", self.logits.shape)
        print("\n")
    

    def add(self, state, context, logits, values_ext, values_int, actions, rewards_ext, rewards_int, dones):  
        self.states[self.ptr]       = state.detach().to(dtype=self.dtype, device="cpu").clone() 
        self.context[self.ptr]      = context.detach().to(dtype=self.dtype, device="cpu").clone() 
        self.logits[self.ptr]       = logits.detach().float().to(device="cpu").clone() 
        
        self.values_ext[self.ptr]   = values_ext.squeeze(1).detach().float().to(device="cpu").clone() 
        self.values_int[self.ptr]   = values_int.squeeze(1).detach().float().to(device="cpu").clone() 
        
        self.actions[self.ptr]      = torch.from_numpy(actions)     
        
        self.rewards_ext[self.ptr]  = torch.from_numpy(rewards_ext).float()
        self.rewards_int[self.ptr]  = torch.from_numpy(rewards_int).float()

        self.dones[self.ptr]            = torch.from_numpy(dones).float()
        
        self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True
        return False 
 

    def clear(self):    
        self.states     = torch.zeros((self.buffer_size, self.envs_count, ) + self.state_shape, dtype=self.dtype)
        self.context    = torch.zeros((self.buffer_size, self.envs_count, ) + self.context_shape, dtype=self.dtype)
        self.logits     = torch.zeros((self.buffer_size, self.envs_count, self.actions_size), dtype=torch.float32)
        
        self.values_ext = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        
        self.values_int = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)        

        self.actions    = torch.zeros((self.buffer_size, self.envs_count, ), dtype=int)
        
        self.rewards_ext    = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)
        self.rewards_int    = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)

        self.dones      = torch.zeros((self.buffer_size, self.envs_count, ), dtype=torch.float32)


        self.ptr = 0  
 
    def compute_returns(self, gamma_ext, gamma_int, lam = 0.95):
        self.returns_ext, self.advantages_ext = self._gae(self.rewards_ext, self.values_ext, self.dones, gamma_ext, lam)
        self.returns_int, self.advantages_int = self._gae(self.rewards_int, self.values_int, self.dones, gamma_int, lam)
        
        
        #reshape buffer for faster batch sampling
        self.states     = self.states.reshape((self.buffer_size*self.envs_count, ) + self.state_shape)
        self.context    = self.context.reshape((self.buffer_size*self.envs_count, ) + self.context_shape)

        self.logits     = self.logits.reshape((self.buffer_size*self.envs_count, self.actions_size))

        self.values_ext = self.values_ext.reshape((self.buffer_size*self.envs_count, ))
        self.values_int = self.values_int.reshape((self.buffer_size*self.envs_count, ))        
     
        self.actions    = self.actions.reshape((self.buffer_size*self.envs_count, ))
        

        self.rewards_ext = self.rewards_ext.reshape((self.buffer_size*self.envs_count, ))
        self.rewards_int = self.rewards_int.reshape((self.buffer_size*self.envs_count, ))

      
        self.dones         = self.dones.reshape((self.buffer_size*self.envs_count, ))

        self.returns_ext      = self.returns_ext.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_ext   = self.advantages_ext.reshape((self.buffer_size*self.envs_count, ))

        self.returns_int      = self.returns_int.reshape((self.buffer_size*self.envs_count, ))
        self.advantages_int   = self.advantages_int.reshape((self.buffer_size*self.envs_count, ))

        


    def sample_batch(self, batch_size, device, dtype = torch.float32):
        indices         = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))

        states          = self.states[indices].to(dtype=dtype, device=device)
        context         = self.context[indices].to(dtype=dtype, device=device)
        logits          = self.logits[indices].to(dtype=dtype, device=device)
        
        actions         = self.actions[indices].to(device=device)
         
        returns_ext     = self.returns_ext[indices].to(dtype=dtype, device=device)
        returns_int     = self.returns_int[indices].to(dtype=dtype, device=device)

        advantages_ext  = self.advantages_ext[indices].to(dtype=dtype, device=device)
        advantages_int  = self.advantages_int[indices].to(dtype=dtype, device=device)


        return states, context, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int
    

    def sample_state_pairs(self, batch_size, device, dtype = torch.float32):
        count           = self.buffer_size*self.envs_count

        indices_now     = torch.randint(0, self.envs_count*self.buffer_size, size=(batch_size, ))
        indices_next    = torch.clip(indices_now + self.envs_count, 0, count-1)

        states_now      = (self.states[indices_now]).to(dtype=dtype, device=device)
        states_next     = (self.states[indices_next]).to(dtype=dtype, device=device)

        actions         = (self.actions[indices_now]).to(device=device)   

        return states_now, states_next, actions
    


     
    def _gae(self, rewards, values, dones, gamma, lam):
        buffer_size = rewards.shape[0]
        envs_count  = rewards.shape[1]
        
        returns     = torch.zeros((buffer_size, envs_count), dtype=self.dtype)
        advantages  = torch.zeros((buffer_size, envs_count), dtype=self.dtype)

        last_gae    = torch.zeros((envs_count), dtype=self.dtype)
        
        for n in reversed(range(buffer_size-1)):
            delta           = rewards[n] + gamma*values[n+1]*(1.0 - dones[n]) - values[n]
            last_gae        = delta + gamma*lam*last_gae*(1.0 - dones[n])
            
            returns[n]      = last_gae + values[n]
            advantages[n]   = last_gae
 
        return returns.to(self.dtype), advantages.to(self.dtype)
