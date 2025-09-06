import torch

class TrajectoryBufferIM:

    def __init__(self, buffer_size, n_envs, device="cpu"):
        self.buffer_size    = buffer_size
        self.n_envs         = n_envs
        self.device         = device

        self.clear()   

    def clear(self):
        self.buffer = {}
        self.ptr    = 0
    
    def add(self, **kwargs):
        if self.is_full():
            return False
        
        # buffer not initialised
        if not(self.buffer):
            # first call: preallocate tensors based on provided keys
            for key, value in kwargs.items():
                value = torch.as_tensor(value, device=self.device)
                shape = (self.buffer_size, self.n_envs) + value.shape[1:]

                self.buffer[key] = torch.zeros(shape, device=self.device, dtype=value.dtype)

                #print("new item ", str(key), self.buffer[key].shape, self.buffer[key].dtype)


        # add values into buffer
        for key, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=self.device).detach()
            else:
                value = value.to(self.device).detach()

            self.buffer[key][self.ptr].copy_(value)

        self.ptr = self.ptr + 1

        return True

    
    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True
        return False 

    
    def compute_returns(self, gamma, lam = 0.95):
        rewards_ext = self.buffer["rewards_ext"]
        values_ext  = self.buffer["values_ext"]

        rewards_int = self.buffer["rewards_int"]
        values_int  = self.buffer["values_int"]

        dones   = self.buffer["dones"].float()

        # compute returns and advantages using gae
        returns_ext, advantages_ext = self._gae(rewards_ext, values_ext, dones, gamma, lam)
        returns_int, advantages_int = self._gae(rewards_int, values_int, dones, gamma, lam)

        self.buffer["returns_ext"]    = returns_ext.to(values_ext.dtype)
        self.buffer["advantages_ext"] = advantages_ext.to(values_ext.dtype)

        self.buffer["returns_int"]    = returns_int.to(values_int.dtype)
        self.buffer["advantages_int"] = advantages_int.to(values_int.dtype)
        

        #reshape buffer for faster batch sampling
        total_size = self.buffer_size*self.n_envs
        for key, arr in self.buffer.items():
            flatten = arr.reshape(total_size, *arr.shape[2:])
            self.buffer[key] = flatten


    def sample_batch(self, batch_size, device):
        total_size = self.buffer_size*self.n_envs
        idx = torch.randint(0, total_size, (batch_size,), device=self.device)

        batch = {}
        for key, arr in self.buffer.items():
            batch[key] = arr[idx].to(device)

        return batch


    def sample_states(self, batch_size, device):
        total_size = self.buffer_size*self.n_envs
        idx = torch.randint(0, total_size, (batch_size,), device=self.device)

        result = self.buffer["states"][idx].to(device)

        return result
    

    def sample_states_seq(self, batch_size, time_distances, device, dtype = None):
        if dtype is None:
            dtype = torch.float32

        total_size = self.buffer_size*self.n_envs

        states_result = []
        labels_result = []      

        indices_now = torch.randint(0, self.n_envs*self.buffer_size, size=(batch_size, ))

        for n in range(len(time_distances)):
            d_max   = time_distances[n]
            indices = indices_now + self.n_envs*torch.randint(0, d_max + 1, size=(batch_size, ))
            indices = torch.clip(indices, 0, total_size-1)

            states = (self.states[indices]).to(dtype=dtype, device=device)
            labels = (self.labels[indices]).to(device=device) 

            states_result.append(states)
            labels_result.append(labels)

       
        return states_result, labels_result
      
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
