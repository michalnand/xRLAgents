import torch 
import numpy

  
class EpisodicBuffer():
    def __init__(self, buffer_size, n_envs, shape):
        
        self.buffer = torch.zeros((buffer_size, n_envs, ) + shape, dtype=torch.float32)

        self.ptr = 0


    def add(self, x):
        self.buffer[self.ptr] = x.detach().cpu().float()
        self.ptr = (self.ptr + 1)%self.buffer.shape[0]

    def reset(self, idx, initial_value = None):
        if initial_value is None:
            self.buffer[:, idx] = 0
        else:
            self.buffer[:, idx] = initial_value


    # x.shape = (n_envs, n_features)
    def compare(self, x, top_n):

        x = x.detach().cpu().float()

        # d.shape = (buffer_size, n_envs)
        d = ((self.buffer - x.unsqueeze(0))**2).mean(axis=-1)

        values_sorted, idx_sorted = torch.sort(d, dim=-1)

        top_values  = values_sorted[:top_n]   # (top_n, batch_size, n_envs)
        top_idx     = idx_sorted[:top_n]  

        mean_top_values = top_values.mean(dim=0)    

        return mean_top_values