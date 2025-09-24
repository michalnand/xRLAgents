import torch 


class EpisodicBuffer():
    def __init__(self, buffer_size, top_n_count, shape):
        
        self.top_n_count = top_n_count
        self.buffer = torch.zeros((buffer_size, ) + shape, dtype=torch.float32)

        self.ptr = 0


    def step(self, x):
        # shape = (batch_size, buffer_size)
        similarity = self._similarity(x, self.buffer)
        
        scores, _  = similarity.max(dim=-1)

        # find least correlated items ins x
        _, indices = torch.sort(scores)

        # update buffer only with most relevant (least correlated)
        for n in range(self.top_n_count):
            idx = indices[n]
            self.buffer[self.ptr] = x[idx].detach().cpu().float()
            self.ptr = (self.ptr + 1)%self.buffer.shape[0]

        novelty = -scores
        return novelty


    def _similarity(self, a, b):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)

        cos_sim = a_norm @ b_norm.T

        return cos_sim
    




'''
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
'''