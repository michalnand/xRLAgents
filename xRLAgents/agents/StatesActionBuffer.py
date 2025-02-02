import torch



class StatesActionBuffer:

    def __init__(self, buffer_size, state_shape, n_envs):

        self.buffer_size = buffer_size
        self.n_envs      = n_envs
        self.state_shape = state_shape

        self.clear()

        print("creating StatesActionBuffer")
        print("states  : ", self.states.shape)
        print("actions : ", self.actions.shape)
        print("\n")
        
    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True
        return False 
    
    def clear(self):
        self.states   = torch.zeros((self.buffer_size, self.n_envs, ) + self.state_shape, dtype=torch.float32)
        self.actions  = torch.zeros((self.buffer_size, self.n_envs, ), dtype=int)

        self.ptr = 0

    def add(self, state, actions):
        self.states[self.ptr]  = state.detach().to("cpu").clone() 
        self.actions[self.ptr] = torch.from_numpy(actions)
        
        self.ptr+= 1

    def sample_state_pairs(self, batch_size, device):
        count           = self.buffer_size*self.n_envs  

        indices_now     = torch.randint(0, self.n_envs*self.buffer_size, size=(batch_size, ))
        indices_next    = torch.clip(indices_now + self.n_envs, 0, count-1)

        states_now      = (self.states[indices_now]).to(device)
        states_next     = (self.states[indices_next]).to(device)

        actions         = (self.actions[indices_now]).to(device)   

        return states_now, states_next, actions

