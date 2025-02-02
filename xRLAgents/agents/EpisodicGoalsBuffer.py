import torch

class FeaturesExtractor:    

    def __init__(self, batch_size, state_shape, n_frames):

        self.n_frames   = n_frames
        self.buffer     = torch.zeros((n_frames, batch_size, ) + state_shape, dtype=torch.float32)

    def reset(self, env_id):
        self.buffer[:, env_id] = 0.0

    def clear(self):
        self.buffer[:] = 0.0

    def step(self, states):
        self.buffer     = torch.roll(self.buffer, 1, dims=0) 
        self.buffer[0]  = states.detach().clone()

        a = self.buffer[0:self.n_frames//2]
        b = self.buffer[self.n_frames//2:]

        diff = a.mean(dim=0) - b.mean(dim=0)
        z = torch.abs(diff)
        z = torch.nn.functional.avg_pool2d(z, (2, 2), stride=(2, 2))
        z = z.flatten(1)
        
        return z




class EpisodicGoalsBuffer:

    def __init__(self, buffer_size, batch_size, state_shape, n_frames = 2, alpha = 0.1, add_threshold = 0.85):

        self.fe = FeaturesExtractor(batch_size, state_shape, n_frames)

        dummy_state = torch.randn(state_shape)
        features    = self.fe.step(dummy_state)
        n_features  = features.shape[-1]

        self.features_mu  = torch.zeros((batch_size, buffer_size, n_features))
        self.features_var = torch.ones((batch_size, buffer_size, n_features))

        self.key_states = torch.zeros((batch_size, buffer_size, ) + state_shape)
        self.ptrs       = torch.zeros((batch_size, ), dtype=int)

        self.buffer_size = buffer_size
        self.fe.clear()

        self.alpha         = alpha
        self.add_threshold = add_threshold

        print("EpisodicGoalsBuffer")
        print("features_mu  : ", self.features_mu.shape)
        print("features_var : ", self.features_var.shape)
        print("key_states   : ", self.key_states.shape)
        print("\n")


    def reset(self, env_id):
        self.fe.reset(env_id)

        self.features_mu[env_id,  :] = 0.0
        self.features_var[env_id, :] = 1.0

    def step(self, states):
        batch_size  = states.shape[0]
        features    = self.fe.step(states.to("cpu"))


        dist = features.unsqueeze(1) - self.features_mu
        dist = torch.abs(dist).mean(dim=-1)

    
        #dist_min_v   = torch.min(dist, dim=1)[0]
        dist_min_idx = torch.argmin(dist, dim=1)

     
        rewards = torch.zeros((batch_size, ))
        refresh_indices = -torch.ones((batch_size, ), dtype=int)
        for n in range(batch_size):

            # update stats for nearest
            idx = dist_min_idx[n]
            
            # mean and variance update
            delta = features[n] - self.features_mu[n][idx]
            self.features_mu[n][idx]  = (1.0 - self.alpha)*self.features_mu[n][idx]  + self.alpha*features[n]
            self.features_var[n][idx] = (1.0 - self.alpha)*self.features_var[n][idx] + self.alpha*(delta**2)

            # compute confidence using z-score
            features_sigma  = (self.features_var[n][idx] + 0.1)**0.5
            z_score         = (features[n] - self.features_mu[n][idx])/features_sigma
            confidence      = 1.0 - ((z_score**2).mean()**0.5)

            # if confidence is low, create new goal state
            if confidence < self.add_threshold:
                idx = self.ptrs[n]
                self.features_mu[n, idx]    = features[n]
                self.key_states[n, idx]     = states[n]

                refresh_indices[n] = idx

                self.ptrs[n] = (self.ptrs[n] + 1)%self.buffer_size

                # new key state discovered, generate reward
                rewards[n] = 1.0

        return self.key_states, rewards, refresh_indices
