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

    def __init__(self, buffer_size, batch_size, state_shape, n_frames = 2, alpha = 0.1, add_threshold = 0.9, device = "cpu"):

        self.device = device
        self.fe = FeaturesExtractor(batch_size, state_shape, n_frames)

        self.downsample = int(buffer_size**0.5)

        dummy_state = torch.randn(state_shape)
        features    = self.fe.step(dummy_state)
        n_features  = features.shape[-1]

        self.features_mu  = torch.zeros((batch_size, buffer_size, n_features)).to(self.device)
        self.features_var = torch.ones((batch_size, buffer_size, n_features)).to(self.device)

        self.key_states = torch.zeros((batch_size, buffer_size, ) + state_shape).to(self.device)
        
        grid_size = int(buffer_size ** 0.5)
        self.tiled_state = torch.zeros((batch_size, state_shape[0], (grid_size*state_shape[1])//self.downsample, (grid_size*state_shape[2])//self.downsample)).to(self.device)

        self.ptrs       = torch.zeros((batch_size, ), dtype=int).to(self.device)

        self.buffer_size = buffer_size
        self.fe.clear()

        self.alpha         = alpha
        self.add_threshold = add_threshold

        print("EpisodicGoalsBuffer")
        print("features_mu  : ", self.features_mu.shape)
        print("features_var : ", self.features_var.shape)
        print("tiled_state  : ", self.tiled_state.shape)
        print("key_states   : ", self.key_states.shape)
        print("\n")


    def reset(self, env_id):
        self.fe.reset(env_id)

        self.features_mu[env_id,  :] = 0.0
        self.features_var[env_id, :] = 1.0
        self.tiled_state[env_id]     = 0.0
        
        self.ptrs[env_id]            = 0

    def step(self, states):
        batch_size  = states.shape[0]
        features    = self.fe.step(states.to(self.device))

        print(features.device,  self.features_mu.device)

        dist = features.unsqueeze(1) - self.features_mu
        dist = torch.abs(dist).mean(dim=-1)

    
        #dist_min_v   = torch.min(dist, dim=1)[0]
        dist_min_idx = torch.argmin(dist, dim=1)

     
        rewards = torch.zeros((batch_size, )).to(self.device)
        refresh_indices = -torch.ones((batch_size, ), dtype=int).to(self.device)
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

                self.tiled_state[n] = self._downsample_and_tile(self.key_states[n], self.downsample)

                self.ptrs[n] = (self.ptrs[n] + 1)%self.buffer_size

                # new key state discovered, generate reward
                rewards[n] = 1.0

        stats = {}
        tmp = self.ptrs.float().cpu().detach().numpy()
        stats["mean"] = tmp.mean()
        stats["std"]  = tmp.std()
        stats["max"]  = tmp.max()

        return self.key_states, self.tiled_state, rewards, refresh_indices, stats


    def _downsample_and_tile(self, key_states, scale):
        buffer_size, ch, height, width = key_states.shape
        grid_size = int(buffer_size ** 0.5)
        
        # Downsample using average pooling
        downsampled = torch.nn.functional.avg_pool2d(key_states.view(-1, ch, height, width), kernel_size=scale, stride=scale)
        _, _, new_height, new_width = downsampled.shape


        tiled = downsampled.view(grid_size, grid_size, ch, new_height, new_width)  # (s, s, ch, h, w)
        tiled = tiled.permute(2, 0, 3, 1, 4)  # (ch, s, h, s, w)
        tiled = tiled.reshape(ch, grid_size * new_height, grid_size * new_width)  # (ch, height*s, width*s)

        return tiled

       