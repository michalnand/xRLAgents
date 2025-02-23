import torch
import numpy

class FeaturesExtractor:    

    def __init__(self, downsample = 2):

        self.state_prev = None
        self.state_curr = None
        self.downsample = downsample

    def reset(self, env_id):
        self.state_prev[env_id] = 0.0
        self.state_curr[env_id] = 0.0

    def clear(self):    
        self.state_prev[:] = 0.0
        self.state_curr[:] = 0.0

    def step(self, states):
        if self.state_prev is None:
            self.state_prev = states.detach().clone().to(device = states.device, dtype = states.dtype)
        
        if self.state_curr is None:
            self.state_curr = states.detach().clone().to(device = states.device, dtype = states.dtype)

        self.state_prev = self.state_curr.clone()
        self.state_curr = states.clone()


        diff = self.state_curr - self.state_prev

        z = torch.abs(diff)
        z = torch.nn.functional.avg_pool2d(z, (self.downsample, self.downsample), stride=(self.downsample, self.downsample))
        z = z.flatten(1)
        
        return z


class EMAOutlierDetector:
    def __init__(self, alpha = 0.1, epsilon=1e-6):
        self.ema_mean   = None
        self.ema_var    = None
        self.alpha      = alpha  # EMA smoothing factor
        self.epsilon    = epsilon  # Stability constant

    def step(self, x):
        """
        Updates EMA statistics and computes Z-score for outlier detection.
        :param d: Current measured distances (batch_size,)
        :return: z-scores, updated mean and variance
        """
        if self.ema_mean is None:
            self.ema_mean   = x.to(device=x.device, dtype=x.dtype)
            self.ema_var    = torch.ones(x.shape, device=x.device, dtype=x.dtype)
        

        # Update EMA mean
        self.ema_mean   = (1.0 - self.alpha)*self.ema_mean + self.alpha*x

        # Update EMA variance
        diff            = x - self.ema_mean
        self.ema_var    = (1.0 - self.alpha)*self.ema_var + self.alpha*(diff**2)

        # Compute Z-score
        z_score = torch.abs(diff) / (torch.sqrt(self.ema_var) + self.epsilon)

        #print(diff.shape, self.ema_mean, self.ema_var, z_score)
        return z_score


class EpisodicGoalsBufferStats:

    def __init__(self, buffer_size, batch_size, state_shape, add_threshold = 2.5, min_dist = 0.02, device = "cpu", dtype=torch.bfloat16):

        self.device = device
        self.dtype  = dtype
        self.fe     = FeaturesExtractor()

        dummy_state = torch.randn(state_shape).to(self.device)
        features    = self.fe.step(dummy_state)
        n_features  = features.shape[-1]

        self.features  = torch.zeros((batch_size, buffer_size, n_features)).to(device = self.device, dtype = self.dtype)

        self.key_states = torch.zeros((batch_size, buffer_size, ) + state_shape).to(device = self.device, dtype = self.dtype)
        
        self.ptrs       = torch.zeros((batch_size, ), dtype=int).to(self.device)

    
        self.z_score_est = EMAOutlierDetector() 

        self.buffer_size = buffer_size
        self.fe.clear()

        self.add_threshold = add_threshold
        self.min_dist      = min_dist

        print("EpisodicGoalsBuffer")
        print("features     : ", self.features.shape)
        print("key_states   : ", self.key_states.shape)
        print("\n")


    def reset(self, env_id):
        #self.fe.reset(env_id)

        self.features[env_id,  :]    = 0.0
        self.ptrs[env_id]            = 0

    def step(self, states):
        states_tmp  = states.to(device = self.device, dtype = self.dtype)
        
        batch_size  = states_tmp.shape[0]
        #features    = self.fe.step(states_tmp)


        #dist = features.unsqueeze(1) - self.features
        #dist = torch.abs(dist).mean(dim=-1)

        #dist_min_idx    = torch.argmin(dist, dim=1)
        #dist_min_value  = torch.min(dist, dim=1)[0]


        #z_score = self.z_score_est.step(dist_min_value)

        rewards = numpy.zeros((batch_size, ))
       
        '''
        for n in range(batch_size):                        
            # if z-scroe is high
            if z_score[n] > self.add_threshold and dist_min_value[n] > self.min_dist:
                # circular buffer
                idx = self.ptrs[n]%self.buffer_size

                self.features[n, idx]    = features[n].detach()
                self.key_states[n, idx]  = states_tmp[n].detach()

                self.ptrs[n]+= 1

                # new key state discovered, generate reward
                rewards[n] = 1.0
        '''

                
        
        
        # statistics for log
        stats = {}
        tmp = self.ptrs.float().cpu().detach().numpy()
        stats["mean"] = tmp.mean()
        stats["std"]  = tmp.std()
        stats["max"]  = tmp.max()

        #stats["dist_mean"]      = dist.mean().float().cpu().detach().numpy().item()
        #stats["dist_std"]       = dist.std().float().cpu().detach().numpy().item()

        #stats["dist_min_mean"]  = dist_min_value.mean().float().cpu().detach().numpy().item()
        #stats["dist_min_std"]   = dist_min_value.std().float().cpu().detach().numpy().item()

        #stats["z_mean"]  = z_score.mean().float().cpu().detach().numpy().item()
        #stats["z_std"]   = z_score.std().float().cpu().detach().numpy().item()

        return self.key_states, rewards, stats

