import torch

class StateNormalization:

    def __init__(self, mode, states_initial):
        self.mode = mode

        self.device     = states_initial.device
        self.state_mean = torch.from_numpy(states_initial).mean(dim=0).to(self.device).unsqueeze(0)
        self.state_var  = torch.ones(self.state_mean.shape, device=self.device)
        

    def __call__(self, states):
        self._update_normalisation(states)
                
        if self.mode == "ema":
            states_result = self._states_normalise_ema(states)
        elif self.mode == "diff":
            states_result = self._states_normalize_diff(states)
        elif self.mode == "diff_ema":
            states_result = self._states_normalize_diff_ema(states)
        elif self.mode == "none":
            states_result = states
        else:
            raise ValueError("Unsupported state normalization " + str(self.mode))

        return states_result

     
    #update running stats when training enabled
    def _update_normalisation(self, states, alpha = 0.99):
        self.state_mean = alpha*self.state_mean + (1.0 - alpha)*states.mean(dim=0).unsqueeze(0)
        self.state_var  = alpha*self.state_var + (1.0 - alpha)*states.var(dim=0).unsqueeze(0) 

        print("stats ", self.state_mean.shape, self.state_var.shape)

    #normalise mean and variance
    def _states_normalise_ema(self, states):     
        states_norm = (states - self.state_mean)/(torch.sqrt(self.state_var) + 10**-6)
        states_norm = torch.clip(states_norm, -4.0, 4.0)

        print("norm ", self.states_norm.shape)
        return states_norm  

    def _states_normalize_diff(self, states):
        anchor = states[:, 0, :, :].unsqueeze(1)

        past_frames = states[:, 1:, :, :]
        differences = past_frames - anchor
        result = torch.cat([anchor, differences], dim=1)
    
        return result

    def _states_normalize_diff_ema(self, states):
        states_ema = self._states_normalise_ema(states)

        anchor = states_ema[:, 0, :, :].unsqueeze(1)
    
        past_frames = states_ema[:, 1:, :, :]
        differences = past_frames - anchor
        result = torch.cat([anchor, differences], dim=1)
    
        return result

