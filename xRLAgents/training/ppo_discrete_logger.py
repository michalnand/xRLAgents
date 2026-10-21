import torch
import numpy

class PPODiscreteLogger:

    def __init__(self, buffer_size = 256):
        self.result_training = {}
        self.result_buffer   = {}

        self.actions_buffer = []
        self.buffer_size    = buffer_size

    # called every agent step (will other variables history helps ? - mu, var, value ?)
    def update_step(self, actions):
        self.actions_buffer.append(numpy.array(actions))

        # remove first
        if len(self.actions_buffer) >= self.buffer_size:
            self.actions_buffer.pop(0)


    # called after training episodes done (every 128 steps)
    def udpate_training(self, actions, values, returns, logits_old, logits_new, loss_actor, loss_critic, loss_entropy, dp=6):
        self.result_training = {}
        self.result_training = self.result_training | self._stats(actions.float(), "actions", 10.0, dp)
        self.result_training = self.result_training | self._stats(values, "values", 10.0, dp)
        self.result_training = self.result_training | self._stats(returns, "returns", 4.0, dp) 
        self.result_training = self.result_training | self._stats(logits_new , "logits", 10.0, dp)
        self.result_training = self.result_training | self._stats(logits_new - logits_old , "logits_diff", 10.0, dp)


        self.result_training["loss_actor"]   = round(torch.mean(loss_actor).item(), dp)
        self.result_training["loss_critic"]  = round(torch.mean(loss_critic).item(), dp)
        self.result_training["loss_entropy"] = round(torch.mean(loss_entropy).item(), dp)



    def get(self):
        # compute may contain more complicated computation, get is called every 128 or 1024 steps
        return self.result_training 


    def _stats(self, value, prefix, sat_th, dp):
        result = {}
        result[prefix + "_mean"]       = round(torch.mean(value).item(), dp)
        result[prefix + "_std"]        = round(torch.std(value).item(), dp)
        result[prefix + "_mag_mean"]   = round(torch.mean(value**2).item(), dp)
        result[prefix + "_mag_std"]    = round(torch.std(value**2).item(), dp)

        saturation_ratio = (torch.abs(value) > sat_th).float().mean()
        result[prefix + "_saturation_ratio"]  = round(saturation_ratio.item(), dp)
    
        return result
