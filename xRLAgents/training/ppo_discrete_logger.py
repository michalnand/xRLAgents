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
        temporal_result = self._compute_temporal()
        return self.result_training  | temporal_result


    def _stats(self, value, prefix, sat_th, dp):
        result = {}
        result[prefix + "_mean"]       = round(torch.mean(value).item(), dp)
        result[prefix + "_std"]        = round(torch.std(value).item(), dp)
        result[prefix + "_mag_mean"]   = round(torch.mean(value**2).item(), dp)
        result[prefix + "_mag_std"]    = round(torch.std(value**2).item(), dp)

        saturation_ratio = (torch.abs(value) > sat_th).float().mean()
        result[prefix + "_saturation_ratio"]  = round(saturation_ratio.item(), dp)
    
        return result



    def _compute_temporal(self, dp):

        result = {}
        
        if len(self.actions_buffer) < 10:
            return result

        # [time, environments, action dimensions]
        actions = numpy.stack(self.actions_buffer, axis=0)
                
        total_elements = actions.size

        # 1. Relative counts histogram & normalized entropy
        unique_actions, counts = numpy.unique(actions, return_counts=True)
        relative_counts = counts / total_elements

        # Build dictionary: action_x : relative_count_of_action_x
        result_dict = {"action_"+ str(int(act)): round(float(rc), dp) for act, rc in zip(unique_actions, relative_counts)}

        # Compute Shannon Entropy: H = -sum(p * log2(p))
        entropy = -numpy.sum(relative_counts * numpy.log2(relative_counts))

        # Normalize entropy into [0, 1] range based on the number of unique actions observed
        num_unique = len(unique_actions)
        if num_unique <= 1:
            norm_entropy = 0.0
        else:
            max_entropy     = numpy.log2(num_unique)
            norm_entropy    = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        result_dict["actions_normalized_entropy"] = round(norm_entropy, dp)

        # 2. Consecutive matching ratios (cropping initial steps automatically via slicing)
        # m1: actions[n] == actions[n-1] -> shape (999, 128)
        m1 = actions[1:] == actions[:-1]
        ratio_1 = float(numpy.mean(m1)) 

        # m2: m1 and actions[n] == actions[n-2] -> shape (998, 128)
        m2 = m1[1:] & (actions[2:] == actions[:-2])
        ratio_2 = float(numpy.mean(m2))

        # m3: m2 and actions[n] == actions[n-3] -> shape (997, 128)
        m3 = m2[1:] & (actions[3:] == actions[:-3])
        ratio_3 = float(numpy.mean(m3))

        # m4: m3 and actions[n] == actions[n-4] -> shape (996, 128)
        m4 = m3[1:] & (actions[4:] == actions[:-4])
        ratio_4 = float(numpy.mean(m4))


        result_dict["actions_diff_1_ratio"] = round(ratio_1, dp)
        result_dict["actions_diff_2_ratio"] = round(ratio_2, dp)
        result_dict["actions_diff_3_ratio"] = round(ratio_3, dp)
        result_dict["actions_diff_4_ratio"] = round(ratio_4, dp)

        return result_dict