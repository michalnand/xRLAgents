import torch
import numpy

class PPOContinuousLogger:

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
    def udpate_training(self, values, returns, mu, var, loss_actor, loss_critic, loss_entropy, dp=6):
        self.result_training = {}
        self.result_training = self.result_training | self._stats(values, "values", 10.0, dp)
        self.result_training = self.result_training | self._stats(returns, "returns", 4.0, dp) 
        self.result_training = self.result_training | self._stats(mu, "actions",    0.95, dp)

        self.result_training = self.result_training | self._stats(var, "var",    0.95, dp)

        self.result_training["loss_actor"]   = round(torch.mean(loss_actor).item(), dp)
        self.result_training["loss_critic"]  = round(torch.mean(loss_critic).item(), dp)
        self.result_training["loss_entropy"] = round(torch.mean(loss_entropy).item(), dp)
        


    def get(self):
        # compute may contain more complicated computation, get is called every 128 or 1024 steps
        self.result_buffer = self._compute_temporal()
        return self.result_training | self.result_buffer


    def _stats(self, value, prefix, sat_th, dp):
        result = {}
        result[prefix + "_mean"]       = round(torch.mean(value).item(), dp)
        result[prefix + "_std"]        = round(torch.std(value).item(), dp)
        result[prefix + "_mag_mean"]   = round(torch.mean(value**2).item(), dp)
        result[prefix + "_mag_std"]    = round(torch.std(value**2).item(), dp)

        saturation_ratio = (torch.abs(value) > sat_th).float().mean()
        result[prefix + "_saturation_ratio"]  = round(saturation_ratio.item(), dp)
    
        return result


    def _compute_temporal(self, dp=6):
        result = {}

        if len(self.actions_buffer) < 2:
            return result

        # [time, environments, action dimensions]
        actions = numpy.stack(self.actions_buffer, axis=0)

        # Temporal differences: [time-1, environments, action dimensions]
        delta = numpy.diff(actions, axis=0)
        abs_delta = numpy.abs(delta)

        result["action_delta_mean"] = round(float(abs_delta.mean()), dp)
        result["action_delta_rms"] = round(float(numpy.sqrt(numpy.mean(delta ** 2))), dp)
        result["action_delta_p95"] = round(float(numpy.percentile(abs_delta, 95)), dp)

        # Fraction of consecutive action pairs that change sign.
        # Ignore pairs where either action is near zero.
        eps = 1e-5
        valid = (numpy.abs(actions[:-1]) > eps) & (numpy.abs(actions[1:]) > eps)
        flips = (actions[:-1] * actions[1:]) < 0

        result["action_sign_flip_ratio"] = round(float(flips[valid].mean()) if valid.any() else 0.0, dp)

        # Correlation between consecutive actions, pooled over
        # time and environments separately for each action dimension.
        correlations = []
        for j in range(actions.shape[-1]):
            x = actions[:-1, :, j].reshape(-1)
            y = actions[1:, :, j].reshape(-1)

            if x.size > 1 and x.std() > eps and y.std() > eps:
                correlations.append(numpy.corrcoef(x, y)[0, 1])

        result["action_lag1_corr"] = round(float(numpy.mean(correlations)) if correlations else 0.0, dp)

        # Per-dimension temporal variability, then average.
        result["action_temporal_std"] = round(float(actions.std(axis=0).mean()), dp)

        return result