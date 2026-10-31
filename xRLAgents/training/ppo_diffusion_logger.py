import torch
import numpy

class PPODiffusionLogger:

    def __init__(self, num_envs, num_actions, num_bins = 512, num_spectral_samples = 8192):
        self.num_envs       = num_envs
        self.num_actions    = num_actions
        self.num_bins       = num_bins
        self.num_spectral_samples = num_spectral_samples

        self.curr_episode_ptr = numpy.zeros((num_envs, ), dtype=int)

        self.result_ppo   = {}
        self.result_ssl   = {}
        self.result_im    = {}

        self.actions_buffer     = []
        self.rewards_ext_buffer = []
        self.rewards_int_buffer = []

        for n in range(self.num_envs):
            self.actions_buffer.append([])
            self.rewards_ext_buffer.append([])
            self.rewards_int_buffer.append([])

        self.rewards_ext_stats = numpy.zeros((num_bins, ))
        self.rewards_int_stats = numpy.zeros((num_bins, ))
        self.actions_stats     = numpy.zeros((num_bins, num_actions))

        self.rewards_episode_ext_curr = numpy.zeros((num_envs, ))
        self.rewards_episode_ext      = numpy.zeros((num_envs, ))
        self.episodes      = numpy.zeros((num_envs, ))

        self.result_summary  = {}
        self.result_ppo  = {}
        self.result_ssl  = {}
        self.result_im  = {}


        self.spectral_samples = None 
        self.spectral_ptr     = 0

        self.k_coeff = 0.1


    # called every agent step (will other variables history helps ? - mu, var, value ?)
    def update_step(self, actions, rewards_ext, rewards_int, dones, z):
        for n in range(self.num_envs):
            self.rewards_ext_buffer[n].append(rewards_ext[n])
            self.rewards_int_buffer[n].append(rewards_int[n])
            self.actions_buffer[n].append(actions[n])


        dones_idx = numpy.where(dones)[0]

        if self.spectral_samples is None:
            num_features          = z.shape[-1]
            self.spectral_samples = numpy.zeros((self.num_spectral_samples, num_features))

        for n in range(z.shape[0]):
            self.spectral_samples[self.spectral_ptr] = z[n]
            self.spectral_ptr = (self.spectral_ptr+1)%self.num_spectral_samples

        self.rewards_episode_ext_curr+= rewards_ext

        for n in dones_idx:

            if len(self.rewards_ext_buffer[n]) > 32:
                tmp = self._map_list_to_histogram(self.rewards_ext_buffer[n], self.num_bins)
                self.rewards_ext_stats = (1.0 - self.k_coeff)*self.rewards_ext_stats + self.k_coeff*tmp

                tmp = self._map_list_to_histogram(self.rewards_int_buffer[n], self.num_bins)
                self.rewards_int_stats = (1.0 - self.k_coeff)*self.rewards_int_stats + self.k_coeff*tmp

                tmp = self._list_to_action_histogram(self.actions_buffer[n], num_actions=self.num_actions, num_bins=self.num_bins)
                self.actions_stats = (1.0 - self.k_coeff)*self.actions_stats + self.k_coeff*tmp



            self.rewards_ext_buffer[n]  = []
            self.rewards_int_buffer[n]  = []
            self.actions_buffer[n]      = []

            self.rewards_episode_ext = numpy.array(self.rewards_episode_ext_curr)
            self.rewards_episode_ext_curr[n] = 0
            self.episodes[n]+= 1

        self.result_summary = {}
        self.result_summary["episodes_mean"]    = float(round(self.episodes.mean(), 6))
        self.result_summary["episodes_std"]     = float(round(self.episodes.std(), 6))
        self.result_summary["rewards_ext_mean"] = float(round(rewards_ext.mean(), 6))
        self.result_summary["rewards_ext_std"]  = float(round(rewards_ext.std(), 6))
        
        self.result_summary["rewards_ext_episode_mean"] = float(round((self.rewards_episode_ext).mean(), 6))
        self.result_summary["rewards_ext_episode_std"]  = float(round((self.rewards_episode_ext).std(), 6))
        self.result_summary["rewards_ext_episode_max"]  = float(round(numpy.max(self.rewards_episode_ext), 6))
        
        self.result_summary["rewards_int_mean"] = float(round(rewards_int.mean(), 6))
        self.result_summary["rewards_int_std"]  = float(round(rewards_int.std(), 6))
        self.result_summary["rewards_int_max"]  = float(round(numpy.max(rewards_int), 6))

        self.result_summary["z_mag_mean"]       = float(round((z**2).mean(), 6))
        self.result_summary["z_mag_std"]        = float(round((z**2).std(), 6))



    def update_training(self, loss_actor, loss_entropy, loss_critic, returns_ext, returns_int, advantages_ext, advantages_int, ssl_metrics, im_metrics, dp=6):
        self.result_ppo = {}
        self.result_ppo["loss_actor"]   = round(torch.mean(loss_actor).item(), dp)
        self.result_ppo["loss_entropy"] = round(torch.mean(loss_entropy).item(), dp)
        self.result_ppo["loss_critic"]  = round(torch.mean(loss_critic).item(), dp)

        self.result_ppo = self.result_ppo | self._stats(returns_ext.float(), "returns_ext", dp)
        self.result_ppo = self.result_ppo | self._stats(returns_int.float(), "returns_int", dp)
        self.result_ppo = self.result_ppo | self._stats(advantages_ext.float(), "advantages_ext", dp)
        self.result_ppo = self.result_ppo | self._stats(advantages_int.float(), "advantages_int", dp)

        self.result_ssl  = ssl_metrics
        self.result_im   = im_metrics




    def get(self):
        result_log = []
        result_log.append({"summary" : self.result_summary})
        result_log.append({"ppo" : self.result_ppo})
        result_log.append({"ssl" : self.result_ssl})
        result_log.append({"im" : self.result_im})
        result_log.append({"rewards_ext" : self._numpy_array_to_list(self.rewards_ext_stats)})
        result_log.append({"rewards_int" : self._numpy_array_to_list(self.rewards_int_stats)})
        result_log.append({"actions"     : self._numpy_array_to_list(self.actions_stats)})
        result_log.append({"features"    : self._compute_log_svd_spectrum(self.spectral_samples)})

        return result_log


    def get_default_log_flags(self):
        return {"agent" : ["actions_confidence", "actions_entropy", "actions_diff_1_ratio",  "loss_actor", "loss_critic", "loss_entropy"]}


    def _stats(self, value, prefix, dp):
        result = {}
        result[prefix + "_mean"]       = round(torch.mean(value).item(), dp)
        result[prefix + "_std"]        = round(torch.std(value).item(), dp)
        result[prefix + "_mag_mean"]   = round(torch.mean(value**2).item(), dp)
        result[prefix + "_mag_std"]    = round(torch.std(value**2).item(), dp)
        result[prefix + "_max"]        = round(torch.max(value).item(), dp)
        result[prefix + "_min"]        = round(torch.min(value).item(), dp)
    
        return result



   
    def _map_list_to_histogram(self, lst, num_bins=512, max_list_length=4500):  
        # 1. Truncate list and convert to a numpy array
        arr = numpy.asarray(lst[:max_list_length], dtype=float)
        n = len(arr)
        
        # 2. Vectorized calculation of bin indices based on max_list_length scale
        max_bin_idx = num_bins - 1
        indices = numpy.arange(n)
        bin_indices = numpy.round(indices * max_bin_idx / max_list_length).astype(int)
        bin_indices = numpy.clip(bin_indices, 0, max_bin_idx)
        
        # 3. Vectorized accumulation using numpy.bincount
        sums    = numpy.bincount(bin_indices, weights=arr, minlength=num_bins)
        counts  = numpy.bincount(bin_indices, minlength=num_bins)
        
        # 4. Compute simple averaging for populated bins, leaving unreached bins as 0.0
        result = numpy.zeros(num_bins, dtype=float)
        valid_bins = counts > 0
        result[valid_bins] = sums[valid_bins] / counts[valid_bins]
        
        return result



    def _list_to_action_histogram(self, actions, num_actions=18, num_bins=100, max_list_length=4500):        
        # 1. Truncate list and convert to numpy integer array
        arr = numpy.asarray(actions[:max_list_length], dtype=int)
        n = len(arr)
        
        if n == 0:
            return numpy.zeros((num_bins, num_actions), dtype=float)
            
        # 2. Vectorized calculation of time bin indices based on max_list_length scale
        max_bin_idx = num_bins - 1
        indices = numpy.arange(n)
        bin_indices = numpy.round(indices * max_bin_idx / max_list_length).astype(int)
        bin_indices = numpy.clip(bin_indices, 0, max_bin_idx)
        
        # Ensure action values are safely within valid bounds [0, num_actions - 1]
        action_values = numpy.clip(arr, 0, num_actions - 1)
        
        # 3. Efficient 2D accumulation using flattened bincount and reshape
        flat_indices = bin_indices * num_actions + action_values
        counts = numpy.bincount(flat_indices, minlength=num_bins * num_actions).reshape(num_bins, num_actions).astype(float)
        
        # 4. Normalize per row: sum of actions in each row must equal 1 (handle empty rows safely)
        row_sums = counts.sum(axis=1, keepdims=True)
        result = numpy.divide(counts, row_sums, out=numpy.zeros_like(counts), where=row_sums > 0)
        
        return result


    def _numpy_array_to_list(self, x, dp = 4):
        tmp = numpy.round(x, dp)
        return tmp.tolist()


    def _features_spectrum(self):
        import numpy as np


    def _compute_log_svd_spectrum(self, X):
        """
        Computes the log10-scaled PCA eigenvalue spectrum and explained variance.

        Parameters:
        - X (numpy.ndarray): Input matrix of shape (num_samples, num_features), float32.
        - center (bool): Whether to mean-center the features (recommended for PCA).

        Returns:
        - dict: {
            "spectrum": np.ndarray of log10-scaled PCA eigenvalues (shape: num_features,),
            "explained_variance": np.ndarray of explained variance in percent (shape: num_features,)
        }
        """
        # 1. Mean-center data along features (columns)
        X_processed = X - numpy.mean(X, axis=0)
        
        # 2. Compute SVD (full_matrices=False is optimal since num_samples > num_features)
        _, S, _ = numpy.linalg.svd(X_processed, full_matrices=False)
            
        num_samples = X.shape[0]

        # 3. Compute PCA eigenvalues (variance of each principal component)
        eigenvalues = (S ** 2) / (num_samples - 1)

        # 4. Compute explained variance in percent
        total_variance = numpy.sum(S ** 2)
        exp_var = ((S ** 2) / total_variance) * 100.0

        # 5. Apply log10 scaling to the spectrum with numerical stability clipping
        eps = numpy.finfo(X.dtype).eps
        spectrum_clipped = numpy.maximum(eigenvalues, eps)
        log10_spectrum = numpy.log10(spectrum_clipped)

        return {
            "spectrum": self._numpy_array_to_list(log10_spectrum),
            "explained_variance": self._numpy_array_to_list(exp_var)
        }