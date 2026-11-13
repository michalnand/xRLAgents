import math

import numpy
import torch

from .trajectory_buffer_discrete_im             import *
from .ppo_agent_base                            import *
from .ppo_agent_diffusion_b_hyperparameters     import *

from ..training.ppo_diffusion_b_logger          import *

class PPOAgentDiffusionB(PPOAgentBase):

    def __init__(self, envs, Model, ssl_loss_func, hyperparameters = PPOAgentDiffusionBDefaultHyperparameters, device = "cpu"):

        super().__init__()

        self.envs        = envs
        self.state_shape = envs.observation_shape
        self.num_actions = envs.num_actions
        self.envs_count  = envs.num_envs
        self.device      = device

        self.reward_ext_coeff   = hyperparameters["reward_ext_coeff"]
        self.reward_int_a_coeff = hyperparameters["reward_int_a_coeff"]
        self.reward_int_b_coeff = hyperparameters["reward_int_b_coeff"]


        self.gamma_ext        = hyperparameters["gamma_ext"]
        self.gamma_int        = hyperparameters["gamma_int"]

        self.lam            = hyperparameters["lam"]
        self.eps_clip       = hyperparameters["eps_clip"]
        
        self.adv_ext_coeff  = hyperparameters["adv_ext_coeff"]
        self.adv_int_coeff  = hyperparameters["adv_int_coeff"]

       

        
        self.val_coeff      = hyperparameters["val_coeff"]
        self.entropy_beta   = hyperparameters["entropy_beta"]
        self.n_ppo_steps    = hyperparameters["n_steps"]
        self.n_ppo_epochs = hyperparameters["n_epochs"]
        self.ppo_batch_size     = hyperparameters["ppo_batch_size"]
        self.ssl_batch_size     = hyperparameters["ssl_batch_size"]
        self.max_grad_norm  = hyperparameters["max_grad_norm"]

        self.noise_level_train      = hyperparameters["noise_level_train"]
        self.noise_level_inference  = hyperparameters["noise_level_inference"]
        self.states_max_distance    = hyperparameters["states_max_distance"]

        self.state_normalization = hyperparameters["state_normalization"]
        self.single_frame_im    = hyperparameters["single_frame_im"]

        self.episodic_rewards_stats_buffer_size = hyperparameters["episodic_rewards_stats_buffer_size"]

        self.ssl_loss_func = ssl_loss_func


        self.model = Model(self.state_shape, self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparameters["learning_rate"], eps=1e-5)

        self.buffer = TrajectoryBufferDiscreteIM(hyperparameters["n_steps"], self.state_shape, self.num_actions, self.envs_count)

        self.ppo_logger = PPODiffusionBLogger(len(self.envs), self.num_actions)

        self.print_params()

        states, _ = envs.reset_all()
        if self.state_normalization:
            states           = torch.from_numpy(states).float().to(self.device)
            self.states_mean = states[:, 0].mean(dim=0)
            self.states_var  = torch.ones_like(self.states_mean).to(self.device)

        self.episodic_rewards_stats_buffer = []
        for n in range(self.episodic_rewards_stats_buffer_size):
            self.episodic_rewards_stats_buffer.append({})

        self.reward_buffer_ptr = numpy.zeros((len(self.envs, ), ), dtype=int)
 
   

    def _compute_im_diffusion(self, states, noise_max_level, return_metrics):

        # copy first frame
        if self.single_frame_im:
            states_tmp        = torch.zeros_like(states)
            states_tmp[:, :]  = states[:, 0].unsqueeze(1)
        else:
            states_tmp = states

        # obtain features
        _, z_target = self.model.forward_features(states_tmp)
        z_target = z_target.detach()

        # add noise
        noise_level  = noise_max_level * torch.rand(z_target.shape[0], 1, device=z_target.device)
        noise_target = noise_level * torch.randn_like(z_target)
        
        z_noised = z_target + noise_target

        # predict noise
        noise_pred = self.model.forward_diffusion(z_noised)

        # remove noise - denoising
        z_denoised_pred = z_noised - noise_pred

        # internal motivation, how well we denoise
        reward_im = (z_target - z_denoised_pred).square().mean(dim=-1)

        # MSE loss
        loss = (noise_target - noise_pred).square().mean()


        metrics = {}
        if return_metrics:
            metrics["loss_im"]               = loss.item()
            metrics["reward_im_mean"]        = reward_im.mean().item()
            metrics["reward_im_std"]         = reward_im.std().item()
            metrics["reward_im_min"]         = reward_im.min().item()
            metrics["reward_im_max"]         = reward_im.max().item()
            metrics["noise_pred_error_mean"] = ((noise_target - noise_pred)**2).mean().item()
            metrics["noise_pred_error_std"]  = ((noise_target - noise_pred)**2).std().item()

            
        return reward_im, z_target, loss, metrics


    def _expectation_reward(self, x, key):
        max_count = max(x.values())
        max_count = max(0, max_count)

        res = 1.0 - (x[key] / max_count)

        res = 2*(res - 0.5)

        return res
    
    
    ''' 
        rewards_ext, raw game rewards shape = (num_envs, )
        dones, bool if episode done, shape = (num_envs, )
    '''
    def _compute_im_diversity(self, rewards_ext, dones):
        result           = numpy.zeros(rewards_ext.shape)
        rewarded_indices = numpy.where(rewards_ext > 0)[0]

        # non zero rewards are sparse, loop is fast
        for n in rewarded_indices:
            key = int(rewards_ext[n])
            result[n] = self._expectation_reward(self.episodic_rewards_stats_buffer[n], key)

            if key in self.episodic_rewards_stats_buffer[n]:
                # increment existing
                self.episodic_rewards_stats_buffer[n][key]+= 1
            else:
                # add new key
                self.episodic_rewards_stats_buffer[n][key] = 1

            self.reward_buffer_ptr[n] = (self.reward_buffer_ptr[n] + 1)%self.episodic_rewards_stats_buffer.shape[0]


        # reset ptr when episode done
        dones_indices = numpy.where(dones)[0]
        for n in dones_indices:
            self.reward_buffer_ptr[n] = 0

        return result

    
    #update running stats when training enabled
    def _update_normalization(self, states, alpha = 0.99):
        mean = states[:, 0, :, :].mean(dim=0)
        self.states_mean = alpha*self.states_mean + (1.0 - alpha)*mean

        var = ((states[:, 0, :, :] - mean)**2).mean(dim=0)
        self.states_var  = alpha*self.states_var + (1.0 - alpha)*var 


    #normalize mean and variance
    def _states_normalize(self, states):     
        states_norm = (states - self.states_mean)/(torch.sqrt(self.states_var) + 10**-6)
        states_norm = torch.clip(states_norm, -4.0, 4.0)

        return states_norm  

    def _states_normalize_mean_std(self, states):
        states_norm = (states - states.mean(dim=(1, 2, 3), keepdim=True))/(states.std(dim=(1, 2, 3), keepdim=True) + 1e-8)
        states_norm = torch.clip(states_norm, -4.0, 4.0)
        return states_norm


    def step(self, states):
        states_t = torch.from_numpy(states).float().to(self.device)

        if self.state_normalization:
            self._update_normalization(states_t)
            states_t = self._states_normalize(states_t)
        else:
            states_t = self._states_normalize_mean_std(states_t)

            
        with torch.no_grad():
            logits, value_ext, value_int = self.model(states_t)

        actions = self._sample_actions(logits)


        states_next, rewards_ext, terminated, truncated, info = self.envs.step(actions)

        dones = terminated | truncated


        with torch.no_grad():
            rewards_int_a, z_im, _, _ = self._compute_im_diffusion(states_t, self.noise_level_inference, False)


        rewards_ext_raw = numpy.array(info["reward"], dtype=int)
        

        rewards_int_a = rewards_int_a.detach().cpu().numpy()
        rewards_int_b = self._compute_im_diversity(rewards_ext_raw, dones)


        rewards_ext_scaled = self.reward_ext_coeff*rewards_ext       
        rewards_int_scaled = numpy.clip(self.reward_int_a_coeff*rewards_int_a + self.reward_int_b_coeff*rewards_int_b, -1, 1)
        
        self.buffer.add(states_t, logits, value_ext, value_int, actions, rewards_ext_scaled, rewards_int_scaled, dones)


        self.ppo_logger.update_step(actions, rewards_ext, rewards_int_a, rewards_int_b, dones, z_im.detach().float().cpu().numpy())

        if self.buffer.is_full():
            with torch.no_grad():
                states_t = torch.from_numpy(states_next).float().to(self.device)
                if self.state_normalization:
                    states_t = self._states_normalize(states_t)
                else:
                    states_t = self._states_normalize_mean_std(states_t)    

                _, value_ext, value_int = self.model(states_t)

            self.buffer.compute_returns(self.gamma_ext, self.gamma_int, lam = self.lam, last_value_ext=value_ext, last_value_int=value_int, normalize=False)
            
            self._training()
            self.buffer.clear()

        
        # episodes ends reset is handled outside
        return states_next, rewards_ext, terminated, truncated, info

    def step_inference(self, states):
        states_t = torch.from_numpy(states).float()

        if self.state_normalization:
            states_t = self._states_normalize(states_t)

        states_t = states_t.float().to(self.device)


        with torch.no_grad():
            logits, value_ext, value_int = self.model(states_t)

        with torch.no_grad():
            _, z, _, _ = self._compute_im_diffusion(states_t, 0, False)
        
        z = z.detach().float().cpu().numpy()

        actions = self._sample_actions(logits)

        obs, rewards, terminated, truncated, info = self.envs.step(actions)

        return obs, rewards, terminated, truncated, info, z

 
    def _training(self):

        num_batches = (self.n_ppo_epochs*self.envs_count*self.n_ppo_steps)//self.ppo_batch_size


        for n in range(num_batches):
            states, logits_old, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.buffer.sample_batch(self.ppo_batch_size, self.device)

            # main PPO loss
            loss_actor, loss_critic, loss_entropy = self._ppo_loss(states, logits_old,  actions, returns_ext, returns_int, advantages_ext, advantages_int)
        
            states_a, states_b, distances = self.buffer.sample_batch_pairs(self.ssl_batch_size, self.states_max_distance, self.device)

            # copy first frame
            if self.single_frame_im:
                states_a_tmp        = torch.zeros_like(states_a)
                states_a_tmp[:, :]  = states_a[:, 0].unsqueeze(1)

                states_b_tmp        = torch.zeros_like(states_b)
                states_b_tmp[:, :]  = states_b[:, 0].unsqueeze(1)
            else:
                states_a_tmp = states_a 
                states_b_tmp = states_b


            # self supervised regularization
            loss_distance, loss_ssl, loss_ssl_metrics = self.ssl_loss_func(self.model, states_a_tmp, states_b_tmp, distances, n == (num_batches-1))


            # diffusion model IM    
            states_im = self.buffer.sample_states(self.ssl_batch_size, self.device)  
            _, _, loss_im, loss_im_metrics = self._compute_im_diffusion(states_im, self.noise_level_train, n == (num_batches-1))
        
            
            # final loss
            loss = loss_actor + loss_entropy + loss_critic + loss_distance + loss_ssl + loss_im

            self.optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            


        self.ppo_logger.update_training(loss_actor, loss_entropy, loss_critic, returns_ext, returns_int, advantages_ext, advantages_int, loss_ssl_metrics, loss_im_metrics, self.episodic_rewards_stats_buffer)


    
    def _ppo_loss(self, states, logits_old,  actions, returns_ext, returns_int, advantages_ext, advantages_int):
        logits_new, values_ext, values_int = self.model(states)

        log_probs_old = self._log_prob(actions, logits_old)
        log_probs_new = self._log_prob(actions, logits_new)

        log_ratio = log_probs_new - log_probs_old
        ratio     = torch.exp(log_ratio)


        advantages  = self.adv_ext_coeff*advantages_ext + self.adv_int_coeff*advantages_int
        advantages  = advantages.detach() 

        #advantages normalisation 
        advantages_norm  = (advantages - advantages.mean())/(advantages.std() + 1e-8)


        # PPO clipped surrogate
        p1          = ratio*advantages_norm
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_norm
        loss_actor  = -torch.min(p1, p2).mean()

        # compute entropy loss, to avoid greedy strategy
        # L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        log_probs_all = torch.nn.functional.log_softmax(logits_new, dim=1)
        probs_new = log_probs_all.exp()

        entropy = -(probs_new * log_probs_all).sum(dim=1)
        loss_entropy = -self.entropy_beta * entropy.mean()
        
        # critic MSE
        loss_critic = self.val_coeff*((returns_ext - values_ext.squeeze(1))**2).mean()
        loss_critic+= self.val_coeff*((returns_int - values_int.squeeze(1))**2).mean()

        return loss_actor, loss_critic, loss_entropy


    # bached smapling of 1 of N actions
    def _sample_actions(self, logits):
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.sample().cpu().numpy()
    

    def _log_prob(self, actions, logits):
        tmp = torch.nn.functional.log_softmax(logits, dim = 1)
        result  = tmp[range(len(actions)), actions]
        return result

   
    def get_metrics(self):
        return self.ppo_logger.get()


    def get_default_log_flags(self):
        return self.ppo_logger.get_default_log_flags()


    def save(self, result_path):
        super().save(result_path)

        if self.state_normalization:
            torch.save(self.states_mean, result_path + "/checkpoints/states_mean.pt")
            torch.save(self.states_var, result_path + "/checkpoints/states_var.pt")

    def load(self, result_path):
        super().load(result_path)

        if self.state_normalization:
            self.states_mean  = torch.load(result_path + "/checkpoints/states_mean.pt", map_location=self.device)
            self.states_var   = torch.load(result_path + "/checkpoints/states_var.pt", map_location=self.device)
        

