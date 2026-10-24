import math

import numpy
import torch

from .trajectory_buffer_discrete_im            import *
from .ppo_agent_base                        import *
from .ppo_agent_discrete_hyperparameters    import *

from ..training.ppo_diffusion_logger         import *

from .loss_self_supervised import SIGRegLoss

class PPOAgentDiffusion(PPOAgentBase):

    def __init__(self, envs, Model, hyperparameters = PPOAgentDiscreteDefaultHyperparameters, device = "cpu"):

        super().__init__()

        self.envs        = envs
        self.state_shape = envs.observation_shape
        self.num_actions = envs.num_actions
        self.envs_count  = envs.num_envs
        self.device      = device

        self.reward_ext_coeff = hyperparameters["reward_ext_coeff"]
        self.reward_int_coeff = hyperparameters["reward_int_coeff"]

        self.gamma_ext        = hyperparameters["gamma_ext"]
        self.gamma_int        = hyperparameters["gamma_int"]

        self.lam            = hyperparameters["lam"]
        self.eps_clip       = hyperparameters["eps_clip"]
        
        self.adv_ext_coeff  = hyperparameters["adv_ext_coeff"]
        self.adv_int_coeff  = hyperparameters["adv_int_coeff"]

       

        
        self.val_coeff      = hyperparameters["val_coeff"]
        self.entropy_beta   = hyperparameters["entropy_beta"]
        self.n_ppo_steps    = hyperparameters["n_steps"]
        self.n_ppo_episodes = hyperparameters["n_episodes"]
        self.ppo_batch_size     = hyperparameters["ppo_batch_size"]
        self.ssl_batch_size     = hyperparameters["ssl_batch_size"]
        self.max_grad_norm  = hyperparameters["max_grad_norm"]

        self.noise_level_train      = hyperparameters["noise_level_train"]
        self.noise_level_inference  = hyperparameters["noise_level_inference"]
        self.states_max_distance    = hyperparameters["states_max_distance"]

        self.state_normalization = hyperparameters["state_normalization"]
        self.single_frame_im = hyperparameters["single_frame_im"]


        self.model = Model(self.state_shape, self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparameters["learning_rate"], eps=1e-5)

        self.buffer = TrajectoryBufferDiscreteIM(hyperparameters["n_steps"], self.state_shape, self.num_actions, self.envs_count)

        self.loss_ssl_func = SIGRegLoss()

        self.ppo_logger = PPODiffusionLogger(len(self.envs), self.num_actions)


        self.print_params()


        states, _ = envs.reset_all()
        if self.state_normalization:
            states           = torch.from_numpy(states).float().to(self.device)
            self.states_mean = states.mean(dim=0)
            self.states_std  = torch.ones_like(self.states_mean).to(self.device)



    def _compute_im(self, states, noise_max_level, return_metrics):

        # obtain features
        _, z_target = self.model.forward_features(states)
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

    def _update_normalization(self, states, alpha = 0.99):
        self.states_mean = alpha*self.states_mean + (1 - alpha)*states.mean(dim=0)
        self.states_std  = alpha*self.states_std  + (1 - alpha)*states.std(dim=0)
    
    def _normalize_states(self, states):
        result = (states - self.states_mean.unsqueeze(0))/(self.states_std.unsqueeze(0) + 1e-6)
        result = torch.clip(result, -4.0, 4.0)

        return result

    def step(self, states):
        states_t = torch.from_numpy(states).float().to(self.device)

        if self.state_normalization:
            self._update_normalization(states_t)
            states_t = self._normalize_states(states_t)
        

        with torch.no_grad():
            logits, value_ext, value_int = self.model(states_t)

        actions = self._sample_actions(logits)

        states_next, rewards_ext, terminated, truncated, info = self.envs.step(actions)

        dones = terminated | truncated

        with torch.no_grad():

            # copy first frame
            if self.single_frame_im:
                states_tmp        = torch.zeros_like(states_t)
                states_tmp[:, :]  = states_t[:, 0].unsqueeze(1)
            else:
                states_tmp = states_t

            rewards_int, z_im, _, _ = self._compute_im(states_tmp, self.noise_level_inference, False)
            

        rewards_ext_scaled = numpy.clip(self.reward_ext_coeff*rewards_ext, 0, 1)
        rewards_int_scaled = numpy.clip(self.reward_int_coeff*rewards_int.detach().cpu().numpy(), 0, 1)
        
        self.buffer.add(states_t, logits, value_ext, value_int, actions, rewards_ext_scaled, rewards_int_scaled, dones)


        self.ppo_logger.update_step(actions, rewards_ext, rewards_int.detach().cpu().numpy(), dones, z_im.detach().float().cpu().numpy())

        if self.buffer.is_full():
            with torch.no_grad():
                states_t = torch.from_numpy(states_next).float().to(self.device)
                states_t = self._normalize_states(states_t)
                _, value_ext, value_int = self.model(states_t)

            self.buffer.compute_returns(self.gamma_ext, self.gamma_int, lam = self.lam, last_value_ext=value_ext, last_value_int=value_int)

            self._training()
            self.buffer.clear()


        # episodes ends reset is handled outside
        return states_next, rewards_ext, terminated, truncated, info

    def step_inference(self, states):
        states_t = torch.from_numpy(states).float()

        if self.state_normalization:
            states_t = self._normalize_states(states_t)

        states_t = states_t.float().to(self.device)


        with torch.no_grad():
            logits, value_ext, value_int = self.model(states_t)

        actions = self._sample_actions(logits)

        obs, rewards, terminated, truncated, info = self.envs.step(actions)

        return obs, rewards, terminated, truncated, info

 
    def _training(self):

        num_batches = (self.n_ppo_episodes*self.envs_count*self.n_ppo_steps)//self.ppo_batch_size

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
            loss_distance, loss_ssl, loss_ssl_metrics = self._ssl_loss(states_a_tmp, states_b_tmp, distances, n == (num_batches-1))

            
            # diffusion model IM
            _, _, loss_im, loss_im_metrics = self._compute_im(states_a_tmp, self.noise_level_train, n == (num_batches-1))
            

            # final loss
            loss = loss_actor + loss_entropy + loss_critic + loss_distance + loss_ssl + loss_im
            
            self.optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()


        self.ppo_logger.update_training(loss_actor, loss_entropy, loss_critic, returns_ext, returns_int, advantages_ext, advantages_int, loss_ssl_metrics, loss_im_metrics)


    def _ssl_loss(self, states_a, states_b, distance, return_metrics):
        distance = distance.float().to(self.device)

        _, za = self.model.forward_features(states_a)
        _, zb = self.model.forward_features(states_b)

        dist_pred_pos = self.model.forward_distance(za - zb)
        dist_pred_neg = self.model.forward_distance(zb - za)

        loss_distance_pos = ((distance  - dist_pred_pos)**2).mean()
        loss_distance_neg = ((-distance - dist_pred_neg)**2).mean()

        loss_distance = loss_distance_pos + loss_distance_neg

        loss_ssl = self.loss_ssl_func(za) + self.loss_ssl_func(zb)

        metrics = {}
        if return_metrics:
            shuffled_idx             = torch.randperm(states_a.shape[0])

            metrics["loss_ssl"]      = loss_ssl.item()
            metrics["loss_distance"] = loss_distance.item()
            
            metrics["z_mag_mean"]    = (za**2).mean().item()
            metrics["z_mag_std"]     = (za**2).std().item()
            metrics["z_mean"]        = za.mean().item()
            metrics["z_std"]         = za.std().item()

            # similar states distances (mean + std)
            metrics["dz_sim_mag_mean"]  = ((za - zb)**2).mean().item()
            metrics["dz_sim_mag_std"]   = ((za - zb)**2).std().item()

            # different states distances (mean + std)
            metrics["dz_shuffled_mag_mean"]  = ((za - zb[shuffled_idx])**2).mean().item()
            metrics["dz_shuffled_mag_std"]   = ((za - zb[shuffled_idx])**2).std().item()

        return loss_distance, loss_ssl, metrics

    

    def _ppo_loss(self, states, logits_old,  actions, returns_ext, returns_int, advantages_ext, advantages_int):
        logits_new, values_ext, values_int = self.model(states)

        log_probs_old = self._log_prob(actions, logits_old)
        log_probs_new = self._log_prob(actions, logits_new)

        log_ratio = log_probs_new - log_probs_old
        ratio     = torch.exp(log_ratio)


        # PPO clipped surrogate
        advantages  = self.adv_ext_coeff*advantages_ext + self.adv_int_coeff*advantages_int
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
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

