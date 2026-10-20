import math

import numpy
import torch

from .trajectory_buffer_continuous          import *
from .ppo_agent_base                        import *
from .ppo_agent_continuous_hyperparameters  import *

from ..training.ppo_continuous_logger             import *

class PPOAgentContinuous(PPOAgentBase):

    def __init__(self, envs, Model, hyperparameters = PPOAgentContinuousDefaultHyperparameters, device = "cpu"):

        super().__init__()

        self.envs        = envs
        self.state_shape = envs.observation_shape
        self.num_actions = envs.action_shape[0]
        self.envs_count  = envs.num_envs
        self.device      = device

        self.gamma          = hyperparameters["gamma"]
        self.lam            = hyperparameters["lam"]
        self.eps_clip       = hyperparameters["eps_clip"]
        self.adv_coeff      = hyperparameters["adv_coeff"]
        self.val_coeff      = hyperparameters["val_coeff"]
        self.entropy_beta   = hyperparameters["entropy_beta"]
        self.n_ppo_steps    = hyperparameters["n_steps"]
        self.n_ppo_episodes = hyperparameters["n_episodes"]
        self.batch_size     = hyperparameters["batch_size"]
        self.max_grad_norm  = hyperparameters["max_grad_norm"]

        self.model = Model(self.state_shape, self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparameters["learning_rate"], eps=1e-5)

        self.buffer = TrajectoryBufferContinuous(hyperparameters["n_steps"], self.state_shape, self.num_actions, self.envs_count)
        

        self.ppo_logger = PPOContinuousLogger()

        self.print_params()

        

    def step(self, states):
        states_t = torch.from_numpy(states).float().to(self.device)

        with torch.no_grad():
            pi_mean, pi_var, value = self.model(states_t)

        actions = self._sample_actions(pi_mean, pi_var)

        obs, rewards, terminated, truncated, info = self.envs.step(actions)

        dones = terminated | truncated

      
        # add to buffer
        self.buffer.add(states_t, pi_mean, pi_var, value, actions, rewards, dones)

        if self.buffer.is_full():
            self.buffer.compute_returns(self.gamma, self.lam, value)
            self._training()
            self.buffer.clear()

        self.ppo_logger.update_step(actions)

        # episodes ends reset is handled outside
        return obs, rewards, terminated, truncated, info

    def step_inference(self, states):
        states_t = torch.from_numpy(states).float().to(self.device)

        with torch.no_grad():
            pi_mean, pi_var, value = self.model(states_t)

        actions = self._sample_actions(pi_mean, pi_var)

        obs, rewards, terminated, truncated, info = self.envs.step(actions)

        return obs, rewards, terminated, truncated, info

 
    def _training(self):

        num_batches = (self.n_ppo_episodes*self.envs_count*self.n_ppo_steps)//self.batch_size

        for n in range(num_batches):
            states, actions_mu, actions_var, actions, returns, advantages = self.buffer.sample_batch(self.batch_size, self.device)

            pi_mean, pi_var, values = self.model(states)

            log_prob_new = self._log_prob(actions, pi_mean, pi_var)
            log_prob_old = self._log_prob(actions, actions_mu, actions_var)

            log_ratio = log_prob_new - log_prob_old
            ratio     = torch.exp(log_ratio)


            # PPO clipped surrogate
            advantages  = self.adv_coeff*advantages
            p1          = ratio*advantages
            p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
            loss_actor  = -torch.min(p1, p2).mean()

            # gaussian entropy, maximized
            entropy      = (0.5*torch.log(2.0*math.pi*math.e*pi_var)).sum(dim=1).mean()
            loss_entropy = -self.entropy_beta*entropy

            # critic MSE
            loss_critic = self.val_coeff*((returns - values.squeeze(1))**2).mean()

            # final loss
            loss = loss_actor + loss_entropy + loss_critic
            
            self.optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            
        self.ppo_logger.udpate_training(values, returns, pi_mean, pi_var, loss_actor, loss_critic, loss_entropy)


    # gaussian distribution
    def _sample_actions(self, pi_mean, pi_var):
        actions = pi_mean + torch.sqrt(pi_var) * torch.randn_like(pi_mean)
        return actions.cpu().numpy().astype(numpy.float32)

    def _log_prob(self, actions, mu, var):
        return (-((actions - mu)**2) / (2.0*var) - 0.5*torch.log(2.0*math.pi*var)).sum(dim=1)

   
    def get_metrics(self):
        return self.ppo_logger.get()

