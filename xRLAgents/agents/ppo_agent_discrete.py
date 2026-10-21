import math

import numpy
import torch

from .trajectory_buffer_discrete            import *
from .ppo_agent_base                        import *
from .ppo_agent_discrete_hyperparameters    import *

from ..training.ppo_discrete_logger         import *

class PPOAgentDiscrete(PPOAgentBase):

    def __init__(self, envs, Model, hyperparameters = PPOAgentDiscreteDefaultHyperparameters, device = "cpu"):

        super().__init__()

        self.envs        = envs
        self.state_shape = envs.observation_shape
        self.num_actions = envs.num_actions
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

        self.buffer = TrajectoryBufferDiscrete(hyperparameters["n_steps"], self.state_shape, self.num_actions, self.envs_count)
        

        self.ppo_logger = PPODiscreteLogger()

        self.print_params()

        

    def step(self, states):
        states_t = torch.from_numpy(states).float().to(self.device)

        with torch.no_grad():
            logits, value = self.model(states_t)

        actions = self._sample_actions(logits)

        states_next, rewards, terminated, truncated, info = self.envs.step(actions)

        dones = terminated | truncated
      
        # add to buffer
        self.buffer.add(states_t, logits, value, actions, rewards, dones)

        self.ppo_logger.update_step(actions)

        if self.buffer.is_full():
            with torch.no_grad():
                states_t = torch.from_numpy(states_next).float().to(self.device)
                _, value = self.model(states_t)

            self.buffer.compute_returns(self.gamma, self.lam, value)
            self._training()
            self.buffer.clear()


        # episodes ends reset is handled outside
        return states_next, rewards, terminated, truncated, info

    def step_inference(self, states):
        states_t = torch.from_numpy(states).float().to(self.device)

        with torch.no_grad():
            logits, value = self.model(states_t)

        logits = logits.cpu().detach().float().numpy()

        actions = numpy.argmax(logits, axis=1)

        obs, rewards, terminated, truncated, info = self.envs.step(actions)

        return obs, rewards, terminated, truncated, info

 
    def _training(self):

        num_batches = (self.n_ppo_episodes*self.envs_count*self.n_ppo_steps)//self.batch_size

        for n in range(num_batches):
            states, logits_old,  actions, returns, advantages = self.buffer.sample_batch(self.batch_size, self.device)

            logits_new, values = self.model(states)

            log_probs_old = self._log_prob(actions, logits_old)
            log_probs_new = self._log_prob(actions, logits_new)

            log_ratio = log_probs_new - log_probs_old
            ratio     = torch.exp(log_ratio)


            # PPO clipped surrogate
            advantages  = self.adv_coeff*advantages
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
            loss_critic = self.val_coeff*((returns - values.squeeze(1))**2).mean()

            # final loss
            loss = loss_actor + loss_entropy + loss_critic
            
            self.optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            
        self.ppo_logger.udpate_training(actions, values, returns, logits_old, logits_new, loss_actor, loss_critic, loss_entropy)


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

