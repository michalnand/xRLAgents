import torch 
import numpy

from .TrajectoryBufferContinuous  import *


class AgentPPOContinuous():
    def __init__(self, envs, Model, gamma = 0.99, entropy_beta = 0.001, n_steps = 1024, batch_size = 256, result_path = "./"):
        self.envs = envs
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # agent hyperparameters
        self.gamma              = gamma
        self.entropy_beta       = entropy_beta
        self.eps_clip           = 0.2 
        self.adv_coeff          = 1.0
        self.val_coeff          = 0.5

        self.steps              = n_steps
        self.batch_size         = batch_size
        
        self.training_epochs    = 10
        self.envs_count         = len(envs)
        self.learning_rate      = 0.00025
 

        self.state_shape    =self.envs[0].observation_space.shape
        self.actions_count  =self.envs[0].action_space.shape[0]

        # create mdoel
        self.model = Model(self.state_shape, self.actions_count)
        self.model.to(self.device)
        print(self.model)

        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.trajectory_buffer = TrajectoryBufferContinuous(self.steps, self.state_shape, self.actions_count, self.envs_count)

        # logger for results
        self.result_path = result_path

        self.logger      = None
        self.rl_stats    = None
       
  
    def step(self, states, training_enabled):        
        states_t = torch.tensor(states, dtype=torch.float).to(self.device)

        # obtain model output, logits and values
        mu_t, var_t, values_t  = self.model.forward(states_t)

        #print(mu_t.min(), mu_t.max(), mu_t.mean(), var_t.min(), var_t.max(), var_t.mean())

        # sample action from normal distribution
        actions = self._sample_action(mu_t.detach().cpu().numpy(), var_t.detach().cpu().numpy())
       
        # environment step
        states_new, rewards, dones, infos = self.envs.step(actions)

        #put into policy buffer
        if training_enabled:
            # logging stats
            if self.logger is None:
                self.logger      = ValuesLogger(self.result_path + "training.log")

            if self.rl_stats is None:
                self.rl_stats    = RLStats(self.envs_count)

            # add into log
            iterations, episodes, reward_episode_mean, reward_episode_std, reward_episode_max = self.rl_stats.add(rewards, dones)
            
            # monitored variables
            self.logger.add("iterations", iterations)
            self.logger.add("episodes", episodes)
            self.logger.add("reward_episode_mean", reward_episode_mean)
            self.logger.add("reward_episode_std", reward_episode_std)
            self.logger.add("reward_episode_max", reward_episode_max)

            # top PPO training part
            # store trajecotry
            self.trajectory_buffer.add(states_t, mu_t, var_t, values_t, actions, rewards, dones)

            # if buffer is full, run training loop
            if self.trajectory_buffer.is_full():
                self.trajectory_buffer.compute_returns(self.gamma)
                self.train()
                self.trajectory_buffer.clear()  

            # save log
            if iterations%128 == 0:
                self.logger.save()
                self.logger.print()

        return states_new, rewards, dones, infos
    
    def save(self):
        torch.save(self.model.state_dict(), self.result_path + "/model.pt")

    def load(self):
        self.model.load_state_dict(torch.load(self.result_path + "/model.pt", map_location = self.device))

    def _sample_action(self, mu, var):
        sigma    = numpy.sqrt(var) + 10**-4
        action   = numpy.random.normal(mu, sigma)
        action   = numpy.clip(action, -1, 1)
        return action
    
    def train(self): 
        samples_count = self.steps*self.envs_count
        batch_count = samples_count//self.batch_size

        # epoch training
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                
                # sample batch
                states, actions_mu, actions_var, actions, returns, advantages = self.trajectory_buffer.sample_batch(self.batch_size, self.device)
                
                # compute main PPO loss
                loss_ppo = self.loss_ppo(states, actions_mu, actions_var, actions, returns, advantages)

                self.optimizer.zero_grad()        
                loss_ppo.backward()

                # gradient clip for stabilising training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

         

    '''
        main PPO loss
    '''
    def loss_ppo(self, states, actions_mu, actions_var, actions, returns, advantages):
        mu_new, var_new, values_new  = self.model.forward(states)

        
        '''
            compute critic loss, as MSE
            L = (T - V(s))^2
        '''
        values_new = values_new.squeeze(1)
        loss_value = (returns.detach() - values_new)**2
        loss_value = loss_value.mean()

        ''' 
            compute actor loss, surrogate loss
        '''
        advantages  = self.adv_coeff*advantages.unsqueeze(1).detach()
        advantages  = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)

        log_probs_old = self._log_prob(actions, actions_mu, actions_var).detach()
        log_probs_new = self._log_prob(actions, mu_new, var_new)

        ratio       = torch.exp(log_probs_new - log_probs_old)
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()  
    
       
        '''
            compute entropy loss, to avoid greedy strategy
            H = ln(sqrt(2*pi*var))
        ''' 
        loss_entropy = -(torch.log(2.0*torch.pi*var_new) + 1.0)/2.0
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = self.val_coeff*loss_value + loss_policy + loss_entropy

        self.logger.add("loss_policy",       loss_policy.detach().cpu().numpy().item())
        self.logger.add("loss_value",      loss_value.detach().cpu().numpy().item())
        self.logger.add("loss_entropy",     loss_entropy.detach().cpu().numpy().item())

        return loss

    def _log_prob(self, action, mu, var):
        p1 = -((action - mu)**2)/(2.0*var + 0.00000001)
        p2 = -torch.log(torch.sqrt(2.0*torch.pi*var)) 

        return p1 + p2
