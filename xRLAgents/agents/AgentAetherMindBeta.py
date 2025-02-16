import torch 
import numpy

from .TrajectoryBufferIM  import *
from ..training.ValuesLogger           import *

from .EpisodicGoalsBufferStats import *

class AgentAetherMindBeta(): 
    def __init__(self, envs, Config, Model):
        self.envs = envs
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = Config()

        # agent hyperparameters
        self.gamma_int          = config.gamma_int
        self.gamma_ext          = config.gamma_ext

        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip
        
        self.adv_ext_coeff      = config.adv_ext_coeff
        self.adv_int_coeff      = config.adv_int_coeff
        self.val_coeff          = config.val_coeff
        self.reward_ext_coeff   = config.reward_ext_coeff
        self.reward_goal_coeff  = config.reward_goal_coeff
        self.reward_int_coeff   = config.reward_int_coeff

        self.steps              = config.steps
        self.batch_size         = config.batch_size
        self.ss_batch_size      = config.ss_batch_size
        
        self.training_epochs    = config.training_epochs
        
        learning_rate             = config.learning_rate
        self.im_ssl_loss          = config.im_ssl_loss
        self.im_noise             = config.im_noise
        self.alpha_min            = config.alpha_min
        self.alpha_max            = config.alpha_max
        self.alpha_inf            = config.alpha_inf
        self.denoising_steps      = config.denoising_steps

        self.state_normalise    = config.state_normalise

        context_size       = config.context_size
        add_threshold      = config.add_threshold

        if hasattr(config, "dtype"):
            self.dtype = config.dtype
        else:
            self.dtype = torch.float32

        self.dtype = torch.bfloat16


        self.n_envs         = len(envs)
        state_shape         = self.envs.observation_space.shape

        self.state_shape    = (state_shape[0] + context_size, state_shape[1], state_shape[2])

        self.actions_count  = self.envs.action_space.n

        # create mdoel
        self.model = Model(self.state_shape, self.actions_count)
        self.model.to(self.device)
        
        self.model = self.model.to(dtype=self.dtype, device="cuda")


        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.trajectory_buffer      = TrajectoryBufferIM(self.steps, self.state_shape, self.actions_count, self.n_envs, self.dtype)
        self.episodic_goals_buffer  = EpisodicGoalsBufferStats(context_size, self.n_envs, (1, state_shape[1], state_shape[2]), add_threshold)
        
        self.episode_steps          = numpy.zeros(self.n_envs, dtype=int)


        
        # optional, for state mean and variance normalisation        
        self.state_mean  = torch.zeros((self.state_shape[1], self.state_shape[2]), dtype=self.dtype, device=self.device)

        for e in range(self.n_envs):
            state, _ = self.envs.reset(e)
            self.state_mean+= torch.from_numpy(state[0]).to(self.dtype).to(self.device)

        self.state_mean/= self.n_envs
        self.state_var = torch.ones(self.state_mean.shape, dtype=self.dtype, device=self.device)
       
        

        # result loggers
        self.log_rewards_goal   = ValuesLogger("rewards_goal")
        self.log_rewards_int    = ValuesLogger("rewards_int")
        self.log_loss_ppo       = ValuesLogger("loss_ppo")
        self.log_loss_diffusion = ValuesLogger("loss_diffusion")
        self.log_loss_im_ssl    = ValuesLogger("loss_im_ssl")
        self.log_goals          = ValuesLogger("goals")


        # print parameters summary
        print("\n\n\n\n")

        print("model")
        print(self.model)
        print("\n\n")

        print("device               ", self.device)
        print("dtype                ", self.dtype)
        print("gamma_ext            ", self.gamma_ext)
        print("gamma_int            ", self.gamma_int)
        print("entropy_beta         ", self.entropy_beta)
        print("eps_clip             ", self.eps_clip)
        print("adv_ext_coeff        ", self.adv_ext_coeff)
        print("adv_int_coeff        ", self.adv_int_coeff)
        print("val_coeff            ", self.val_coeff)
        print("reward_ext_coeff     ", self.reward_ext_coeff)
        print("reward_goal_coeff    ", self.reward_goal_coeff)
        print("reward_int_coeff     ", self.reward_int_coeff)
        print("steps                ", self.steps)
        print("batch_size           ", self.batch_size)
        print("ss_batch_size        ", self.ss_batch_size)
        print("training_epochs      ", self.training_epochs)
        print("learning_rate        ", learning_rate)
        print("im_ssl_loss          ", self.im_ssl_loss)
        print("im_noise             ", self.im_noise)
        print("alpha_min            ", self.alpha_min)
        print("alpha_max            ", self.alpha_max)
        print("alpha_inf            ", self.alpha_inf)
        print("denoising_steps      ", self.denoising_steps)
        print("state_normalise      ", self.state_normalise)
        print("context_size         ", context_size)
        print("add_threshold        ", add_threshold)
        
        print("\n\n")
        
     
  
    def step(self, states, training_enabled):     
        states = torch.from_numpy(states).to(self.dtype).to(self.device)

        context, rewards_goal, goals_stats = self.episodic_goals_buffer.step(states[:, 0].unsqueeze(1))
        self.log_goals.add_dictionary(goals_stats)

        context = context.squeeze(2).to(device = self.device, dtype=self.dtype)

        states_t = torch.concatenate([states, context], axis=1)

        if self.state_normalise:
            self._update_normalisation(states_t, alpha = 0.99)
            states_t = self._state_normalise(states_t)

        # obtain model output, logits and values, use abstract state space z
        z = self.model.forward_features(states_t)
        logits_t, values_ext_t, values_int_t = self.model.forward_actor_critic(z)

        actions = self._sample_actions(logits_t)
      
        # environment step  
        states_new, rewards_ext, dones, infos = self.envs.step(actions)

        # sum external rewards with goal discovery reward
        rewards_ext_goal = self.reward_ext_coeff*rewards_ext + self.reward_goal_coeff*rewards_goal

        # internal motivaiotn based on diffusion
        rewards_int, _     = self._internal_motivation(states_t, self.alpha_inf, self.alpha_inf, self.denoising_steps)
        rewards_int        = rewards_int.float().detach().cpu().numpy()
        rewards_int_scaled = numpy.clip(self.reward_int_coeff*rewards_int, 0.0, 1.0)

        # top PPO training part
        if training_enabled:     
            # put trajectory into policy buffer
            self.trajectory_buffer.add(states_t, logits_t, values_ext_t, values_int_t, actions, rewards_ext_goal, rewards_int_scaled, dones, self.episode_steps)

            # if buffer is full, run training loop
            if self.trajectory_buffer.is_full():
                self.trajectory_buffer.compute_returns(self.gamma_ext, self.gamma_int)
                
                self.train()

                self.trajectory_buffer.clear()
          

        self.episode_steps+= 1 

        dones_idx = numpy.where(dones)[0]
        for i in dones_idx:
            self.episode_steps[i] = 0

            self.episodic_goals_buffer.reset(i)

        self.log_rewards_goal.add("mean", rewards_goal.mean())
        self.log_rewards_goal.add("std",  rewards_goal.std())

        self.log_rewards_int.add("mean", rewards_int.mean())
        self.log_rewards_int.add("std",  rewards_int.std())

        return states_new, rewards_ext, dones, infos
    

    def save(self, result_path):
        torch.save(self.model.state_dict(), result_path + "/model.pt")

    def load(self, result_path):
        self.model.load_state_dict(torch.load(result_path + "/model.pt", map_location = self.device))

    def get_logs(self):
        return [self.log_rewards_goal, self.log_rewards_int, self.log_loss_ppo, self.log_loss_diffusion, self.log_loss_im_ssl, self.log_goals]

    def train(self): 
        samples_count = self.steps*self.n_envs
        batch_count = samples_count//self.batch_size

        # epoch training
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                
                # sample batch
                states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.trajectory_buffer.sample_batch(self.batch_size, self.device)
                
                # compute main PPO loss
                loss_ppo = self._loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                self.optimizer.zero_grad()        
                loss_ppo.backward()

                # gradient clip for stabilising training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

         
        
        batch_count = samples_count//self.ss_batch_size
        
        #main IM training loop
        for batch_idx in range(batch_count):    
            #internal motivation loss, MSE diffusion    
            states_now, states_next, _, _   = self.trajectory_buffer.sample_state_pairs(self.ss_batch_size, self.device)
            _, loss_diffusion  = self._internal_motivation(states_now, self.alpha_min, self.alpha_max, self.denoising_steps)


            #self supervised target regularisation
            states_now, states_next, actions, _ = self.trajectory_buffer.sample_state_pairs(self.ss_batch_size, self.device)
            loss_ssl, info_ssl = self.im_ssl_loss(self.model, states_now, states_next, actions)

            #final IM loss
            loss_im = loss_diffusion.mean() + loss_ssl


            self.optimizer.zero_grad()        
            loss_im.mean().backward() 
            self.optimizer.step() 

            # log results
            self.log_loss_diffusion.add("mean", loss_diffusion.float().mean().detach().cpu().numpy())
            self.log_loss_diffusion.add("std", loss_diffusion.float().std().detach().cpu().numpy())
                
            for key in info_ssl:
                self.log_loss_im_ssl.add(str(key), info_ssl[key])




    # sample action, probs computed from logits
    def _sample_actions(self, logits):
        logits                = logits.to(torch.float32)
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().cpu().numpy()

        return actions


    # state denoising ability novely detection
    def _internal_motivation(self, states, alpha_min, alpha_max, denoising_steps):
      
        # obtain taget features from states and noised states
        z_target  = self.model.forward_im_features(states).detach()

        # add noise into features
        z_noised, noise, alpha = self.im_noise(z_target, alpha_min, alpha_max)

        z_denoised = z_noised.to(self.dtype).detach().clone()
    
        # denoising by diffusion process
        for n in range(denoising_steps):
            noise_hat = self.model.forward_im_diffusion(z_denoised)
            z_denoised = z_denoised - noise_hat

        # denoising novelty
        novelty    = ((z_target - z_denoised)**2).mean(dim=1)

        # MSE noise loss prediction
        noise_pred = z_noised - z_denoised
        loss = ((noise - noise_pred)**2).mean(dim=1)
        
        return novelty.detach(), loss





    # main PPO loss
    def _loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):

        z = self.model.forward_features(states)

        # detach all except first
        z[:, 1:] = z[:, 1:].detach()

        logits_new, values_ext_new, values_int_new = self.model.forward_actor_critic(z)


        #critic loss
        loss_critic = self._ppo_critic_loss(values_ext_new, returns_ext, values_int_new, returns_int)

        #actor loss        
        advantages  = self.adv_ext_coeff*advantages_ext + self.adv_int_coeff*advantages_int
        advantages  = advantages.detach() 

        #advantages normalisation 
        advantages_norm  = (advantages - advantages.mean())/(advantages.std() + 1e-8)

        #PPO main actor loss
        loss_policy, loss_entropy = self._ppo_actor_loss(logits, logits_new, advantages_norm, actions, self.eps_clip, self.entropy_beta)


        #total loss
        loss = self.val_coeff*loss_critic + loss_policy + loss_entropy


        self.log_loss_ppo.add("loss_policy",  loss_policy.float().detach().cpu().numpy().item())
        self.log_loss_ppo.add("loss_critic",  loss_critic.float().detach().cpu().numpy().item())
        self.log_loss_ppo.add("loss_entropy", loss_entropy.float().detach().cpu().numpy().item())

        return loss


        


    #MSE critic loss
    def _ppo_critic_loss(self, values_ext_new, returns_ext, values_int_new, returns_int):
        ''' 
        compute external critic loss, as MSE
        L = (T - V(s))^2 
        '''
        values_ext_new  = values_ext_new.squeeze(1)
        loss_ext_value  = (returns_ext.detach() - values_ext_new)**2
        loss_ext_value  = loss_ext_value.mean()

        '''
        compute internal critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_new  = values_int_new.squeeze(1)
        loss_int_value  = (returns_int.detach() - values_int_new)**2
        loss_int_value  = loss_int_value.mean()
        
        loss_critic     = loss_ext_value + loss_int_value
        return loss_critic


    #PPO actor loss
    def _ppo_actor_loss(self, logits, logits_new, advantages, actions, eps_clip, entropy_beta):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        probs_new     = torch.nn.functional.softmax(logits_new, dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

        ''' 
        compute actor loss, surrogate loss
        '''
        log_probs_new_  = log_probs_new[range(len(log_probs_new)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_new_ - log_probs_old_)
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)*advantages
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()

        ''' 
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy = entropy_beta*loss_entropy.mean()

        return loss_policy, loss_entropy



    #update running stats when training enabled
    def _update_normalisation(self, states, alpha = 0.99):
        mean = states.mean(dim=(0, 1))
        self.state_mean = alpha*self.state_mean + (1.0 - alpha)*mean

        var = ((states - mean)**2).mean(dim=(0, 1))
        self.state_var  = alpha*self.state_var + (1.0 - alpha)*var 

    #normalise mean and variance
    def _state_normalise(self, states):     
        states_norm = (states - self.state_mean)/(torch.sqrt(self.state_var) + 10**-6)
        states_norm = torch.clip(states_norm, -4.0, 4.0)
    
        return states_norm  





