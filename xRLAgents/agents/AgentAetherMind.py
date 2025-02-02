import torch 
import numpy

from .TrajectoryBufferIM  import *
from ..training.ValuesLogger           import *

from .StatesActionBuffer  import *
from .EpisodicGoalsBuffer import *


class ContextualState:

    def __init__(self, n_envs, model_forward_func, state_size, context_size, frame_stacking, device):
        self.n_envs         = n_envs
        self.context_size   = context_size
        self.frame_stacking = frame_stacking

        self.model_forward_func = model_forward_func
        self.state_t = torch.zeros((n_envs, frame_stacking, state_size), dtype=torch.float32, device=device)

        x = torch.randn((1, 1, ) + state_size)
        z = self.model_forward_func(x).detach()
        
        n_features = z.shape[-1]
        
        self.contextual_state = torch.zeros((n_envs, context_size + frame_stacking, n_features), dtype=torch.float32, device=device)

    def reset(self, env_id):
        self.state_t[env_id]          = 0.0
        self.contextual_state[env_id] = 0.0

    def step(self, state, contextual_states, refresh_indices, refresh_all = False):
        self.state_t        = torch.roll(self.state_t, 1, 1)
        self.state_t[:, 0]  = state[0].detach()

        # add current features on first position
        # if required refresh all, forward is passed turth all stored states
        if refresh_all:
            count = self.frame_stacking
        else:
            count = 1
        
        for n in range(count):
            z = self.model_forward_func(self.state_t[:, n].unsqueeze(0))
            self.contextual_state[:, n] = z.detach().clone()

        # if required refresh all, new features are computed for whole contextual buffer
        # env-wise batching for faster processing
        if refresh_all:
            for n in range(self.context_size):
                z = self.model_forward_func(contextual_states[:, n].unsqueeze(0))
                self.contextual_state[:, n + self.frame_stacking] = z.detach()
        else:
        # udpate only places where new context added
        # this is sparse event and can be processed in for loop one by one
        # in future process in batch
            for n in range(self.n_envs):
                idx = refresh_indices[n]
                if idx > -1:
                    z = self.model_forward_func(contextual_states[n, idx].unsqueeze(0).unsqueeze(1))
                    self.contextual_state[n, idx + self.frame_stacking] = z.squeeze(0).detach()

        return self.contextual_state




class AgentAetherMind(): 
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


        frame_stacking     = config.frame_stacking
        context_size       = config.context_size
        add_threshold      = config.add_threshold

        


        self.n_envs         = len(envs)
        
        state_shape        = self.envs.observation_space.shape

        self.state_shape        = (1, state_shape[1], state_shape[2])
        self.actions_count      = self.envs.action_space.n

        # create mdoel
        self.model = Model(self.state_shape, self.actions_count)
        self.model.to(self.device)

        

        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # initialise buffers

        self.states_action_buffer = StatesActionBuffer(self.steps, self.state_shape, self.n_envs)

        self.trajectory_buffer  = TrajectoryBufferIM(self.steps, self.state_shape, self.actions_count, self.n_envs)

        # contextual buffer for creating context, and refreshing features
        self.goals_buffer       = EpisodicGoalsBuffer(context_size, self.n_envs, self.state_shape, n_frames = 2, alpha = 0.1, add_threshold = add_threshold)
        self.contextual_state   = ContextualState(self.n_envs, self.model.features_forward, self.state_shape, context_size, frame_stacking, self.device)
    


        self.episode_steps     = numpy.zeros(self.n_envs, dtype=int)

        self.refresh_all = False


        # optional, for state mean and variance normalisation        
        self.state_mean  = numpy.zeros(self.state_shape, dtype=numpy.float32)

        for e in range(self.n_envs):
            state, _ = self.envs.reset(e)
            self.state_mean+= state.copy()

        self.state_mean/= self.n_envs
        self.state_var = numpy.ones(self.state_shape,  dtype=numpy.float32)


        # result loggers
        self.log_rewards_goal   = ValuesLogger("rewards_goal")
        self.log_rewards_int    = ValuesLogger("rewards_int")
        self.log_loss_ppo       = ValuesLogger("loss_ppo")
        self.log_loss_diffusion = ValuesLogger("loss_diffusion")
        self.log_loss_im_ssl    = ValuesLogger("loss_im_ssl")


        # print parameters summary
        print("\n\n\n\n")

        print("model")
        print(self.model)
        print("\n\n")

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
        print("frame_stacking       ", frame_stacking)
        print("context_size         ", context_size)
        print("add_threshold        ", add_threshold)

        print("\n\n")
        
     
  
    

    def reset_contextual_state(self, env_id):
        self.contextual_state[env_id, :] = 0.0

    def step(self, states, training_enabled):     

        states_norm = self._state_normalise(states, training_enabled)   
        states_t    = torch.tensor(states_norm, dtype=torch.float).to(self.device)
        states_t    = states_t[:, 0].unsqueeze(1)

        # this need optimisation, run only on states which change
        key_states, goal_rewards_t, refresh_indices = self.goals_buffer.step(states_t)

        self.contextual_state.step(states_t, key_states, refresh_indices, self.refresh_all)

       
        # concatenate current state wit context
        contextual_state = torch.concatenate([states_t, key_states], dim=1)

        print("contextual_state = ", contextual_state.shape)

        # returns (n_envs, channels + context_size, n_features)
        z = self.model.forward_features(contextual_state).detach()

        print("z = ", z.shape)

        # obtain model output, logits and values, use abstract state space z
        logits_t, values_ext_t, values_int_t  = self.model.forward(z)

        # sample action, probs computed from logits
        actions = self._sample_actions(logits_t)
      
       
        # environment step  
        states_new, rewards_ext, dones, infos = self.envs.step(actions)

        rewards_goal      = goal_rewards_t.detach().cpu().numpy()
        rewards_ext_goals = self.reward_ext_coeff*rewards_ext + self.reward_goal_coeff*rewards_goal
        print("rewards_ext_goals = ", rewards_ext_goals.shape)

        # obtain only current state, located on first position
        z_tmp = z[:, 0].detach()

        rewards_int, _     = self._internal_motivation(z_tmp, self.alpha_inf, self.alpha_inf, self.denoising_steps)
        rewards_int        = rewards_int.detach().cpu().numpy()
        rewards_int_scaled = numpy.clip(self.reward_int_coeff*rewards_int, 0.0, 1.0)

        # top PPO training part
        self.refresh_all = False
        if training_enabled:   
            
            # add into buffer
            self.states_action_buffer.add(states_t, actions) 
            
            # put trajectory into policy buffer
            self.trajectory_buffer.add(z, logits_t, values_ext_t, values_int_t, actions, rewards_ext_goals, rewards_int_scaled, dones, self.episode_steps)

            # if buffer is full, run training loop
            if self.trajectory_buffer.is_full():
                self.trajectory_buffer.compute_returns(self.gamma_ext, self.gamma_int)
                
                # TODO
                #self.train()

                self.states_action_buffer.clear()
                self.trajectory_buffer.clear()

                self.refresh_all = True
          

        self.episode_steps+= 1 

        dones_idx = numpy.where(dones)[0]
        for i in dones_idx:
            self.episode_steps[i] = 0
            self.goals_buffer.reset(i)
            self.contextual_state.reset(i)


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
        return [self.log_rewards_goal, self.log_rewards_int, self.log_loss_ppo, self.log_loss_diffusion, self.log_loss_im_ssl]

    def train(self): 
        samples_count = self.steps*self.n_envs
        batch_count = samples_count//self.batch_size

        # PPO epoch training loop
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                # sample batch 
                z, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.trajectory_buffer.sample_batch(self.batch_size, self.device)
                
                # compute main PPO loss
                loss_ppo = self._loss_ppo(z, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                self.optimizer.zero_grad()        
                loss_ppo.backward()

                # gradient clip for stabilising training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

         
        
        batch_count = samples_count//self.ss_batch_size
        
        #main features and IM training loop
        for batch_idx in range(batch_count):    
            
            states_now, states_next, actions = self.states_action_buffer.sample_states_pairs(self.ss_batch_size, self.device)
            
            # obtain features
            z_now   = self.model.forward_features(states_now)
            z_next  = self.model.forward_features(states_next)

            # internal motivation loss, MSE diffusion    
            _, loss_diffusion  = self._internal_motivation(z_now.detach(), self.alpha_min, self.alpha_max, self.denoising_steps)

            # self supervised target regularisation
            loss_ssl, info_ssl = self.im_ssl_loss(self.model, z_now, z_next, actions)

            # final IM loss
            loss_im = loss_diffusion.mean() + loss_ssl

            self.optimizer.zero_grad()        
            loss_im.mean().backward() 
            self.optimizer.step() 

            # log results
            self.log_loss_diffusion.add("mean", loss_diffusion.mean().detach().cpu().numpy())
            self.log_loss_diffusion.add("std", loss_diffusion.std().detach().cpu().numpy())
                
            for key in info_ssl:
                self.log_loss_im_ssl.add(str(key), info_ssl[key])

    # sample action, probs computed from logits
    def _sample_actions(self, logits):
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

        z_denoised = z_noised.detach().clone()
    
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
        logits_new, values_ext_new, values_int_new  = self.model.forward(states)


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



        self.log_loss_ppo.add("loss_policy",  loss_policy.detach().cpu().numpy().item())
        self.log_loss_ppo.add("loss_critic",  loss_critic.detach().cpu().numpy().item())
        self.log_loss_ppo.add("loss_entropy", loss_entropy.detach().cpu().numpy().item())

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

    def _state_normalise(self, states, training_enabled, alpha = 0.99): 

        if self.state_normalise:
            #update running stats only during training
            if training_enabled:
                mean = states.mean(axis=0)
                self.state_mean = alpha*self.state_mean + (1.0 - alpha)*mean
        
                var = ((states - mean)**2).mean(axis=0)
                self.state_var  = alpha*self.state_var + (1.0 - alpha)*var 
             
            #normalise mean and variance
            states_norm = (states - self.state_mean)/(numpy.sqrt(self.state_var) + 10**-6)
            states_norm = numpy.clip(states_norm, -4.0, 4.0)
        
        else:
            states_norm = states
        
        return states_norm