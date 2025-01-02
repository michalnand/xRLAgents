import torch 
import numpy

from .TrajectoryBufferIM  import *
from ..training.ValuesLogger           import *

from .AdaptiveGoalsBuffer import *  
  
class AgentPPOInDiffD(): 
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
        self.reward_int_coeff   = config.reward_int_coeff
        self.goal_reach_coeff   = config.goal_reach_coeff
        self.goal_steps_coeff   = config.goal_steps_coeff


        self.steps              = config.steps
        self.batch_size         = config.batch_size
        self.ss_batch_size      = config.ss_batch_size
        
        self.training_epochs    = config.training_epochs
        
        self.learning_rate        = config.learning_rate
        self.im_ssl_loss          = config.im_ssl_loss
        self.im_noise             = config.im_noise
        self.alpha_min            = config.alpha_min
        self.alpha_max            = config.alpha_max
        self.alpha_inf            = config.alpha_inf
        self.denoising_steps      = config.denoising_steps

        self.state_normalise    = config.state_normalise


        self.envs_count         = len(envs)
        state_shape             = self.envs.observation_space.shape
        self.actions_count      = self.envs.action_space.n

        self.state_shape = (state_shape[0] + 1, state_shape[1], state_shape[2])
        
        # create mdoel
        self.model = Model(self.state_shape, self.actions_count)
        self.model.to(self.device)

        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.trajectory_buffer = TrajectoryBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count)

        #self.goals_buffer      = GoalsBuffer(state_shape[1], state_shape[2], 256)
        self.goals_buffer = AdaptiveGoalsBuffer(self.envs_count, 256, state_shape[1], state_shape[2], 0.1)


        self.episode_steps     = numpy.zeros(self.envs_count, dtype=int)
        self.episode_score     = numpy.zeros((self.envs_count, ))


        # optional, for state mean and variance normalisation        
        self.state_mean  = numpy.zeros(self.state_shape, dtype=numpy.float32)

        for e in range(self.envs_count):
            state, _ = self.envs.reset(e)
            
            state = numpy.concatenate([state, numpy.zeros((1, state.shape[1], state.shape[2]))], axis=0)
            self.state_mean+= state.copy()

        self.state_mean/= self.envs_count
        self.state_var = numpy.ones(self.state_shape,  dtype=numpy.float32)

        self.agent_mode  = numpy.zeros((self.envs_count, ))
        self.goal_ids    = numpy.zeros((self.envs_count, ), dtype=int)
        self.goal_states = numpy.zeros((self.envs_count, 1, state_shape[1], state_shape[2]))

        # initial goals
        for e in range(self.envs_count):
            self.goal_ids[e], self.goal_states[e] = self.goals_buffer.get_goal()


        # result loggers
        self.log_rewards_int        = ValuesLogger("rewards_int")
        self.log_loss_ppo           = ValuesLogger("loss_ppo")
        self.log_loss_diffusion     = ValuesLogger("loss_diffusion")
        self.log_loss_im_ssl        = ValuesLogger("loss_im_ssl")
        self.log_goals              = ValuesLogger("goals")

        # goals stats
        self.goal_reached_flag_sum = numpy.zeros((self.envs_count, ))
        self.goal_reached_flag     = numpy.zeros((self.envs_count, ))

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
        print("reward_int_coeff     ", self.reward_int_coeff)
        print("goal_reach_coeff     ", self.goal_reach_coeff)
        print("goal_steps_coeff     ", self.goal_steps_coeff)
        print("steps                ", self.steps)
        print("batch_size           ", self.batch_size)
        print("ss_batch_size        ", self.ss_batch_size)
        print("training_epochs      ", self.training_epochs)
        print("learning_rate        ", self.learning_rate)
        print("im_ssl_loss          ", self.im_ssl_loss)
        print("im_noise             ", self.im_noise)
        print("alpha_min            ", self.alpha_min)
        print("alpha_max            ", self.alpha_max)
        print("alpha_inf            ", self.alpha_inf)
        print("denoising_steps      ", self.denoising_steps)
        print("state_normalise      ", self.state_normalise)

        print("\n\n")

    def _create_states(self, states, goals):
        result = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)

        ch = states.shape[1]
        result[:, 0:ch] = states

        if goals is not None:
            result[:, ch:] = goals

        return result
  
    def step(self, states, training_enabled):    

        states_goals = self._create_states(states, self.goal_states)

        states_norm = self._state_normalise(states_goals, training_enabled)   

        states_t = torch.tensor(states_norm, dtype=torch.float).to(self.device)

        # obtain model output, logits and values
        logits_t, values_ext_t, values_int_t  = self.model.forward(states_t)

        # sample action, probs computed from logits
        actions = self._sample_actions(logits_t)
      
       
        # environment step  
        states_new, rewards_ext, dones, infos = self.envs.step(actions)

        rewards_int, _     = self._internal_motivation(states_t, self.alpha_inf, self.alpha_inf, self.denoising_steps)
        rewards_int        = rewards_int.detach().cpu().numpy()
        rewards_int_scaled = numpy.clip(self.reward_int_coeff*rewards_int, 0.0, 1.0)

        self.episode_score+= rewards_ext


        # add or refresh target state
        rewards_ext_g = rewards_ext.copy()

        reach_reward, steps_reward, goal_added = self.goals_buffer.step(self, states, self.episode_score, self.episode_steps, self.goal_ids)
        
        for n in range(self.envs_count):
            # reward only when agent in goal reaching mode and goal reached
            if self.agent_mode[n] == 1 and reach_reward[n]:    
                
                # don't reward for reaching initial goal, since it is trivial
                if self.goal_ids[n] != 0:             
                    # reward for reaching goal
                    rewards_ext_g[n]+= reach_reward[n]*self.goal_reach_coeff

                    # extra reward for faster goal reaching
                    if steps_reward:
                        rewards_ext_g[n]+= steps_reward[n]*self.goal_steps_coeff

                # clear goal    
                self.goal_ids[n]    = 0 
                self.goal_states[n] = 0.0

                # agent to common mode
                self.agent_mode[n] = 0

                # store fot statistics
                self.goal_reached_flag[n] = 1.0 

                #print("goal reached ", i, reach_reward, steps_reward, self.episode_steps[i], (self.agent_mode*1.0).mean())

              

        
        # top PPO training part
        if training_enabled:    
            
            # put trajectory into policy buffer
            self.trajectory_buffer.add(states_t, logits_t, values_ext_t, values_int_t, actions, rewards_ext_g, rewards_int_scaled, dones, self.episode_steps, self.agent_mode)

            # if buffer is full, run training loop
            if self.trajectory_buffer.is_full():
                self.trajectory_buffer.compute_returns(self.gamma_ext, self.gamma_int)
                self.train()
                self.trajectory_buffer.clear()

        self.episode_steps+= 1 

        dones_idx = numpy.where(dones)[0]
        for i in dones_idx:
            self.episode_steps[i] = 0
            self.episode_score[i] = 0

            if self.goals_buffer.get_count() > 0:
                # choose new random goal state
                self.goal_ids[i], self.goal_states[i] = self.goals_buffer.get_goal()

                # goal reaching mode
                self.agent_mode[i] = 1      

            # clear stats
            self.goal_reached_flag_sum[i] = self.goal_reached_flag[i]
            self.goal_reached_flag[i] = 0.0


        self.log_rewards_int.add("mean", rewards_int.mean())
        self.log_rewards_int.add("std", rewards_int.std())

        self.log_goals.add("count", self.goals_buffer.get_count())

        self.log_goals.add("reached mean", self.goal_reached_flag_sum.mean())
        self.log_goals.add("reached std", self.goal_reached_flag_sum.std())
        self.log_goals.add("mode mean", self.agent_mode.mean())
        self.log_goals.add("mode std", self.agent_mode.std())

        return states_new, rewards_ext, dones, infos
    

    def save(self, result_path):
        torch.save(self.model.state_dict(), result_path + "/model.pt")
        self.goals_buffer.save(result_path + "/goals_buffer_")

    def load(self, result_path):
        self.model.load_state_dict(torch.load(result_path + "/model.pt", map_location = self.device))

    def get_logs(self):
        return [self.log_rewards_int, self.log_loss_ppo, self.log_loss_diffusion, self.log_loss_im_ssl, self.log_goals]

    def train(self): 
        samples_count = self.steps*self.envs_count
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
            states_now, states_next, _, _ = self.trajectory_buffer.sample_state_pairs(self.ss_batch_size, self.device)
            _, loss_diffusion  = self._internal_motivation(states_now, self.alpha_min, self.alpha_max, self.denoising_steps)


            #self supervised target regularisation
            states_now, states_next, actions, mode = self.trajectory_buffer.sample_state_pairs(self.ss_batch_size, self.device)
            loss_ssl, info_ssl = self.im_ssl_loss(self.model, states_now, states_next, actions, mode)

            #final IM loss
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