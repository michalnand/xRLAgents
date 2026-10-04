import torch 
import numpy
import os

from .TrajectoryBufferIM            import *
from ..training.ValuesLogger        import *

from ..utils.features_eda           import *
from ..utils.actions_eda            import *

from .state_normalization           import *
from .loss_ppo                      import *



class AgentDiffExp(): 


    def __init__(self, envs, Config, Model, result_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.result_path = result_path

        self.envs       = envs

        self.n_envs     = len(envs)

        self.state_shape = self.envs.obs_shape
        self.n_actions   = self.envs.action_dim


        self.config = Config()


        self.reward_ext_coeff = self.config.reward_ext_coeff
        self.reward_int_coeff = self.config.reward_int_coeff

        self.gamma_ext        = self.config.gamma_ext
        self.gamma_int        = self.config.gamma_int

        self.alpha_inf        = self.config.alpha_inf
        self.alpha_training   = self.config.alpha_training
        self.denoising_steps  = self.config.denoising_steps

        self.batch_size         = self.config.batch_size
        self.ss_batch_size      = self.config.ss_batch_size
        self.n_epoch            = self.config.n_epoch

        self.ppo_steps        = self.config.ppo_steps
        self.adv_ext_coeff    = self.config.adv_ext_coeff
        self.adv_int_coeff    = self.config.adv_int_coeff
        self.eps_clip         = self.config.eps_clip
        self.entropy_beta     = self.config.entropy_beta
        self.val_coeff        = self.config.val_coeff

        # loss weights
        self.w_ppo                = self.config.w_ppo
        self.w_ssl                = self.config.w_ssl
        self.w_diffusion          = self.config.w_diffusion
                
        self.steps_distance       = self.config.steps_distance


        self.features_eda       = FeaturesEDA(self.result_path + "logs/")
        self.actions_eda        = ActionsEDA(self.n_envs, self.n_actions, self.result_path + "logs/")

        self.episode_steps = numpy.zeros((self.n_envs,), dtype=int)
        self.iterations    = 0


        self.model = Model(self.state_shape, self.n_actions)
        self.model.to(self.device)
        self.model = self.model.to(device=self.device)

        print(self.model)

        
        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        self.trajectory_buffer = TrajectoryBufferIM(self.ppo_steps, self.n_envs)
        
        # reset envs and obtains stats for states normalisation (optional)
        states      = self.envs.reset()
        
        # optional normalization
        self.states_normalization = StateNormalization(self.config.states_normalziation, states)


        # result loggers
        self.log_rewards_int    = ValuesLogger("rewards_int")
        self.log_loss_ppo       = ValuesLogger("loss_ppo")
        self.log_loss_diffusion = ValuesLogger("loss_diffusion")
        self.log_loss_im_ssl    = ValuesLogger("loss_im_ssl")
        
        
        
    

    def step(self, states):     
        self.iterations+= 1

        states_t = torch.from_numpy(states)

        # optional states normalization, if none internally skiped
        states_t = self.states_normalization(states_t)

        states_t = states_t.to(self.device)
    
       
        # obtain model output, logits and values, use abstract state space z
        logits_t, values_ext_t, values_int_t = self.model.forward(states_t)
    
        actions = self._sample_actions(logits_t)
        

        # environment step  
        states_new, rewards_ext, dones, infos = self.envs.step(actions)

        rewards_ext_scaled = self.reward_ext_coeff*rewards_ext
        
        # internal motivation
        rewards_int, _     = self._internal_motivation(states_t, self.alpha_inf, self.denoising_steps)
        rewards_int        = rewards_int.float().detach().cpu().numpy()
        
        # clipping
        rewards_int_scaled = numpy.clip(self.reward_int_coeff*rewards_int, 0.0, 1.0)
        
                
        # put trajectory into policy buffer
        self.trajectory_buffer.add(states=states_t, logits=logits_t, values_ext=values_ext_t, values_int=values_int_t, actions=actions, rewards_ext=rewards_ext_scaled, rewards_int=rewards_int_scaled, dones=dones, steps=self.episode_steps)

        # actions EDA, every step
        self.actions_eda(actions, dones, rewards_ext)

        # if buffer is full, run training loop
        if self.trajectory_buffer.is_full():
            self.trajectory_buffer.compute_returns(self.gamma_ext, self.gamma_int)
            self.train()

            if (self.iterations%(8*self.ppo_steps)) == 0:
                self._features_eda(2048)
                self.actions_eda.save_log(self.iterations)
                    
            self.trajectory_buffer.clear()

        

        self.log_rewards_int.add("mean", rewards_int.mean())
        self.log_rewards_int.add("std",  rewards_int.std())

        

        # next episode step
        self.episode_steps+= 1

        # reset episode steps counter
        done_idx = numpy.where(dones)[0]
        for i in done_idx:
            self.episode_steps[i] = 0

                                
        return states_new, rewards_ext, dones, infos


    def train(self): 
        samples_count = self.ppo_steps*self.n_envs
        batch_count = samples_count//self.batch_size

        # epoch training
        for e in range(self.n_epoch):
            for batch_idx in range(batch_count):
                
                # sample batch
                batch = self.trajectory_buffer.sample_batch(self.batch_size, self.device)

                states          = batch["states"]
                logits          = batch["logits"]
                actions         = batch["actions"]
                returns_ext     = batch["returns_ext"]
                returns_int     = batch["returns_int"]
                advantages_ext  = batch["advantages_ext"]
                advantages_int  = batch["advantages_int"]


                # compute main PPO loss
                loss_ppo, info_ppo = loss_ppo_im(self.model, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int,  self.adv_ext_coeff, self.adv_int_coeff, self.eps_clip, self.entropy_beta, self.val_coeff)


                #internal motivation loss, MSE diffusion    
                states, _  = self.trajectory_buffer.sample_states(self.ss_batch_size, self.device)  
                _, loss_diffusion  = self._internal_motivation(states, self.alpha_training, 1)

                #self supervised target regularisation  
                states_a, states_b, distances = self.trajectory_buffer.sample_causal_states(self.ss_batch_size, self.steps_distance, self.device)

            
                loss_ssl, info_ssl = self.config.im_ssl_loss(self.model, states_a, states_b, distances)

                # total loss    
                loss = self.w_ppo*loss_ppo + self.w_diffusion*loss_diffusion + self.w_ssl*loss_ssl

                
                self.optimizer.zero_grad()        
                loss.backward()

                # gradient clip for stabilising training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 


                # log results
                for key in info_ppo:
                    self.log_loss_ppo.add(str(key), info_ppo[key])
                
                for key in info_ssl:
                    self.log_loss_im_ssl.add(str(key), info_ssl[key])

                self.log_loss_diffusion.add("loss_diffusion", loss_diffusion.float().detach().cpu().numpy())


        
             
       
    

    

    # agent save and load model
    def save(self):
        #torch.save(self.model.state_dict(), result_path + "/model.pt")
        pass

    def load(self): 
        self.model.load_state_dict(torch.load(self.result_path + "/model.pt", map_location = self.device))

    def get_logs(self):
        logs = [self.log_rewards_int, self.log_loss_ppo, self.log_loss_diffusion, self.log_loss_im_ssl]
        return logs

            
    def _update_logs(self, infos, rewards_int):
        self.log_rewards_int.add("mean", rewards_int.mean())
        self.log_rewards_int.add("std",  rewards_int.std())
       
            

    # sample action, probs computed from logits
    def _sample_actions(self, logits):
        logits                = logits.to(torch.float32)
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().cpu().numpy()

        return actions



    def _features_eda(self, num_samples):
        za = []
        zb = []

        num_batches = num_samples//self.ss_batch_size
        for n in range(num_batches):
            states_a, states_b, _ = self.trajectory_buffer.sample_causal_states(self.ss_batch_size, self.steps_distance, self.device)  

            _, z_tmp = self.model.forward_features(states_a)
            z_tmp = z_tmp.detach().cpu()
            za.append(z_tmp)

            _, z_tmp = self.model.forward_features(states_b)
            z_tmp = z_tmp.detach().cpu()
            zb.append(z_tmp)

        za = torch.vstack(za)
        zb = torch.vstack(zb)    

        self.features_eda(self.iterations, za, zb)

       



    # state denoising ability novely detection
    def _internal_motivation(self, states, alpha_max, denoising_steps):
        # obtain taget features from states and noised states
        _, z_target  = self.model.forward_features(states)
        z_target     = z_target.detach()

        # add noise into features
        z_noised, noise, alpha = self.config.noise_func(z_target, 0, alpha_max)

        z_denoised = z_noised.detach().clone()
    
        # denoising by diffusion process
        for n in range(denoising_steps):
            noise_hat  = self.model.forward_im_diffusion(z_denoised)
            z_denoised = z_denoised - noise_hat

        # denoising novelty
        novelty    = ((z_target - z_denoised)**2).mean(dim=1)

        # MSE noise loss prediction
        noise_pred = z_noised - z_denoised
        loss = ((noise - noise_pred)**2).mean()
        
        return novelty.detach(), loss


