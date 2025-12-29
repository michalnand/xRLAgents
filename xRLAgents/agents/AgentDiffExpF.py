import torch 
import numpy

from .TrajectoryBufferIM  import *
from ..training.ValuesLogger           import *



class AgentDiffExpF(): 
    def __init__(self, envs, Config, Model):
        self.envs = envs
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        config = Config()


        if hasattr(config, "dtype"):
            self.dtype = config.dtype
        else:
            self.dtype = torch.float32

        # agent hyperparameters
        self.gamma_int          = config.gamma_int
        self.gamma_ext          = config.gamma_ext

        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip
        
        self.adv_ext_coeff      = config.adv_ext_coeff
        self.adv_int_coeff      = config.adv_int_coeff
        self.val_coeff          = config.val_coeff
        self.reward_ext_coeff   = config.reward_ext_coeff
        self.reward_int_a_coeff = config.reward_int_a_coeff
        self.reward_int_b_coeff = config.reward_int_b_coeff

        self.steps              = config.steps
        self.batch_size         = config.batch_size
        self.ss_batch_size      = config.ss_batch_size
        
        self.training_epochs    = config.training_epochs
        
        learning_rate             = config.learning_rate
        self.im_ssl_loss          = config.im_ssl_loss
        self.im_noise             = config.im_noise
        self.im_single_frame      = config.im_single_frame
        self.alpha_min            = config.alpha_min
        self.alpha_max            = config.alpha_max
        self.alpha_inf            = config.alpha_inf
        self.denoising_steps      = config.denoising_steps
        

        self.time_distances       = config.time_distances
        
        self.state_normalise        = config.state_normalise

        self.buffer_size          = config.buffer_size
        self.buffer_prob          = config.buffer_prob
        self.buffer_ptr           = 0

        if hasattr(config, "rnn_policy"):
            self.rnn_policy         = config.rnn_policy
            self.rnn_shape          = config.rnn_shape
        else:
            self.rnn_policy         = False
            self.rnn_shape          = None
        
        if hasattr(config, "reward_shaping"):
            self.reward_shaping = config.reward_shaping
        else:
            self.reward_shaping = None  

       
        self.n_envs         = len(envs)
        self.state_shape    = self.envs.observation_space.shape

        self.actions_count  = self.envs.action_space.n

        # create mdoel
        if self.rnn_policy:
            self.model = Model(self.state_shape, self.actions_count, self.rnn_shape)
        else:
            self.model = Model(self.state_shape, self.actions_count)

        self.model.to(self.device)
        
        self.model = self.model.to(dtype=self.dtype, device="cuda")


        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.trajectory_buffer = TrajectoryBufferIM(self.steps, self.n_envs)
        self.states_buffer     = torch.zeros((self.buffer_size, ) + self.state_shape, dtype=torch.float32)
        self.states_ptr        = 0

        # optional, for state mean and variance normalisation
        if self.state_normalise:
            self.state_mean  = torch.zeros((self.state_shape[1], self.state_shape[2]), dtype=self.dtype, device=self.device)

            for e in range(self.n_envs):
                state, _ = self.envs.reset(e)
                self.state_mean+= torch.from_numpy(state[0]).to(self.dtype).to(self.device)

            self.state_mean/= self.n_envs
            self.state_var = torch.ones(self.state_mean.shape, dtype=self.dtype, device=self.device)
        else:
            for e in range(self.n_envs):
                state, _ = self.envs.reset(e)
        
        if self.rnn_policy:
            self.hidden_state_t = torch.zeros((self.n_envs, ) + self.rnn_shape).to(self.dtype).to(self.device)


        self.episode_steps = torch.zeros((self.n_envs, ), dtype=int)


        # result loggers
        self.log_rewards_int    = ValuesLogger("rewards_int")
        self.log_loss_ppo       = ValuesLogger("loss_ppo")
        self.log_loss_im        = ValuesLogger("loss_im")
        self.log_loss_im_ssl    = ValuesLogger("loss_im_ssl")

        if self.rnn_policy:
            self.log_rnn        = ValuesLogger("rnn")

        self.saving_enabled     = False
        self.iterations         = 0
        self.room_ids           = []

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
        print("reward_int_a_coeff   ", self.reward_int_a_coeff)
        print("reward_int_b_coeff   ", self.reward_int_b_coeff)
        print("steps                ", self.steps)
        print("batch_size           ", self.batch_size)
        print("ss_batch_size        ", self.ss_batch_size)
        print("training_epochs      ", self.training_epochs)
        print("learning_rate        ", learning_rate)
        print("im_ssl_loss          ", self.im_ssl_loss)
        print("im_noise             ", self.im_noise)
        print("im_single_frame      ", self.im_single_frame)
        print("alpha_min            ", self.alpha_min)
        print("alpha_max            ", self.alpha_max)
        print("alpha_inf            ", self.alpha_inf)
        print("denoising_steps      ", self.denoising_steps)
        print("time_distances       ", self.time_distances)
        print("state_normalise      ", self.state_normalise)
        print("buffer_size          ", self.buffer_size)
        print("buffer_prob          ", self.buffer_prob)
        print("rnn_policy           ", self.rnn_policy)
        print("rnn_shape            ", self.rnn_shape)  
        
        print("\n\n")
        
     
    def step(self, states, training_enabled):     
        states_t = torch.from_numpy(states).to(self.dtype).to(self.device)

        # add new state with small probability
        for n in range(self.n_envs):
            p = self.buffer_prob

            # warm buffer start for initialisation
            if self.buffer_ptr < self.buffer_size:
                p = 10*p

            p = 10.0
            if numpy.random.rand() < self.buffer_prob:
                self.states_buffer[self.buffer_ptr%self.buffer_size] = states_t[n].cpu().clone()
                self.buffer_ptr = self.buffer_ptr + 1


        if self.state_normalise:
            self._update_normalisation(states_t, alpha = 0.99)
            states_t = self._state_normalise(states_t)

        # obtain model output, logits and values, use abstract state space z
        if self.rnn_policy:
            logits_t, values_ext_t, values_int_t, hidden_state_new = self.model.forward(states_t, self.hidden_state_t)
        else:
            logits_t, values_ext_t, values_int_t = self.model.forward(states_t)

        actions = self._sample_actions(logits_t)
      
        # environment step  
        states_new, rewards_ext, dones, infos = self.envs.step(actions)

        # optional rewards shaping
        if self.reward_shaping is not None:
            rewards_ext_scaled = self.reward_ext_coeff*self.reward_shaping(states, rewards_ext, infos)
        else:
            rewards_ext_scaled = self.reward_ext_coeff*rewards_ext

    
        # internal motivaiotn based on diffusion
        rewards_int_a, _ = self._novelty_internal_motivation(states_t, self.alpha_inf, self.alpha_inf, self.denoising_steps)
        rewards_int_b, _ = self._diversity_internal_motivation(states_t, torch.ones((states_t.shape[0], ), device=self.device))

        rewards_int_a = rewards_int_a.float().detach().cpu().numpy()    
        rewards_int_b = rewards_int_b.float().detach().cpu().numpy()


        rewards_int_scaled = numpy.clip(self.reward_int_a_coeff*rewards_int_a + self.reward_int_b_coeff*rewards_int_b, -1.0, 1.0)


        if "room_id" in infos[0]:
            resp = self._process_room_ids(infos)
            self.room_ids.append(resp)

            
        # top PPO training part
        if training_enabled:   
            # put trajectory into policy buffer
            if self.rnn_policy:
                self.trajectory_buffer.add(states=states_t, logits=logits_t, values_ext=values_ext_t, values_int=values_int_t, actions=actions, rewards_ext=rewards_ext_scaled, rewards_int=rewards_int_scaled, dones=dones, steps=self.episode_steps, hidden_state=self.hidden_state_t)
            else:
                self.trajectory_buffer.add(states=states_t, logits=logits_t, values_ext=values_ext_t, values_int=values_int_t, actions=actions, rewards_ext=rewards_ext_scaled, rewards_int=rewards_int_scaled, dones=dones, steps=self.episode_steps)

            # if buffer is full, run training loop
            if self.trajectory_buffer.is_full():
                if self.saving_enabled:
                    self.save_features()
                    self.saving_enabled = False

                self.trajectory_buffer.compute_returns(self.gamma_ext, self.gamma_int)
                
                self.train()

                self.trajectory_buffer.clear()

        
        self.episode_steps+= 1


        if self.rnn_policy:
            self.hidden_state_t = hidden_state_new.detach().clone()

        # reset episode steps counter and hidden state
        done_idx = numpy.where(dones)[0]
        for i in done_idx:
            self.episode_steps[i] = 0

            if self.rnn_policy:
                self.hidden_state_t[i]  = 0.0

        if self.rnn_policy:
            self.log_rnn.add("mean", self.hidden_state_t.mean().detach().cpu().float().numpy().item())
            self.log_rnn.add("std", self.hidden_state_t.std().detach().cpu().float().numpy().item())
            self.log_rnn.add("mean_mag", (self.hidden_state_t**2).mean().detach().cpu().float().numpy().item())
            self.log_rnn.add("std_mag", (self.hidden_state_t**2).std().detach().cpu().float().numpy().item())
            
        self.iterations+= 1
     
        self.log_rewards_int.add("mean_a", rewards_int_a.mean())
        self.log_rewards_int.add("std_a",  rewards_int_a.std())
        self.log_rewards_int.add("mean_b", rewards_int_b.mean())
        self.log_rewards_int.add("std_b",  rewards_int_b.std())
        
        return states_new, rewards_ext, dones, infos
    

    def save(self, result_path):
        torch.save(self.model.state_dict(), result_path + "/model.pt")
        self.result_path    = result_path
        self.saving_enabled = True

    def load(self, result_path):
        self.model.load_state_dict(torch.load(result_path + "/model.pt", map_location = self.device))

    def save_features(self):
        print("saving features to ", self.result_path, " in step ", self.iterations)

        # obtain features from current buffer
        # we save total steps*n_envs features (e.g. 128x128)
        z_ppo = []
        z_im  = []
        z_denoised = []
        for n in range(self.steps):
            x = self.trajectory_buffer.buffer["states"][n]  
            x = x.to(device=self.device, dtype=self.dtype)

            z_ppo_tmp, z_im_tmp = self.model.forward_features(x)

            # ppo features
            z_ppo.append(z_ppo_tmp.detach().cpu().float().numpy())

            # im features
            z_im.append(z_im_tmp.detach().cpu().float().numpy())

            # diffusion prediction
            noise_hat = self.model.forward_im_diffusion(z_im_tmp)
            z_hat = z_im_tmp - noise_hat

            z_denoised.append(z_hat.detach().cpu().float().numpy())

        # save features as numpy array
        z_ppo = numpy.array(z_ppo)
        z_im  = numpy.array(z_im)
        z_denoised = numpy.array(z_denoised)

        f_name = self.result_path + "/z_ppo_" + str(self.iterations) + ".npy"
        numpy.save(f_name, z_ppo)

        f_name = self.result_path + "/z_im_" + str(self.iterations) + ".npy"
        numpy.save(f_name, z_im)    

        f_name = self.result_path + "/z_denoised_" + str(self.iterations) + ".npy"
        numpy.save(f_name, z_denoised)      

        # save episode steps count
        steps = []
        for n in range(self.steps): 
            s = self.trajectory_buffer.buffer["steps"][n]
            s = s.detach().cpu().int().numpy()
            steps.append(s) 

        steps = numpy.array(steps)

        f_name = self.result_path + "/steps_" + str(self.iterations) + ".npy"
        numpy.save(f_name, steps)


        if len(self.room_ids) > 0:
            count = steps.shape[0]
            room_ids = numpy.array(self.room_ids)
            room_ids = room_ids[-count:, :]

            f_name = self.result_path + "/rooms_" + str(self.iterations) + ".npy"
            numpy.save(f_name, room_ids)

            self.room_ids = []

            print("room_ids  ", room_ids.shape)


        print("z_ppo  ", z_ppo.shape)  
        print("z_im   ", z_im.shape)  
        print("z_denoised   ", z_denoised.shape)  
        print("steps  ", steps.shape)  

        print("features saved\n\n")

     

    def get_logs(self):

        logs = [self.log_rewards_int, self.log_loss_ppo, self.log_loss_im, self.log_loss_im_ssl]

        if self.rnn_policy:
            logs.append(self.log_rnn)

        return logs


    def train(self): 
        samples_count = self.steps*self.n_envs
        batch_count = samples_count//self.batch_size

        # epoch training
        for e in range(self.training_epochs):
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
                if self.rnn_policy:
                    hidden_state  = batch["hidden_state"]
                    loss_ppo, info_ppo = self._loss_ppo(states, hidden_state, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)
                else:
                    loss_ppo, info_ppo = self._loss_ppo(states, None, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)



                # sample batch
                states         = self.trajectory_buffer.sample_states(self.ss_batch_size, self.device)
                states_old     = self._sample_buffer_states(self.ss_batch_size, self.state_normalise, self.device)

                # internal motivation loss
                #  MSE for diffusion    
                _, loss_diffusion    = self._novelty_internal_motivation(states, self.alpha_min, self.alpha_max, self.denoising_steps)
                
                # binary cross entropy for diversity loss
               
                # positive samples
                pos_labels           = torch.ones((states.shape[0], ), device=self.device)
                _, loss_diversity_a  = self._diversity_internal_motivation(states, pos_labels)
                
                # negative samples
                neg_labels           = torch.zeros((states.shape[0], ), device=self.device)
                _, loss_diversity_b  = self._diversity_internal_motivation(states_old, neg_labels)

                loss_diversity       = loss_diversity_a + loss_diversity_b



                #self supervised target regularisation
                states_seq, labels = self.trajectory_buffer.sample_states_seq(self.ss_batch_size, self.time_distances, self.device)

                # single frame input for internal motivation
                # dont use frame stacking, just copy current frame
                if self.im_single_frame:   
                    states_seq_tmp = []

                    for n in range(len(states_seq)):                
                        tmp = torch.zeros_like(states_seq[n])
                        tmp[:, :] = states_seq[n][:, 0].unsqueeze(1)

                        states_seq_tmp.append(tmp)
                else:
                    states_seq_tmp = states_seq     


                loss_ssl, info_ssl = self.im_ssl_loss(self.model, states_seq_tmp, labels)

                # total loss    
                loss = loss_ppo + loss_diffusion + loss_diversity + loss_ssl

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

                self.log_loss_im.add("loss_diffusion", loss_diffusion.float().detach().cpu().numpy())
                self.log_loss_im.add("loss_diversity", loss_diversity.float().detach().cpu().numpy())
            


         
        
        

    # sample action, probs computed from logits
    def _sample_actions(self, logits):
        logits                = logits.to(torch.float32)
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().cpu().numpy()

        return actions


    # state denoising ability novely detection
    def _novelty_internal_motivation(self, states, alpha_min, alpha_max, denoising_steps):
      
        # single frame input for internal motivation
        # dont use frame stacking, just copy current frame
        if self.im_single_frame:
            states_tmp = torch.zeros_like(states)
            states_tmp[:, :] = states[:, 0].unsqueeze(1)
        else:
            states_tmp = states

        # obtain taget features from states and noised states
        _, z_target  = self.model.forward_features(states_tmp)
        z_target     = z_target.detach()

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
        loss = ((noise - noise_pred)**2).mean()
        
        return novelty.detach(), loss

    def _diversity_internal_motivation(self, states, labels):

        # single frame input for internal motivation
        # dont use frame stacking, just copy current frame
        if self.im_single_frame:
            states_tmp = torch.zeros_like(states)
            states_tmp[:, :] = states[:, 0].unsqueeze(1)
        else:
            states_tmp = states

        # obtain taget features from states and noised states
        _, z  = self.model.forward_features(states_tmp)
        z     = z.detach()
        
        y_pred = self.model.forward_im_discriminator(z)
        y_pred = y_pred.squeeze(1)

        loss_func = torch.nn.BCELoss()
        loss = loss_func(y_pred, labels)

        # scaling into -1 .. 1 range
        diversity = 2*y_pred - 1

        return diversity.detach(), loss



    # main PPO loss
    def _loss_ppo(self, states, hidden_states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):
        if hidden_states is not None:
            logits_new, values_ext_new, values_int_new, _ = self.model.forward(states, hidden_states)
        else:
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

        # logs
        info = {}
        info["loss_policy"]  = loss_policy.float().detach().cpu().numpy().item()
        info["loss_critic"]  = loss_critic.float().detach().cpu().numpy().item()
        info["loss_entropy"] = loss_entropy.float().detach().cpu().numpy().item()

        return loss, info


        


    #MSE critic loss
    def _ppo_critic_loss(self, values_ext_new, returns_ext, values_int_new, returns_int):
        ''' 
        compute external critic loss, as MSE
        L = (T - V(s))^2 
        '''
        loss_ext_value  = (returns_ext.detach() - values_ext_new)**2
        loss_ext_value  = loss_ext_value.mean()

        '''
        compute internal critic loss, as MSE
        L = (T - V(s))^2
        '''
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


    def _sample_buffer_states(self, ss_batch_size, normalise, device):
        indices = torch.randint(0, self.buffer_size, (ss_batch_size, ))

        result = self.states_buffer[indices]
        result = result.to(device)

        if normalise:
            result = self._state_normalise(result)

        return result

    def _process_room_ids(self, infos):
        result = numpy.zeros(len(infos), dtype=int)
        for n in range(len(infos)):
            result[n] = int(infos[n]["room_id"])

        return result
    
