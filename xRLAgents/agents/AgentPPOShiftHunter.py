import torch 
import numpy

from .TrajectoryBufferIM  import *
from ..training.ValuesLogger           import *



class AgentPPOShiftHunter(): 
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
        self.reward_int_coeff   = config.reward_int_coeff

        self.steps              = config.steps
        self.batch_size         = config.batch_size
        self.ss_batch_size      = config.ss_batch_size
        
        self.training_epochs    = config.training_epochs
        
        learning_rate           = config.learning_rate
        
        self.buffer_size        = config.buffer_size
        self.buffer_prob        = config.buffer_prob

       
        self.n_envs         = len(envs)

        if hasattr(self.envs, "observation_shape"):
            self.state_shape    = self.envs.observation_shape
        else:
            self.state_shape    = self.envs.observation_space.shape

        if hasattr(self.envs, "actions_count"):
            self.actions_count  = self.envs.actions_count
        else:
            self.actions_count  = self.envs.action_space.n


        # create mdoel
        self.model = Model(self.state_shape, self.actions_count)
        self.model = self.model.to(device=self.device, dtype=self.dtype).to()


        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.trajectory_buffer = TrajectoryBufferIM(self.steps, self.n_envs)

        self.states_buffer     = torch.zeros((self.buffer_size, ) + self.state_shape, dtype=self.dtype)

        # optional, for state mean and variance normalisation
        for e in range(self.n_envs):
            state, _ = self.envs.reset(e)

        self.states_buffer[:] = torch.from_numpy(state)
        

        # result loggers
        self.log_rewards_int    = ValuesLogger("rewards_int")
        self.log_loss_ppo       = ValuesLogger("loss_ppo")
        self.log_loss_im        = ValuesLogger("loss_im")

        
        self.saving_enabled = False
        self.iterations = 0

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
        print("reward_int_coeff     ", self.reward_int_coeff)
        print("steps                ", self.steps)
        print("batch_size           ", self.batch_size)
        print("ss_batch_size        ", self.ss_batch_size)
        print("training_epochs      ", self.training_epochs)
        print("learning_rate        ", learning_rate)
        
        print("buffer_size          ", self.buffer_size)
        print("buffer_prob          ", self.buffer_prob)

        print("\n\n")
        
     
    def step(self, states, training_enabled):     
        states_t = torch.from_numpy(states).to(self.dtype).to(self.device)

        # update circular states buffer
        for n in range(self.n_envs):    
            if numpy.random.rand() < self.buffer_prob: 
                # store to random place
                idx = numpy.random.randint(0, self.states_buffer.shape[0])      
                self.states_buffer[idx] = states_t[n].to("cpu").clone()

            
        logits_t, values_ext_t, values_int_t = self.model.forward(states_t)

        actions = self._sample_actions(logits_t)
      
        # environment step  
        states_new, rewards_ext, dones, infos = self.envs.step(actions)


        rewards_ext_scaled = self.reward_ext_coeff*rewards_ext
    
        # internal motivaiotn based on diffusion
        rewards_int = self._internal_motivation(states_t)
       

        rewards_int  = rewards_int.float().detach().cpu().numpy()
    
        rewards_int_scaled = numpy.clip(self.reward_int_coeff*rewards_int, -1.0, 1.0)


            
        # top PPO training part
        if training_enabled:   
            # put trajectory into policy buffer
            self.trajectory_buffer.add(states=states_t, logits=logits_t, values_ext=values_ext_t, values_int=values_int_t, actions=actions, rewards_ext=rewards_ext_scaled, rewards_int=rewards_int_scaled, dones=dones)

            # if buffer is full, run training loop
            if self.trajectory_buffer.is_full():    
                self.trajectory_buffer.compute_returns(self.gamma_ext, self.gamma_int)
                
                self.train()

                self.trajectory_buffer.clear()

                self.log_rewards_int.add("mean", rewards_int.mean())    
                self.log_rewards_int.add("std",  rewards_int.std())

        self.iterations+= 1
       
        return states_new, rewards_ext, dones, infos
    

    def save(self, result_path):
        torch.save(self.model.state_dict(), result_path + "/model.pt")
        self.result_path    = result_path
        self.saving_enabled = True

    def load(self, result_path):
        self.model.load_state_dict(torch.load(result_path + "/model.pt", map_location = self.device))

    def get_logs(self):
        logs = [self.log_rewards_int, self.log_loss_ppo, self.log_loss_im]
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
                loss_ppo = self._loss_ppo(states, None, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                self.optimizer.zero_grad()        
                loss_ppo.backward()

                # gradient clip for stabilising training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

         
        
        batch_count = samples_count//self.ss_batch_size

        
        #main IM training loop
        for batch_idx in range(batch_count):        
            #internal motivation loss, MSE diffusion    
            states  = self.trajectory_buffer.sample_states(self.ss_batch_size, self.device)


            indices = torch.randint(0, self.states_buffer.shape[0], size=(self.ss_batch_size, ))
            states_buffer_batch = self.states_buffer[indices]
            states_buffer_batch = states_buffer_batch.to(self.device)

            # discriminator loss
            loss_disc, acc_disc = self._discriminator_loss(states, states_buffer_batch)
    
            self.optimizer.zero_grad()        
            loss_disc.mean().backward() 
            self.optimizer.step() 

            # log results
            self.log_loss_im.add("loss_disc", loss_disc.float().detach().cpu().numpy().item())
            self.log_loss_im.add("disc_acc", acc_disc)

                
            


    # sample action, probs computed from logits
    def _sample_actions(self, logits):
        logits                = logits.to(torch.float32)
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().cpu().numpy()

        return actions


    # state denoising ability novely detection
    def _internal_motivation(self, states, th = 0.75):      
        # obtain taget features from states and noised states
        z_target  = self.model.forward_im_features(states)

        # diversity novelty
        y = self.model.forward_im_disc(z_target)

        y = y[:, 0]     


        k = 1.0/(1.0 - th)
        q = 0.0 - k*th

        y = torch.clip(k*y + q, -1.0, 1.0)
        
        return y.detach()

    '''
        states_curr     : batch sample of current states, shape (batch_size, ) + state_shape
        states_buffer   : full buffer of past states, shape (buffer_size, ) + state_shape
    '''
    def _discriminator_loss(self, states_curr, states_buffer):
        batch_size = states_curr.shape[0]

        # obtain features, trained via self supervised loss
        z_curr   = self.model.forward_im_features(states_curr)
        z_buffer = self.model.forward_im_features(states_buffer)

        # discriminator output, with sigmoid, binary classification
        current_pred = self.model.forward_im_disc(z_curr)
        buffer_pred  = self.model.forward_im_disc(z_buffer)

        
        # current fresh states have target 1
        # buffer old states have target 0
        loss_func = torch.nn.BCELoss()

        prediction  = torch.cat([current_pred, buffer_pred], dim=0)
        target      = torch.cat([torch.ones_like(current_pred), torch.zeros_like(buffer_pred)], dim=0)
        loss_disc   = loss_func(prediction, target)


        # statistics, accuracy computation, ( TP + TN ) / total_samples
        acc = (current_pred > 0.5).float().sum() + (buffer_pred < 0.5).float().sum()
        acc = acc/(2*batch_size) 

        return loss_disc, acc.float().detach().cpu().numpy().item()


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

