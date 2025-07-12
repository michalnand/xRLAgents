import torch 
import numpy

from .TrajectoryBufferModes     import *
from ..training.ValuesLogger    import *
  
class AgentPPOModes():
    def __init__(self, envs, Config, Model):
        self.envs = envs
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = Config()

        # agent hyperparameters
        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip
        self.adv_coeff          = config.adv_coeff
        self.val_coeff          = config.val_coeff

        self.steps              = config.steps
        self.batch_size         = config.batch_size
        
        self.training_epochs    = config.training_epochs
        
        self.learning_rate      = config.learning_rate

        self.num_modes         = config.num_modes
        self.int_reward_coeff   = config.int_reward_coeff

        if hasattr(config, "entropy_scaled"):
            self.entropy_scaled = config.entropy_scaled
        else:
            self.entropy_scaled = False


        self.envs_count         = len(envs)
        self.state_shape        = self.envs.observation_space.shape
        self.actions_count      = self.envs.action_space.n

        # create mdoel
        self.model = Model(self.state_shape, self.actions_count, self.num_modes)
        self.model.to(self.device)
        print(self.model)

        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.trajectory_buffer = TrajectoryBufferModes(self.steps, self.state_shape, self.actions_count, self.envs_count)

        self.mode_id = numpy.zeros((self.envs_count, ), dtype=int)

        self.log_loss_ppo  = ValuesLogger("loss_ppo")
        self.log_im        = ValuesLogger("im") 
        

        self.log_im.add("im_reward_mean", 0.0)
        self.log_im.add("im_reward_std", 0.0)
        self.log_im.add("loss_mode", 0.0)
        self.log_im.add("acc_mode", 0.0)


        
  
    def step(self, states, training_enabled):        
        states_t = torch.tensor(states, dtype=torch.float).to(self.device)
        mode_t   = torch.tensor(self.mode_id, dtype=int).to(self.device)
        
        # obtain model output, logits and values
        logits_t, values_t  = self.model.forward(states_t, mode_t)


        im_reward, _, _ = self._self_im_reward(states_t, mode_t)

        self.log_im.add("im_reward_mean", im_reward.mean())
        self.log_im.add("im_reward_std", im_reward.std())


    

        actions = self._sample_actions(logits_t)
       
        # environment step
        states_new, rewards, dones, infos = self.envs.step(actions)

        rewards_all = rewards + self.int_reward_coeff*im_reward

        # top PPO training part
        if training_enabled:
            
            # put trajectory into policy buffer
            self.trajectory_buffer.add(states_t, logits_t, values_t, actions, rewards_all, dones, self.mode_id)

            # if buffer is full, run training loop
            if self.trajectory_buffer.is_full():
                self.trajectory_buffer.compute_returns(self.gamma)
                self.train()
                self.trajectory_buffer.clear()



        dones_idx = numpy.where(dones)[0]
        for e in dones_idx:
            self.mode_id[e] = numpy.random.randint(0, self.num_modes)
        
        return states_new, rewards, dones, infos
    
    def save(self, result_path):
        torch.save(self.model.state_dict(), result_path + "/model.pt")

    def load(self, result_path):
        self.model.load_state_dict(torch.load(result_path + "/model.pt", map_location = self.device, weights_only=True))

    def get_logs(self):
        result = []
        result.append(self.log_loss_ppo)
        result.append(self.log_im)
    
        return result
    

     # sample action, probs computed from logits
    def _sample_actions(self, logits):
        logits                = logits.to(torch.float32)
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().cpu().numpy()

        return actions
    
    def train(self): 
        samples_count = self.steps*self.envs_count
        batch_count = samples_count//self.batch_size

        # epoch training
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                
                # sample batch
                states, logits, actions, returns, advantages, modes = self.trajectory_buffer.sample_batch(self.batch_size, self.device)
                
                # compute main PPO loss
                loss_ppo = self.loss_ppo(states, logits, actions, returns, advantages, modes)

                self.optimizer.zero_grad()        
                loss_ppo.backward()

                # gradient clip for stabilising training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

        
        for batch_idx in range(batch_count):
            states, logits, actions, returns, advantages, modes = self.trajectory_buffer.sample_batch(self.batch_size, self.device)
            _, loss_mode, acc = self._self_im_reward(states, modes)

            self.optimizer.zero_grad()  
            loss_mode.backward()   
            self.optimizer.step()

            self.log_im.add("loss_mode", loss_mode.detach().cpu().numpy().item())
            self.log_im.add("acc_mode", acc.detach().cpu().numpy().item())


     

    # PPO loss call
    def loss_ppo(self, states, logits, actions, returns, advantages, modes):
        logits_new, values_new  = self.model.forward(states, modes)
        return self._loss_ppo(logits, actions, returns, advantages, logits_new, values_new)

   

    '''
        PPO loss
    '''
    def _loss_ppo(self, logits, actions, returns, advantages, logits_new, values_new):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        probs_new     = torch.nn.functional.softmax(logits_new,     dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

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
        advantages  = self.adv_coeff*advantages.detach() 
        advantages  = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)

        log_probs_new_  = log_probs_new[range(len(log_probs_new)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_new_ - log_probs_old_)
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()  
    
        '''
            compute entropy loss, to avoid greedy strategy
            L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = self.val_coeff*loss_value + loss_policy + loss_entropy


        self.log_loss_ppo.add("loss_policy",  loss_policy.detach().cpu().numpy().item())
        self.log_loss_ppo.add("loss_value",   loss_value.detach().cpu().numpy().item())
        self.log_loss_ppo.add("loss_entropy", loss_entropy.detach().cpu().numpy().item())


        return loss


    
    def _self_im_reward(self, s, z):

        logits = self.model.forward_mode(s)  # shape: [B, num_skills]
        #log_probs = torch.nn.functional.log_softmax(logits, dim=1)  # log p(z | s)

        # Reward: log-probability of true skill
        #reward = log_probs[torch.arange(len(z)), z]  # shape: [B]

        # Discriminator loss: cross-entropy
        loss = torch.nn.functional.cross_entropy(logits, z)


        probs = torch.nn.functional.softmax(logits, dim=1) 
        
        reward = probs[torch.arange(len(z)), z]

        if self.entropy_scaled:
            reward = (reward - reward.mean())/(reward.std() + 1e-8)
            reward = torch.clip(reward, 0.0, 4.0)
                 
        pred = torch.argmax(logits, dim=-1)
        acc = (pred == z).float().mean()


        reward = reward.detach().cpu().numpy()

        return reward, loss, acc