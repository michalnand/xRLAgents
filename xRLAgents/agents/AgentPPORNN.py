import torch 
import numpy

from .TrajectoryBufferRNN       import *
from ..training.ValuesLogger    import *

class AgentPPORNN():
    def __init__(self, envs, Model, gamma = 0.998, entropy_beta = 0.001, n_steps = 128, batch_size = 256):
        self.envs = envs
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # agent hyperparameters
        self.gamma              = gamma
        self.entropy_beta       = entropy_beta
        self.eps_clip           = 0.1 
        self.adv_coeff          = 1.0
        self.val_coeff          = 0.5

        self.steps              = n_steps
        self.batch_size         = batch_size
        
        self.training_epochs    = 4
        self.envs_count         = len(envs)
        self.learning_rate      = 0.0001
        self.seq_length         = 4
 

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        # create mdoel
        self.model = Model(self.state_shape, self.actions_count)
        self.model.to(self.device)
        print(self.model)

        rnn_shape = self.model.rnn_shape

        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.trajctory_buffer = TrajectoryBufferRNN(self.steps, self.state_shape, rnn_shape, self.actions_count, self.envs_count)

        self.hidden_state_t = torch.zeros((self.envs_count, ) + rnn_shape, dtype=torch.float32, device=self.device)

        self.log_loss_ppo = ValuesLogger("loss_ppo")
        self.log_rnn      = ValuesLogger("rnn")

     
     
       
  
    def step(self, states, training_enabled):        
        states_t = torch.tensor(states, dtype=torch.float).to(self.device)

        # obtain model output, logits and values
        logits_t, values_t, hidden_state_new_t  = self.model.forward(states_t, self.hidden_state_t)

        # sample action, probs computed from logits
        action_probs_t        = torch.nn.functional.softmax(logits_t, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
       
        # environment step
        states_new, rewards, dones, infos = self.envs.step(actions)

        # top PPO training part
        if training_enabled:
            
            # put trajectory into policy buffer
            self.trajctory_buffer.add(states_t, logits_t, values_t, actions, rewards, dones, self.hidden_state_t)

            # if buffer is full, run training loop
            if self.trajctory_buffer.is_full():
                self.trajctory_buffer.compute_returns(self.gamma)
                self.train()
                self.trajctory_buffer.clear()

        # update hidden state for next step
        self.hidden_state_t = hidden_state_new_t.detach().clone()

        # reset hidden state where episode done
        dones_idx = numpy.where(dones)[0]
        for idx in dones_idx:
            self.hidden_state_t[idx] = 0.0

        # hidden state stats
        h = self.hidden_state_t.detach().cpu().numpy()

        hb_corr = (((h@h.T)/h.shape[-1])**2).mean()
        he_corr = (((h.T@h)/h.shape[-1])**2).mean()

        self.log_rnn.add("mean",   (h**2).mean())
        self.log_rnn.add("std",    h.std())
        self.log_rnn.add("min",    h.min())
        self.log_rnn.add("max",    h.max())
        self.log_rnn.add("hb_corr",  hb_corr)
        self.log_rnn.add("he_corr",  he_corr)


        return states_new, rewards, dones, infos
    
    def save(self, result_path):
        torch.save(self.model.state_dict(), result_path + "/model.pt")

    def load(self, result_path):
        self.model.load_state_dict(torch.load(result_path + "/model.pt", map_location = self.device))

    def get_logs(self):
        return [self.log_loss_ppo, self.log_rnn]
    
    def train(self): 
        samples_count = self.steps*self.envs_count
        batch_count = samples_count//self.batch_size

        # epoch training
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                
                # sample batch
                states, logits, actions, returns, advantages, hidden_states = self.trajctory_buffer.sample_batch(self.seq_length, self.batch_size, self.device)
                
                # compute main PPO loss
                loss_ppo = self.loss_ppo(states, logits, actions, returns, advantages, hidden_states)

                self.optimizer.zero_grad()        
                loss_ppo.backward()

                # gradient clip for stabilising training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

         

    '''
        main PPO loss
    '''
    def loss_ppo(self, states, logits, actions, returns, advantages, hidden_states):
        logits_new, values_new, hidden_state_new = self.model.forward_seq(states, hidden_states)

        '''
            use only last step from sequence    
        '''
        logits     = logits[-1] 
        actions    = actions[-1]
        returns    = returns[-1]
        advantages = advantages[-1] 

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
        advantages       = self.adv_coeff*advantages.detach() 
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
        loss_entropy = loss_entropy.mean()


        '''
            rnn hidden states regularisation
        '''
        #loss_std = loss_std_func(hidden_state_new)
        #loss_cov = loss_cov_func(hidden_state_new)

        #hm = (hidden_state_new**2).mean(dim=-1)
        #loss_mag = torch.mean(torch.relu(hm - 0.5))
        #oss+= 1.0*loss_std + (1.0/25.0)*loss_cov + 1.0*loss_mag


        # total PPO loss
        loss = self.val_coeff*loss_value + loss_policy + self.entropy_beta*loss_entropy 
        

    

        self.log_loss_ppo.add("loss_policy",  loss_policy.detach().cpu().numpy().item())
        self.log_loss_ppo.add("loss_value",   loss_value.detach().cpu().numpy().item())
        self.log_loss_ppo.add("loss_entropy", loss_entropy.detach().cpu().numpy().item())
        #self.log_loss_ppo.add("loss_std",     loss_std.detach().cpu().numpy().item())
        #self.log_loss_ppo.add("loss_cov",     loss_cov.detach().cpu().numpy().item())
        #self.log_loss_ppo.add("loss_mag",     loss_mag.detach().cpu().numpy().item())

        return loss
