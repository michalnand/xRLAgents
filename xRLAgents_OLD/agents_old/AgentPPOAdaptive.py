import torch 
import numpy

from .TrajectoryBufferRNN       import *
from ..training.ValuesLogger    import *


class TrialLogs:

    def __init__(self, num_envs, num_trials):

        self.num_trials     = num_trials

        self.trial_scores_curr      = numpy.zeros((num_envs, ))
        self.episodes_scores_curr   = numpy.zeros((num_envs, ))

        self.trial_scores           = numpy.zeros((num_trials, ))
        self.episodes_scores        = numpy.zeros((num_trials, ))

        # logs
        self.trials_summary_scores_log  = ValuesLogger("trials_summary_score")
        self.trials_scores_log          = ValuesLogger("trials_score")

        self.episodes_scores_log        = ValuesLogger("episodes_score")

    def step(self, rewards, trial_dones, episode_dones, trial_ids):
        self.trial_scores_curr+= rewards
        self.episodes_scores_curr+= rewards

        # update score per trial
        dones_idx = numpy.where(trial_dones)[0]
        for idx in dones_idx:
            self.trial_scores[trial_ids[idx]]   = self.trial_scores_curr[idx]
            self.trial_scores_curr[idx]         = 0

        # update score per episode
        dones_idx = numpy.where(episode_dones)[0]
        for idx in dones_idx:
            self.episodes_scores[idx] = self.episodes_scores_curr[idx]
            self.episodes_scores_curr[idx] = 0

        # update log for trials
        self.trials_summary_scores_log.add("mean", self.trial_scores_curr.mean())
        self.trials_summary_scores_log.add("std", self.trial_scores_curr.std())
        self.trials_summary_scores_log.add("min", self.trial_scores_curr.min())
        self.trials_summary_scores_log.add("max", self.trial_scores_curr.max())

        for n in range(self.num_trials):
            self.trials_scores_log.add(str(n), self.trial_scores[n])


        # update logs for episodes
        self.episodes_scores_log.add("mean", self.episodes_scores.mean())
        self.episodes_scores_log.add("std", self.episodes_scores.std())
        self.episodes_scores_log.add("min", self.episodes_scores.min())
        self.episodes_scores_log.add("max", self.episodes_scores.max())


class AgentPPOAdaptive():
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
        self.num_trials         = config.num_trials

        
 

        self.envs_count         = len(envs)
        self.state_shape        = self.envs.observation_space.shape
        self.actions_count      = self.envs.action_space.n

        # create mdoel
        self.model = Model(self.state_shape, self.actions_count)
        self.model.to(self.device)
        print(self.model)

        memory_shape = self.model.memory_shape

        
        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.trajectory_buffer = TrajectoryBufferRNN(self.steps, self.state_shape, memory_shape, self.actions_count, self.envs_count)

        self.memory = torch.zeros((self.envs_count, ) + memory_shape)


        # trial counter
        self.trial_ids = numpy.zeros((self.envs_count, ))
    

        self.log_loss_ppo = ValuesLogger("loss_ppo")
        self.log_memory   = ValuesLogger("memory")


        self.trial_logs = TrialLogs(self.envs_count, self.num_trials)


     
     
       
  
    def step(self, states, training_enabled):        
        states_t = torch.tensor(states, dtype=torch.float).to(self.device)

        # obtain model output, logits and values
        logits_t, values_t, memory_state_new_t  = self.model.forward(states_t, self.memory)

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
            self.trajectory_buffer.add(states_t, logits_t, values_t, actions, rewards, dones, self.memory)

            # if buffer is full, run training loop
            if self.trajectory_buffer.is_full():
                self.trajectory_buffer.compute_returns(self.gamma)
                self.train()
                self.trajectory_buffer.clear()
        
        # select which trials are done
        trial_dones   = numpy.zeros((self.envs_count, ), dtype=bool)
        for n in range(len(dones)):
            if infos[n]["trial_done"]:
                trial_dones[n] = True

        # add progress into log
        self.trial_logs.step(rewards, trial_dones, dones, self.trial_ids)

        
        # trial is done, update trial memory
        dones_idx = numpy.where(dones)[0]
        for idx in dones_idx:
            self.memory[n] = memory_state_new_t[n].detach().clone()

            # next trial
            self.trial_ids[n]+= 1

            # initialise new trial
            states_new[n] = self.envs.new_trial(n)

            # maximum trials reached, add episode done
            if self.trial_ids >= self.num_trials:
                dones[idx] = True


        # reset memory state where episode done
        dones_idx = numpy.where(dones)[0]
        for idx in dones_idx:
            self.memory[idx]     = 0.0
            self.trial_ids[idx]  = 0



        # memory state stats
        tmp = self.memory.detach().cpu().numpy()

        a_corr = (((tmp@tmp.T)/tmp.shape[-1])**2).mean()
        b_corr = (((tmp.T@tmp)/tmp.shape[-1])**2).mean() 

        self.log_memory.add("mean",   (tmp**2).mean())
        self.log_memory.add("std",    tmp.std())
        self.log_memory.add("min",    tmp.min())
        self.log_memory.add("max",    tmp.max())
        self.log_memory.add("a_corr",  a_corr)
        self.log_memory.add("b_corr",  b_corr)


        return states_new, rewards, dones, infos
    
    def save(self, result_path):
        torch.save(self.model.state_dict(), result_path + "/model.pt")

    def load(self, result_path):
        self.model.load_state_dict(torch.load(result_path + "/model.pt", map_location = self.device))

    def get_logs(self):
        return [self.log_loss_ppo, self.trial_logs.episodes_scores_log, self.trial_logs.trials_summary_scores_log, self.trial_logs.trials_scores_log]
    
    def train(self): 
        samples_count = self.steps*self.envs_count
        batch_count = samples_count//self.batch_size

        # epoch training
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                
                # sample batch
                states, logits, actions, returns, advantages, hidden_states = self.trajectory_buffer.sample_batch(self.seq_length, self.batch_size, self.device)
                
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
