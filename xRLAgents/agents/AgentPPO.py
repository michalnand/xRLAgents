import torch 
import numpy

from .TrajectoryBuffer  import *
from ..training.ValuesLogger           import *
  
class AgentPPO():
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


        self.n_envs             = len(envs)
        self.state_shape        = self.envs.observation_space.shape
        self.actions_count      = self.envs.action_space.n

       
        self.log_loss_ppo = ValuesLogger("loss_ppo")

        if hasattr(config, "ssl_loss"):
            self.ssl_loss     = config.ssl_loss
            self.log_loss_ssl = ValuesLogger("loss_ssl")
        else:
            self.ssl_loss       = None
            self.log_loss_ssl   = None


        if hasattr(config, "rnn_policy"):
            self.rnn_policy     = config.rnn_policy
            self.rnn_shape      = config.rnn_shape
            self.hidden_state_t = torch.zeros((self.n_envs, ) + self.rnn_shape).to(self.device)
            
            self.log_rnn = ValuesLogger("rnn")
        else:
            self.rnn_policy     = False
            self.rnn_shape      = None
            self.hidden_state_t = None

            self.log_rnn = None

        if hasattr(config, "reward_shaping"):
            self.reward_shaping = config.reward_shaping
        else:
            self.reward_shaping = None  



        # create mdoel
        if self.rnn_policy:
            self.model = Model(self.state_shape, self.actions_count, self.rnn_shape)
        else:
            self.model = Model(self.state_shape, self.actions_count)

        self.model.to(self.device)
        print(self.model)

        # initialise optimizer and trajectory buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.trajectory_buffer = TrajectoryBuffer(self.steps, self.n_envs)

        
  
    def step(self, states, training_enabled):        
        states_t = torch.tensor(states, dtype=torch.float).to(self.device)

        # obtain model output, logits and values
        if self.rnn_policy: 
            logits_t, values_t, hidden_state_new  = self.model.forward(states_t, self.hidden_state_t)
        else:
            logits_t, values_t  = self.model.forward(states_t)

        actions = self._sample_actions(logits_t)
       
        # environment step
        states_new, rewards, dones, infos = self.envs.step(actions)

        # optional rewards shaping
        if self.reward_shaping is not None:
            rewards_ext_scaled = self.reward_shaping(states, rewards, infos)
        else:
            rewards_ext_scaled = rewards


        print(rewards_ext_scaled)   

        # top PPO training part
        if training_enabled:
            
            # put trajectory into policy buffer
            if self.rnn_policy:     
                self.trajectory_buffer.add(states=states_t, logits=logits_t, values=values_t, actions=actions, rewards=rewards_ext_scaled, dones=dones, hidden_state=self.hidden_state_t)
            else:
                self.trajectory_buffer.add(states=states_t, logits=logits_t, values=values_t, actions=actions, rewards=rewards_ext_scaled, dones=dones)

            # if buffer is full, run training loop
            if self.trajectory_buffer.is_full():
                self.trajectory_buffer.compute_returns(self.gamma)
                self.train()
                self.trajectory_buffer.clear()


        # update hidden state
        if self.rnn_policy:
            self.hidden_state_t = hidden_state_new.detach().clone()

            # episode end, clear hidden state
            dones_idx = numpy.where(dones)[0]
            for e in dones_idx:
                self.hidden_state_t[e, :] = 0.0


            self.log_rnn.add("mean", self.hidden_state_t.mean().detach().cpu().float().numpy().item())
            self.log_rnn.add("std", self.hidden_state_t.std().detach().cpu().float().numpy().item())
            self.log_rnn.add("mean_mag", (self.hidden_state_t**2).mean().detach().cpu().float().numpy().item())
            self.log_rnn.add("std_mag", (self.hidden_state_t**2).std().detach().cpu().float().numpy().item())
          
        
        return states_new, rewards, dones, infos
    
    def save(self, result_path):
        torch.save(self.model.state_dict(), result_path + "/model.pt")

    def load(self, result_path):
        self.model.load_state_dict(torch.load(result_path + "/model.pt", map_location = self.device, weights_only=True))

    def get_logs(self):
        result = []
        result.append(self.log_loss_ppo)
    
        if self.log_loss_ssl is not None:
            result.append(self.log_loss_ssl)

        if self.log_rnn is not None:
            result.append(self.log_rnn)
        
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
        samples_count = self.steps*self.n_envs
        batch_count = samples_count//self.batch_size

        # epoch training
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                # sample batch
                batch = self.trajectory_buffer.sample_batch(self.batch_size, self.device)

                states      = batch["states"]
                logits      = batch["logits"]
                actions     = batch["actions"]
                returns     = batch["returns"]
                advantages  = batch["advantages"]


                if self.rnn_policy:
                    hidden_state  = batch["hidden_state"]
                    # compute main PPO loss
                    loss_ppo = self.loss_ppo_rnn(states, hidden_state, logits, actions, returns, advantages)

                else:
                    # compute main PPO loss
                    loss_ppo = self.loss_ppo(states, logits, actions, returns, advantages)

                self.optimizer.zero_grad()        
                loss_ppo.backward()

                # gradient clip for stabilising training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

      

    # PPO loss call
    def loss_ppo(self, states, logits, actions, returns, advantages):
        logits_new, values_new  = self.model.forward(states)
        return self._loss_ppo(logits, actions, returns, advantages, logits_new, values_new)

    
    # PPO loss call 
    def loss_ppo_rnn(self, states, hidden_state, logits, actions, returns, advantages):
        logits_new, values_new, _ = self.model.forward(states, hidden_state)
        return self._loss_ppo(logits, actions, returns, advantages, logits_new, values_new)

    
    # main PPO loss
    def _loss_ppo(self, logits, actions, returns, advantages, logits_new, values_new):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        probs_new     = torch.nn.functional.softmax(logits_new,     dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

        '''
            compute critic loss, as MSE
            L = (T - V(s))^2
        '''
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