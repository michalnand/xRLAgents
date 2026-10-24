import xRLAgents
import torch    

class Config:

    def __init__(self, num_envs = 128, render=False):

        # hyper parameters
        self.result_path    = "results/experiment_01/"
        self.num_envs       = num_envs

        self.log_interval   = 128       

        self.n_training_steps = 500000

        self.hyperparameters = xRLAgents.agents.PPOAgentDiscreteDefaultHyperparameters

        self.hyperparameters["gamma"]           = 0.99
        self.hyperparameters["n_steps"]         = 128
        self.hyperparameters["batch_size"]      = 256
        self.hyperparameters["learning_rate"]   = 0.00025
        self.hyperparameters["entropy_beta"]    = 0.01


        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create envs
        if render:
            render_mode = "human"
        else:
            render_mode = None

        env_name = "Pong-v5"
        self.envs = xRLAgents.VectorizedAtariWrapper(env_name, self.num_envs, render_mode)
        
        # CNN model model
        Model = xRLAgents.cnn_ac_discrete_model.CNNActorCriticDiscrete

        # create agent
        self.agent = xRLAgents.ppo_agent_discrete.PPOAgentDiscrete(self.envs, Model, hyperparameters = self.hyperparameters, device=self.device)

 
            