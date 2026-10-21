import xRLAgents
import drone_env
import torch    

class Config:

    def __init__(self, num_envs = 128, render=False):

        # hyper parameters
        self.result_path    = "results/experiment_02/"
        self.num_envs       = num_envs

        self.log_interval   = 128       

        self.n_training_steps = 100000

        self.hyperparameters = xRLAgents.agents.PPOAgentContinuousDefaultHyperparameters

        self.hyperparameters["gamma"]           = 0.99
        self.hyperparameters["n_steps"]         = 128
        self.hyperparameters["learning_rate"]   = 0.00025
        self.hyperparameters["entropy_beta"]    = 0.0002

        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create envs
        if render:
            render_mode = "human"
            trace = True
        else:
            render_mode = None
            trace = False

        self.envs        = xRLAgents.training.VectorEnv(self.num_envs, drone_env.RaceAviary, gates_randomizer=drone_env.GatesRandomizer(stacked_prob=0.3), num_future_gates=2, render_mode=render_mode, trace=trace)
        
        
        # MLP model
        Model = xRLAgents.mlp_ac_cont_model.MLPActorCriticCont

        # create agent
        self.agent = xRLAgents.ppo_agent_continuous.PPOAgentContinuous(self.envs, Model, hyperparameters = self.hyperparameters, device=self.device)

 
            