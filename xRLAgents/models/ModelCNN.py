import torch


class ModelFeatures(torch.nn.Module):
    def __init__(self, input_shape, n_features):
        super(ModelFeatures, self).__init__()

        fc_size = (input_shape[1]//8) * (input_shape[2]//8)

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),    

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),    

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.SiLU(),        

            torch.nn.Flatten(),       

            torch.nn.Linear(64*fc_size, n_features)
        )

        for i in range(len(self.model)):    
            if hasattr(self.model[i], "weight"):
                torch.nn.init.orthogonal_(self.model[i].weight, 0.5)
                torch.nn.init.zeros_(self.model[i].bias)

    def forward(self, x):
        return self.model(x)


class ModelActorCritic(torch.nn.Module):    
    def __init__(self, n_inputs, n_hidden, n_actions, n_critics = 1):
        super(ModelActorCritic, self).__init__() 

        self.lin_actor0 = torch.nn.Linear(n_inputs, n_hidden)
        self.act_actor0 = torch.nn.SiLU()
        self.lin_actor1 = torch.nn.Linear(n_hidden, n_actions)

        self.lin_critic0 = torch.nn.Linear(n_inputs, n_hidden)
        self.act_critic0 = torch.nn.SiLU()
        self.lin_critic1 = torch.nn.Linear(n_hidden, n_critics) 


        torch.nn.init.orthogonal_(self.lin_actor0.weight, 0.5)
        torch.nn.init.zeros_(self.lin_actor0.bias)
        torch.nn.init.orthogonal_(self.lin_actor1.weight, 0.01)
        torch.nn.init.zeros_(self.lin_actor1.bias)

        torch.nn.init.orthogonal_(self.lin_critic0.weight, 0.5)
        torch.nn.init.zeros_(self.lin_critic0.bias)
        torch.nn.init.orthogonal_(self.lin_critic1.weight, 0.1)
        torch.nn.init.zeros_(self.lin_critic1.bias)


    def forward(self, x):
        a = self.lin_actor0(x)
        a = self.act_actor0(a)
        a = self.lin_actor1(a)

        v = self.lin_critic0(x) 
        v = self.act_critic0(v)
        v = self.lin_critic1(v)

        return a, v
    



    
'''
    CNN layers for visual information extraction
    FC layers for actor + critic
'''
class ModelCNN(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ModelCNN, self).__init__()

        n_features = 1024
        n_hidden   = 512
        self.model_features     = ModelFeatures(input_shape, n_features)
        self.model_actor_critic = ModelActorCritic(n_features, n_hidden, n_actions, 1)


    def forward(self, state):
        # obtain features
        z = self.model_features(state)

        # obtain actor and critic outputs
        logits, value = self.model_actor_critic(z)

        return logits, value
    