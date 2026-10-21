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

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),        

            torch.nn.Flatten(),       

            torch.nn.Linear(128*fc_size, n_features)
        )

        for i in range(len(self.model)):    
            if hasattr(self.model[i], "weight"):
                torch.nn.init.orthogonal_(self.model[i].weight, 0.5)
                torch.nn.init.zeros_(self.model[i].bias)

    def forward(self, x):
        z = self.model(x)
        return z


class ModelMLP(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs):
        super(ModelMLP, self).__init__()

        self.lin_0 = torch.nn.Linear(n_features, n_hidden)
        self.act_0 = torch.nn.SiLU()
        
        self.lin_1 = torch.nn.Linear(n_hidden, n_outputs)


        torch.nn.init.orthogonal_(self.lin_0.weight, 0.5)
        torch.nn.init.zeros_(self.lin_0.bias)
        
        torch.nn.init.orthogonal_(self.lin_1.weight, 0.01)
        torch.nn.init.zeros_(self.lin_1.bias)

        
    def forward(self, x):
        
        y = self.lin_0(x)
        y = self.act_0(y)
        
        y = self.lin_1(y)

        return y



class CNNActorCriticDiscrete(torch.nn.Module):
    def __init__(self, input_shape, n_actions, n_features = 512, n_mlp = 256):
        super().__init__()


        self.encoder = ModelFeatures(input_shape, n_features)
        self.actor   = ModelMLP(n_features, n_mlp, n_actions)
        self.critic  = ModelMLP(n_features, n_mlp, 1)



    def forward(self, x):
        z = self.encoder(x)

        logits = self.actor(z)
        value  = self.critic(z)

        return logits, value