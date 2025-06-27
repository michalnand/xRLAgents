import torch

'''
    two hidden layers FC model for actor critic architecture
'''
class ModelFCContinuous(torch.nn.Module):
    def __init__(self, input_shape, n_actions, n_hidden = 128):
        super(ModelFCContinuous, self).__init__()

        n_inputs = input_shape[0]

        # FC model, with two hidden layers and two output heads

        self.lin0 = torch.nn.Linear(n_inputs, n_hidden)
        self.act0 = torch.nn.SiLU()
        self.lin1 = torch.nn.Linear(n_hidden, n_hidden)
        self.act1 = torch.nn.SiLU()

        self.lin_actor_mu  = torch.nn.Linear(n_hidden, n_actions)
        self.act_mu        = torch.nn.Tanh()
        self.lin_actor_var = torch.nn.Linear(n_hidden, n_actions)
        self.act_var       = torch.nn.Sigmoid()

        self.lin_critic = torch.nn.Linear(n_hidden, 1)

        # orthogonal weight init
        torch.nn.init.orthogonal_(self.lin0.weight, 0.5)
        torch.nn.init.zeros_(self.lin0.bias)
        torch.nn.init.orthogonal_(self.lin1.weight, 0.5)
        torch.nn.init.zeros_(self.lin1.bias)

        # output layers with lower init gain
        torch.nn.init.orthogonal_(self.lin_actor_mu.weight, 0.01)
        torch.nn.init.zeros_(self.lin_actor_mu.bias)
        torch.nn.init.orthogonal_(self.lin_actor_var.weight, 0.01)
        torch.nn.init.zeros_(self.lin_actor_var.bias)
        torch.nn.init.orthogonal_(self.lin_critic.weight, 0.1)
        torch.nn.init.zeros_(self.lin_critic.bias)

    def forward(self, state):
        # obtain features
        z = self.lin0(state)
        z = self.act0(z)
        z = self.lin1(z)
        z = self.act1(z)

        # obtain actor and critic outputs
        mu     = self.lin_actor_mu(z)
        mu     = self.act_mu(mu)

        var     = self.lin_actor_var(z)
        var     = self.act_var(var)

        value  = self.lin_critic(z)

        return mu, var, value
    