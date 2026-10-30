import torch

'''
    - MLP actor critic for continuous actions
    - two hidden layers, SiLU activation
    - actor     : mean head (tanh), var head (softplus)
    - critic    : linear head

    - adjustable output heads  init gain
'''
class MLPActorCriticCont(torch.nn.Module):
    def __init__(self, input_shape, num_outputs, num_hidden = 256, actor_mu_gain = 0.01, actor_var_gain = 0.01, critic_gain = 0.01):
        super().__init__()
        
        self.lin_0 = torch.nn.Linear(input_shape[0], num_hidden)
        self.act_0 = torch.nn.SiLU()
        self.lin_1 = torch.nn.Linear(num_hidden, num_hidden)
        self.act_1 = torch.nn.SiLU()

        self.actor_mu       = torch.nn.Linear(num_hidden, num_outputs)
        self.actor_mu_act   = torch.nn.Tanh()
        self.actor_var      = torch.nn.Linear(num_hidden, num_outputs)
        self.actor_var_act  = torch.nn.Softplus()

        self.critic    = torch.nn.Linear(num_hidden, 1)

        torch.nn.init.orthogonal_(self.lin_0.weight, gain=0.5)
        torch.nn.init.zeros_(self.lin_0.bias)

        torch.nn.init.orthogonal_(self.lin_1.weight, gain=0.5)
        torch.nn.init.zeros_(self.lin_1.bias)

        torch.nn.init.orthogonal_(self.actor_mu.weight, gain=actor_mu_gain)
        torch.nn.init.zeros_(self.actor_mu.bias)
        torch.nn.init.orthogonal_(self.actor_var.weight, gain=actor_var_gain)
        torch.nn.init.zeros_(self.actor_var.bias)
        torch.nn.init.orthogonal_(self.critic.weight, gain=critic_gain)
        torch.nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        z = self.lin_0(x)
        z = self.act_0(z)
        z = self.lin_1(z)
        z = self.act_1(z)

        pi_mean = self.actor_mu(z)
        pi_mean = self.actor_mu_act(pi_mean)

        pi_var  = self.actor_var(z)
        pi_var  = self.actor_var_act(pi_var)

        value   = self.critic(z)

        return pi_mean, pi_var, value

