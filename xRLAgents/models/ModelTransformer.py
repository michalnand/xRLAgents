import torch
import numpy


class ModelInput(torch.nn.Module):
    def __init__(self, n_tokens, d_model, max_length = 128):
        super(ModelInput, self).__init__()

        self.embedding = torch.nn.Embedding(n_tokens, d_model)

        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # frequency scaling
        div_term = torch.exp(torch.arange(0, d_model, dtype=torch.float) * (-numpy.log(10000.0) / d_model))

        self.pos_encoding = torch.cos(position * div_term)        
        self.pos_encoding = self.pos_encoding.unsqueeze(0)


    def forward(self, x):
        y = self.embedding(x) + self.pos_encoding.detach().to(x.device)
        return y


class ModelTransformer(torch.nn.Module):
    def __init__(self, d_model = 256, n_heads=4, num_layers=4, n_hidden = 512):
        super(ModelTransformer, self).__init__()

        # Transformer Encoder Layers
        encoder_layer    = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=n_hidden, dropout=0.1, activation=torch.nn.functional.silu)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)


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
    Transformer model for Atari RAM learning
    FC layers for actor + critic
'''
class ModelRLTransformer(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ModelRLTransformer, self).__init__()

        max_length = input_shape[0]
        n_tokens   = 256
        d_model    = 256
        n_heads    = 4
        num_layers = 4

        self.model_in       = ModelInput(n_tokens, d_model, max_length)
        self.model_features = ModelTransformer(d_model, n_heads, num_layers)

        self.model_actor_critic = ModelActorCritic(d_model, 2*d_model, n_actions, 1)


    def forward(self, state):
        # obtain features
        x = torch.clip(state.long(), 0, 255)

        z = self.model_in(x)
        z = self.model_features(z)

        z = z.mean(dim=1)

        # obtain actor and critic outputs
        logits, value = self.model_actor_critic(z)

        return logits, value
    


        
if __name__ == "__main__":

    input_shape = [128]
    n_actions = 18
    x = torch.randint(0, 255, size=(10, 128))
    
    model = ModelRLTransformer(input_shape, n_actions)

    print(model)

    logits, value  = model(x)

    print(logits.shape, value.shape)