import torch

def _off_diagonal(x):
    mask = 1.0 - torch.eye(x.shape[0], device=x.device)
    return x*mask 

# invariance loss
def loss_mse_func(xa, xb):
    return ((xa - xb)**2).mean()
    
# variance loss
def loss_std_func(x):
    eps = 0.0001 
    std_x = torch.sqrt(x.var(dim=0) + eps)

    loss = torch.mean(torch.relu(1.0 - std_x)) 
    return loss

# covariance loss 
def loss_cov_func(x):
    x_norm = x - x.mean(dim=0)
    cov_x = (x_norm.T @ x_norm) / (x.shape[0] - 1.0)
    
    loss = _off_diagonal(cov_x).pow_(2).sum()/x.shape[1] 
    return loss