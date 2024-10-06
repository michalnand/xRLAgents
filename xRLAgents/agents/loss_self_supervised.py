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


def loss_vicreg_func(za, zb):
    # invariance loss
    inv_loss = loss_mse_func(za, zb)

    # variance loss
    std_loss = loss_std_func(za)
    std_loss+= loss_std_func(zb)
   
    # covariance loss 
    cov_loss = loss_cov_func(za)
    cov_loss+= loss_cov_func(zb)
   
    # total vicreg loss
    loss = 1.0*inv_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    #info for log
    z_mag     = ((za**2).mean()).detach().cpu().numpy().item()
    z_mag_std = ((za**2).std()).detach().cpu().numpy().item()

    inv_loss  = inv_loss.detach().cpu().numpy().item()
    std_loss  = std_loss.detach().cpu().numpy().item()
    cov_loss  = cov_loss.detach().cpu().numpy().item()
    
    info = {}
    info["mag_mean"] = z_mag
    info["mag_std"]  = z_mag_std

    info["inv"] = inv_loss
    info["std"] = std_loss
    info["cov"] = cov_loss

    return loss, info


def loss_contrastive_vicreg_func(za, zb):

    # similarity loss
    sim_loss = ((za - zb)**2).mean(dim=1)
    sim_loss = sim_loss.mean()

    # disimilarity loss
    # random shuffle
    idx = torch.randperm(zb.shape[0])
    zb_perm = zb[idx]

    dif_loss = torch.relu(1.0 - ((za - zb_perm)**2).mean(dim=1))
    dif_loss = dif_loss.mean()


    # variance loss
    std_loss = loss_std_func(za)
    std_loss+= loss_std_func(zb)
   
    # covariance loss 
    cov_loss = loss_cov_func(za)
    cov_loss+= loss_cov_func(zb)
   
    # total contrastive vicreg loss
    loss = 1.0*sim_loss + 1.0*dif_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    #info for log
    z_mag     = ((za**2).mean()).detach().cpu().numpy().item()
    z_mag_std = ((za**2).std()).detach().cpu().numpy().item()

    sim_loss  = sim_loss.detach().cpu().numpy().item()
    dif_loss  = dif_loss.detach().cpu().numpy().item()
    std_loss  = std_loss.detach().cpu().numpy().item()
    cov_loss  = cov_loss.detach().cpu().numpy().item()
    
    info = {}
    info["mag_mean"] = z_mag
    info["mag_std"]  = z_mag_std

    info["sim"] = sim_loss
    info["dif"] = dif_loss
    info["std"] = std_loss
    info["cov"] = cov_loss

    return loss, info