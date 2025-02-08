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

    d        = ((za - zb_perm)**2).mean(dim=1)
    dif_loss = torch.relu(1.0 - d)
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





def _images_ssim(imgs, kernel_size = 5):
    """Computes SSIM for all pairs in a batch using AvgPool2d.

    Args:
        imgs (torch.Tensor): Input batch of shape (n_images, 1, H, W).
        kernel_size (int): Window size for computing local statistics.

    Returns:
        torch.Tensor: SSIM matrix of shape (n_images, n_images).
    """
    n_images, _, H, W = imgs.shape
    pool = torch.nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    # Compute mean and variance for all images
    mu = pool(imgs)  # (n_images, 1, H, W)
    mu_sq = mu ** 2
    sigma_sq = pool(imgs ** 2) - mu_sq  # Variance

    # Expand dimensions for broadcasting
    mu1 = mu.unsqueeze(0)  # (1, n_images, 1, H, W)
    mu2 = mu.unsqueeze(1)  # (n_images, 1, 1, H, W)
    
    sigma1_sq = sigma_sq.unsqueeze(0)  # (1, n_images, 1, H, W)
    sigma2_sq = sigma_sq.unsqueeze(1)  # (n_images, 1, 1, H, W)

    # Compute covariance correctly by flattening to 4D for pooling
    imgs1 = imgs.unsqueeze(0).expand(n_images, -1, -1, -1, -1)  # (n_images, n_images, 1, H, W)
    imgs2 = imgs.unsqueeze(1).expand(-1, n_images, -1, -1, -1)  # (n_images, n_images, 1, H, W)
    
    sigma12 = pool((imgs1 * imgs2).reshape(n_images * n_images, 1, H, W))  # Reshape to (batch, 1, H, W)
    sigma12 = sigma12.reshape(n_images, n_images, 1, H, W) - mu1 * mu2  # Reshape back

    # SSIM constants
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    # Compute SSIM map
    # (n_images, n_images, 1, H, W)
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))  

    # Average over spatial dimensions
    # (n_images, n_images)
    ssim_matrix = ssim_map.mean(dim=(2, 3, 4))  

    return ssim_matrix  

def loss_vicreg_ssim(x, z, z_ssim):
    # structure similarity loss
    # contrastive term
    ssim_target = _images_ssim(x).detach()
    ssim_loss   = ((ssim_target - z_ssim)**2).mean()

    # variance loss
    std_loss = loss_std_func(z)
   
    # covariance loss 
    cov_loss = loss_cov_func(z)
   
    # total vicreg loss
    loss = 1.0*ssim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    #info for log
    z_mag     = ((z**2).mean()).detach().cpu().numpy().item()
    z_mag_std = ((z**2).std()).detach().cpu().numpy().item()

    ssim_loss   = ssim_loss.detach().cpu().numpy().item()
    std_loss    = std_loss.detach().cpu().numpy().item()
    cov_loss    = cov_loss.detach().cpu().numpy().item()
    
    info = {}
    info["mag_mean"] = z_mag
    info["mag_std"]  = z_mag_std

    info["ssim"]    = ssim_loss
    info["std"]     = std_loss
    info["cov"]     = cov_loss

    return loss, info