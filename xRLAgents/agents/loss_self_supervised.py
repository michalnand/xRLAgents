import torch

def _off_diagonal(x):
    mask = 1.0 - torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
    return x*mask 

# invariance loss
def loss_mse_func(xa, xb):
    return ((xa - xb)**2).mean()
    
# variance loss
def loss_std_func(x, upper = 1.0):
    eps = 0.0001 
    std_x = torch.sqrt(x.var(dim=0) + eps)

    loss = torch.mean(torch.relu(upper - std_x)) 
    return loss

# covariance loss 
def loss_cov_func(x):
    x_norm = x - x.mean(dim=0)
    cov_x = (x_norm.T @ x_norm) / (x.shape[0] - 1.0)
    
    loss = _off_diagonal(cov_x).pow_(2).sum()/x.shape[1] 
    return loss


# covariance loss 
def loss_covariance(za, zb):
    mu_a = za.mean(dim=0, keepdim=True)  # (1, n_features)
    mu_b = zb.mean(dim=0, keepdim=True)  # (1, n_features)

    # center the matrices
    za_norm = za - mu_a  # (batch_size, n_features)
    zb_norm = zb - mu_b  # (batch_size, n_features)

    cov_matrix = (za_norm.T @ zb_norm) / (za.shape[0] - 1.0)

    loss = (cov_matrix**2).mean()

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
    z_mag     = ((za**2).mean()).float().detach().cpu().numpy().item()
    z_mag_std = ((za**2).std()).float().detach().cpu().numpy().item()

    inv_loss  = inv_loss.float().detach().cpu().numpy().item()
    std_loss  = std_loss.float().detach().cpu().numpy().item()
    cov_loss  = cov_loss.float().detach().cpu().numpy().item()
    
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



def images_ssim(batch_a: torch.Tensor, batch_b: torch.Tensor, kernel_size=7):
    """Computes SSIM between all pairs of images in two batches.

    Args:
        batch_a (torch.Tensor): First batch of shape (batch_a_size, C, H, W), values in range [0, 1].
        batch_b (torch.Tensor): Second batch of shape (batch_b_size, C, H, W), values in range [0, 1].
        kernel_size (int): Window size for computing local statistics.

    Returns:
        torch.Tensor: SSIM matrix of shape (batch_a_size, batch_b_size).
    """
    batch_a_size, C, H, W = batch_a.shape
    batch_b_size, _, _, _ = batch_b.shape
    pool = torch.nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2).to(batch_a.device)

    # Compute local mean & variance
    # returns (batch_a_size, C, H, W)
    # mean
    mu_a = pool(batch_a) 
    mu_b = pool(batch_b)

    # variance
    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    sigma_a_sq = pool(batch_a ** 2) - mu_a_sq
    sigma_b_sq = pool(batch_b ** 2) - mu_b_sq

    # Compute covariance: reshape before pooling
    batch_a_expanded = batch_a.unsqueeze(1).expand(-1, batch_b_size, -1, -1, -1)  # (batch_a_size, batch_b_size, C, H, W)
    batch_b_expanded = batch_b.unsqueeze(0).expand(batch_a_size, -1, -1, -1, -1)  # (batch_a_size, batch_b_size, C, H, W)

    # Reshape to apply pooling
    batch_product = (batch_a_expanded * batch_b_expanded).reshape(-1, C, H, W)  # (batch_a_size * batch_b_size, C, H, W)
    sigma_ab = pool(batch_product).reshape(batch_a_size, batch_b_size, C, H, W) - mu_a.unsqueeze(1) * mu_b.unsqueeze(0)

    # Correct SSIM constants for [0,1] range
    c1 = (0.01) ** 2  # = 0.0001
    c2 = (0.03) ** 2  # = 0.0009

    # Compute SSIM map
    tmp_a       = (2 * mu_a.unsqueeze(1) * mu_b.unsqueeze(0) + c1) * (2 * sigma_ab + c2)
    tmp_b       = (mu_a_sq.unsqueeze(1) + mu_b_sq.unsqueeze(0) + c1) * (sigma_a_sq.unsqueeze(1) + sigma_b_sq.unsqueeze(0) + c2)
    ssim_map    = tmp_a / tmp_b

    # Compute final SSIM matrix by averaging over channels, height, and width
    ssim_matrix = ssim_map.mean(dim=(2, 3, 4))  # Shape: (batch_a_size, batch_b_size)
    ssim_matrix = torch.clip(ssim_matrix, 0.0, 1.0)

    return ssim_matrix

    

class SIGRegLoss(torch.nn.Module):
    """
    SIGReg for generic features (no temporal dimension)
    Input: (B, D)
    """
    def __init__(self, knots=17, num_proj=512):
        super().__init__()
        self.num_proj = num_proj    

        # integration grid
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)

        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # trapezoidal rule

        # Gaussian characteristic function
        phi = torch.exp(-t.square() / 2.0)

        # windowing (same as paper)
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * window)

    def forward(self, z):
        """
        z: (B, D)
        """
        B, D = z.shape

        device = z.device

        # 1. sample random projection directions
        A = torch.randn(D, self.num_proj, device=device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-8)

        # 2. project → (B, M)
        proj = z @ A

        # 3. expand for t grid → (B, M, K)
        x_t = proj.unsqueeze(-1) * self.t.to(device)

        # 4. empirical characteristic function
        cos_term = torch.cos(x_t).mean(dim=0)  # (M, K)
        sin_term = torch.sin(x_t).mean(dim=0)  # (M, K)

        # 5. compute Epps–Pulley error
        err = (cos_term - self.phi.to(device)).pow(2) + sin_term.pow(2)

        # 6. integrate over t
        stat = err @ self.weights.to(device)  # (M,)

        # 7. scale by sample size (important!)
        stat = stat * B

        # 8. average over projections
        return stat.mean()



def loss_spectral_temporal_per_delta_func(Z, delta, lambda_offdiag=1.0):
    """
    Computes spectral temporal loss for a single lag delta.
    
    Args:
        Z: Tensor of shape (T, B, D)
        delta: temporal lag (int)
        
    Returns:
        scalar loss for this delta
    """
     
    T, B, D = Z.shape

    if delta >= T:
        raise ValueError("delta must be smaller than sequence length")
    
   

    Z_t  = Z[0:T-delta]
    Z_tp = Z[delta:T]

    X = Z_t.reshape(-1, D)
    Y = Z_tp.reshape(-1, D)

    # center
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    # normalize variance (important!)
    X = X / (X.std(0, keepdim=True) + 1e-4)
    Y = Y / (Y.std(0, keepdim=True) + 1e-4)

    N = X.shape[0]

    C = (X.T @ Y) / N

    diag = torch.diagonal(C)
    off_diag = C - torch.diag(diag)

    loss_diag = ((1 - diag) ** 2).mean()
    loss_off = (off_diag ** 2).mean()

    return loss_diag + lambda_offdiag * loss_off


def loss_spectral_temporal_func(Z, deltas=[1, 2, 4, 8]):
    """
    Multi-scale spectral temporal loss.
    """
    total = 0.0
    for delta in deltas:
        total += loss_spectral_temporal_per_delta_func(Z, delta)
    return total / len(deltas)