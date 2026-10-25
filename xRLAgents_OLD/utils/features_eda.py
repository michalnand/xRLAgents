import torch
import os
import json


def _compute_variance_spectrum(z):
    """
    Computes variance thresholds and Effective Rank of the representations.
    """
    z = z.detach()
    z_centered = z - z.mean(dim=0, keepdim=True)

    # Singular values
    S = torch.linalg.svdvals(z_centered)
    eigenvalues = S ** 2
    
    # 1. Variance thresholds
    cum_var_ratio = torch.cumsum(eigenvalues, dim=0) / eigenvalues.sum()
    
    def get_n_features(threshold: float) -> int:
        idx = (cum_var_ratio >= threshold).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            return idx[0].item() + 1 
        return z.shape[1]   
        
    return {
        "exp_var_1s": get_n_features(0.68),
        "exp_var_2s": get_n_features(0.95),
        "exp_var_3s": get_n_features(0.997),
    }

def _compute_covariance(z):
    """
    Computes the off-diagonal feature redundancy.
    """
    z = z.detach()
    batch_size, dim = z.shape
    
    # Needs at least 2 samples for covariance
    if batch_size < 2:
        return {"cov_off_diag_sq": 0.0}
        
    z_centered = z - z.mean(dim=0, keepdim=True)
    
    # Compute empirical covariance matrix (Dim x Dim)
    cov = (z_centered.T @ z_centered) / (batch_size - 1)
    
    # Create mask to select only off-diagonal elements
    mask = ~torch.eye(dim, dtype=torch.bool, device=z.device)
    off_diag_elements = cov[mask]
    
    # Return mean of squared off-diagonal elements (VICReg style)
    return {
        "cov_off_diag_sq": (off_diag_elements ** 2).mean().item()
    }

def features_eda(za, zb = None, output_prefix_str = "", dp = 5):

    z_mag = (za**2).mean()
    z_std = za.std()

    # mean and std
    result = {}
    result[output_prefix_str + "dim"]    = za.shape[1]  
    result[output_prefix_str + "l2_mag"] = round(z_mag.item(), dp)
    result[output_prefix_str + "std"]    = round(z_std.item(), dp)

    # eigen spectrum, how many features required to explain variance
    spectrum   = _compute_variance_spectrum(za)
    covariance = _compute_covariance(za)

    result = result | spectrum | covariance

    # positive and negative (random shuffled) pairs cosine and euclidean similarity
    # negative paris obtained by random shuffle of zb
    if zb is not None:    
        perm_idx = torch.randperm(za.shape[0], device=za.device)

        # cosine similarity
        pos_cos = torch.nn.functional.cosine_similarity(za, zb, dim=-1).mean().item()
        neg_cos = torch.nn.functional.cosine_similarity(za, zb[perm_idx], dim=-1).mean().item()

        result[output_prefix_str + "pos_cos"] = round(float(pos_cos), dp)
        result[output_prefix_str + "neg_cos"] = round(float(neg_cos), dp)

        # euclidean distances
        d_pos    = ((za - zb)**2).mean(dim=-1)
        d_neg    = ((za - zb[perm_idx])**2).mean(dim=-1)
        
        pos_dist        = d_pos.mean().item()
        neg_dist        = d_neg.mean().item()
        pos_dist_std    = d_pos.std().item()
        neg_dist_std    = d_neg.std().item()  

        result[output_prefix_str + "pos_dist"]      = round(pos_dist, dp)
        result[output_prefix_str + "neg_dist"]      = round(neg_dist, dp)
        result[output_prefix_str + "pos_dist_std"]  = round(pos_dist_std, dp)
        result[output_prefix_str + "neg_dist_std"]  = round(neg_dist_std, dp)

    
    return result



class FeaturesEDA:

    def __init__(self, result_path):
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        self.log_file_name = result_path + "features_eda.jsonl"
        f = open(self.log_file_name, "w")
        f.close()
        print("creating log file ", self.log_file_name) 


    def __call__(self, iteration, za, zb = None, dp = 5):
        result_log = {"iteration" : iteration}
        result = features_eda(za, zb, "", dp)

        result_log = result_log | result

        f = open(self.log_file_name, "a+")
        result_str = json.dumps(result_log)
        f.write(result_str + "\n")
        f.close() 
        