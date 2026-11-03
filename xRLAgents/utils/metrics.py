import torch

def compute_regression_metrics(y_pred, y_true, name_prefix = "", dp=6) -> dict:
    # Ensure tensors are flat 1D vectors
    y_pred = y_pred.view(-1).detach()
    y_true = y_true.view(-1).detach()
    
    # Core error calculations
    errors = y_true - y_pred
    abs_errors = torch.abs(errors)
    squared_errors = errors ** 2
    
    # 1. Mean Squared Error (MSE)
    mse     = torch.mean(squared_errors)
    mse_std = torch.std(squared_errors)
    
    # 2. Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(mse)
    
    # 3. Mean Absolute Error (MAE)
    mae = torch.mean(abs_errors)
    
    # 4. R-squared (R2 Score)
    ss_res = torch.sum(squared_errors)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2) + 1e-10
    
    r2 = 1.0 - (ss_res / ss_tot)
   
    result = {}

    result[name_prefix + "mse"] = round(mse.item(), dp)
    result[name_prefix + "mse_std"] = round(mse_std.item(), dp)
    result[name_prefix + "rmse"] = round(rmse.item(), dp)
    result[name_prefix + "mae"] = round(mae.item(), dp)
    result[name_prefix + "r2"] = round(r2.item(), dp)
    

    return result


def ssl_features_metrics(za, zb = None, name_prefix = "", dp=6) -> dict:

    za = za.detach()

    result = {}

    result[name_prefix + "z_mean"]        = round(za.mean().item(), dp)
    result[name_prefix + "z_std"]         = round(za.std().item(), dp)
    result[name_prefix + "z_mag_mean"]    = round((za**2).mean().item(), dp)
    result[name_prefix + "z_mag_std"]     = round((za**2).std().item(), dp)

    if zb is not None:
        zb = zb.detach()

        shuffled_idx             = torch.randperm(za.shape[0])

        dz  = za - zb
        dzs = za - zb[shuffled_idx]

        result["dz_sim_mag_mean"]        = round((dz**2).mean().item(), dp)
        result["dz_sim_mag_std"]         = round((dz**2).std().item(), dp)
        result["dz_shuffled_mag_mean"]   = round((dzs**2).mean().item(), dp)
        result["dz_shuffled_mag_std"]    = round((dzs**2).std().item(), dp)

    return result