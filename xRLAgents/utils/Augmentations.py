import torch

#apply random agumentation
def aug_random_apply(x, p, aug_func):
    mask        = (torch.rand(x.shape[0]) < p).float().to(x.device)
    mask_tmp    = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    y           = (1.0 - mask_tmp)*x + mask_tmp*aug_func(x)
 
    return y, mask  

#uniform aditional noise
def aug_noise(x, k = 0.2): 
    pointwise_noise = k*(2.0*torch.rand(x.shape, device=x.device) - 1.0)
    return x + pointwise_noise  

# random black mask
def aug_mask(x, p = 0.75, gw = 16, gh = 16):
    up_h = x.shape[2]//gh
    up_w = x.shape[3]//gw 

    mask = torch.rand((x.shape[0], x.shape[1], gh, gw), device = x.device)
    
    mask = torch.nn.functional.interpolate(mask, scale_factor = (up_h, up_w), mode="bicubic")
    mask = (mask > (1.0 - p)).float().detach()

    return mask*x 


def Augmentations(aug_names, x):
    if "mask" in aug_names:
        x, _ = aug_random_apply(x, 0.5, aug_mask)
    
    if "noise" in aug_names:
        x, _ = aug_random_apply(x, 0.5, aug_noise)
    
    return x