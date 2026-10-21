PPOAgentDiscreteDefaultHyperparameters = {
    "gamma"         :  0.99, 
    "n_steps"       :  128,
    "batch_size"    :  1024,  
    
    "n_episodes"        : 4, 
    "learning_rate"     : 0.00025,
    
    "eps_clip"          : 0.2,
    "adv_coeff"         : 1.0,
    "val_coeff"         : 0.5,
    "entropy_beta"      : 0.001,
    "lam"               : 0.95,
    "max_grad_norm"     : 0.5,
}