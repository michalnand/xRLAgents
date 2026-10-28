PPOAgentDiffusionDefaultHyperparameters = {
    "reward_ext_coeff"  :  2.0,
    "reward_int_coeff"  :  1.0,

    "gamma_ext"         :  0.998, 
    "gamma_int"         :  0.99, 

    "n_steps"           :  128,
    "ppo_batch_size"    :  512,  
    "ssl_batch_size"    :  128,  
    
    "n_epochs"          : 4, 
    "learning_rate"     : 0.0001,
    
    "eps_clip"          : 0.1,

    "adv_ext_coeff"     : 2.0,
    "adv_int_coeff"     : 1.0,

    "val_coeff"         : 0.5,
    "entropy_beta"      : 0.001,
    "lam"               : 0.95,
    "max_grad_norm"     : 0.5,

    "noise_level_train"     : 0.5,
    "noise_level_inference" : 0.0,
    "states_max_distance"   : 1,

    "state_normalization"   : True,
    "single_frame_im"       : True,
}