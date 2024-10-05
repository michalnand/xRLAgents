
class Config:
    def __init__(self):
        # agent hyperparameters
        self.gamma              = 0.99
        self.entropy_beta       = 0.001
        self.eps_clip           = 0.1 
        self.adv_coeff          = 1.0
        self.val_coeff          = 0.5

        self.steps              = 256
        self.batch_size         = 256
        
        self.training_epochs    = 4
        self.learning_rate      = 0.0001

        self.seq_length         = 16
