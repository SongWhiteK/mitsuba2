"""
Configuration class for VAE
"""

class VAEConfiguration:
    def __init__(self):
        self.use_dropout = True

        ##### CONV #####
        self.ch1 = 32
        self.ch2 = 64
        self.ch3 = 128

        self.stride = 2

        self.pool = 3

        self.n_fn = 16
        self.n_enc = 16
        self.n_dec1 = 256
        self.n_dec2 = 64

        self.loss_weight_pos = 100
        self.loss_weight_abs = 5000

        






