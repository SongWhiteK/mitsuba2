"""
Configuration class for VAE
"""

class VAEConfiguration:
    def __init__(self):
        ##### Network #####
        self.use_dropout = True

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

        ##### Trainer #####
        self.SAMPLE_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_paths\\train_path.csv"
        self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_images"

        self.seed = 1
        self.epoch = 10
        self.loader_args = {"batch_size": 32, "shuffle": False}
        self.lr = 2*1e-4

        






