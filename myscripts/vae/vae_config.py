"""
Configuration class for VAE
"""

class VAEConfiguration:
    def __init__(self):
        ##### Network #####
        self.use_dropout = True

        self.ch1    = 32
        self.ch2    = 64
        self.ch3    = 128

        self.stride = 2

        self.pool   = 3

        self.n_fn   = 16
        self.n_enc  = 16
        self.n_dec1 = 256
        self.n_dec2 = 64

        self.loss_weight_pos    = 1
        self.loss_weight_abs    = 400
        self.loss_weight_latent = 50000

        ##### Trainer #####
        self.data = "mini"
        if(self.data == "test"):
            self.n_per_subdir = 250
            self.SAMPLE_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\test_paths\\train_path.csv"
            self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\test_images"
        elif(self.data == "mini"):
            self.n_per_subdir = 1000
            self.SAMPLE_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_paths_mini\\train_path.csv"
            self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_images_mini"
        elif(self.data == "plane"):
            self.n_per_subdir = 10000
            self.SAMPLE_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_paths_plane\\train_path.csv"
            self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_images_plane"
        else:
            self.n_per_subdir = 10000
            self.SAMPLE_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_paths\\train_path.csv"
            self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_images"
            

        self.seed = 1
        self.epoch = 20
        self.loader_args = {"batch_size": 128, "shuffle": True}
        self.lr = 1*1e-4

        self.LOG_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\log"
        self.MODEL_DIR = "C:\\User\\mineg\\mitsuba2\\myscripts\\vae\\model"

        






