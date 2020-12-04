"""
Configuration class for VAE
"""

class VAEConfiguration:
    def __init__(self):
        ##### Network #####
        self.use_dropout = True

        self.im_size = 63
        self.ch1    = 32
        self.ch2    = 64
        self.ch3    = 128

        self.stride = 2

        self.pool   = 3

        self.n_latent = 4

        self.n_fn   = 16
        self.n_enc  = 16
        self.n_dec1 = 64
        self.n_dec2 = 64

        self.loss_weight_pos    = 1
        self.loss_weight_abs    = 1000
        self.loss_weight_latent = 10


        ##### Trainer #####
        self.data = "plane"
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
            self.SAMPLE_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_paths_127plane\\train_path.csv"
            self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_images_127plane"
        else:
            self.n_per_subdir = 10000
            self.SAMPLE_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_paths_127\\train_path.csv"
            self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_images_127"
            

        self.seed = 1
        self.epoch = 10
        self.loader_args = {"batch_size": 512, "shuffle": True}
        self.lr = 2*1e-3

        self.LOG_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\log"
        self.MODEL_DIR = "C:\\User\\mineg\\mitsuba2\\myscripts\\vae\\model"

        






