"""
Configuration class for VAE
"""
import sys
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
        self.data = "mini"
        if(self.data == "test"):
            self.n_per_subdir = 250
            self.SAMPLE_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\test_paths\\train_path.csv"
            self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\test_images"

        elif(self.data == "mini"):
            self.n_per_subdir = 10
            self.SAMPLE_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_paths_mini\\train_path.csv"
            self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_images_mini"
            self.TEST_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\test_paths_mini\\train_path.csv"
            self.TEST_MAP = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\test_images_mini"

        elif(self.data == "plane"):
            self.n_per_subdir = 10000
            self.SAMPLE_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_paths_63plane\\train_path.csv"
            self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_images_63plane"

        elif(self.data == "full"):
            self.n_per_subdir = 10000
            self.SAMPLE_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_paths\\train_path.csv"
            self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_images"
            self.TEST_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\test_paths\\train_path.csv"
            self.TEST_MAP = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\test_images"

        else:
            sys.exit("Specified data is invalid")
            
        self.model_name = "train_0206_full"
        self.offset = 1
        self.model_path = f"myscripts/vae/model/{self.model_name}/{self.model_name}_{self.offset:02}.pt"

        self.seed = 3
        self.epoch = 5
        self.loader_args = {"batch_size": 32, "shuffle": True}
        self.decay_rate = 0.8
        self.lr = 2*1e-4 * (self.decay_rate ** self.offset)


        self.LOG_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\log"
        self.MODEL_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\vae\\model"

        self.visualize_net = False
        






