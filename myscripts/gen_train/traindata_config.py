import sys

"""
Configuration of generating train data
"""


class TrainDataConfiguration:
    def __init__(self):
        ############## EXECUTION MODES ##############
        self.DEBUG = False
        self.mode = "visual" # visualizing medium appearance
        self.medium_fix = True # generating fixed medium parameters
        self.scale_fix = True # no scale shape objects

        ############## FILE PATHS ##############
        if (self.mode is "visual"):
            self.XML_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\scene_templates\\visual_template.xml"
            self.spp = 64
        elif(self.mode is "sample"):
            self.XML_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\scene_templates\\sample_template.xml"
            self.spp = 1024
        elif(self.mode is "test") :
            self.XML_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\scene_templates\\test_template.xml"
            self.spp = 1024
        else:
            sys.exit("Specify the execution mode")

        self.SERIALIZED_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\scene_templates\\serialized"

        self.SAMPLE_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\sample_files"
        self.TRAIN_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_paths_63"
        self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\height_map"
        self.IMAGE_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_data\\train_images_63"

        self.plane = [False, False, False, True]

        ############## ITERATION NUMBERS ##############
        self.itr_per_shape = 1
        self.num_per_subdir = 10000
        if(self.scale_fix):
            self.scene_batch_size = 1
        else:
            self.scene_batch_size = 1

        ############## RANDOM NUMBER SEED ##############
        self.seed = 12


        ############## TAIN DATA TAG ##############
        self.tag = ["eff_albedo", "albedo", "g", "eta", "sigma_n", "p_in_x", "p_in_y", "p_in_z",
                    "p_out_x", "p_out_y", "p_out_z", "d_in_x", "d_in_y", "d_in_z",
                    "abs_prob", "height_max", "model_id", "id"]
        self.coeff_range = 6
        self.im_size = 63


        if(self.DEBUG):
            print("\033[31m" + "This execution is in DEBUG mode" + "\033[0m")
