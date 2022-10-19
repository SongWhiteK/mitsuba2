import sys
import datetime

"""
Configuration of generating train data
"""

dt_now = datetime.datetime.now()
class TrainDataConfiguration:
    def __init__(self):
        ############## EXECUTION MODES ##############
        self.DEBUG = False

        # This configuration has 4 modes
        # 1. visual
        #   Render the image according to scene_templates/visual_template.xml.
        #   This mode is used to check appearance of medium, shape and camera position.
        # 2. sample
        #   Generate training data.
        # 3. test
        #   This mode is similar to "sample" mode but this mode is simpler scene setting
        #   such as plane geometory and fixed incident position.
        #   This mode is used to get distribution of out going position and absorption probability
        # 4. abs
        #   This mode is similar to "test" mode but this mode samples with various incident angles
        #   This mode is used to plot absorption probability for sphere coordinates
        #   To plot absorption probability with data from this mode, use function map_abs() in myscripts/data_process/plots.py

        # visualizing medium appearance
        self.mode = "sample_per_d"
        # generating fixed medium parameters
        self.medium_fix = True
        # no scale shape objects
        self.scale_fix = True
        self.roop_num = 1
        self.res = 30

        ############## FILE PATHS ##############
        if (self.mode == "visual"):
            self.XML_PATH = "D:\\kenkyu\\mine\\mitsuba2\\myscripts\\gen_train\\scene_templates\\visual_template.xml"
            self.spp = 4096
            self.plane = [False, False, False, True, False, False]
        elif(self.mode == "sample"):
            self.XML_PATH = "D:\\kenkyu\\mine\\mitsuba2\\myscripts\\gen_train\\scene_templates\\sample_template.xml"
            self.spp = 1024
            self.plane = [False, False, False, True, False, False]
        elif(self.mode == "test" or self.mode == "abs"):
            self.init_d = "0,0,1"
            self.XML_PATH = "D:\\kenkyu\\mine\\mitsuba2\\myscripts\\gen_train\\scene_templates\\test_template.xml"
            self.spp = 1024
            self.plane = [True]

        elif(self.mode == "sample_per_d"):
            self.XML_PATH = "D:\\kenkyu\\mine\\mitsuba2\\myscripts\\gen_train\\scene_templates\\sample_per_d_template.xml"
            self.spp = 1024
            self.plane = [False, False, False, True, False, False]
            self.roop_num = 1


            
        else:
            sys.exit("Specify the execution mode")

        # Serialized file directory path
        self.SERIALIZED_DIR = "D:\\kenkyu\\mine\\mitsuba2\\myscripts\\gen_train\\scene_templates\\serialized"

        # Sample file directory path
        if(self.mode != "sample_per_d"):
            self.SAMPLE_DIR = "D:\\kenkyu\\mine\\train_data\\sample_files"
        else:
            self.SAMPLE_DIR = f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\roop_sample\\sample_files_test\\sample_files_{dt_now.day}_{dt_now.hour}_{dt_now.minute}_{dt_now.second}"
            self.SAMPLE_MAIN_DIR = "D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\roop_sample"
            self.SAMPLE_CSV_DIR = "D:\\kenkyu\\mine\\test_spt\\train_path.csv"
        # Final training data file directory path
        self.TRAIN_DIR = "D:\\kenkyu\\mine\\test"
        # Uncliped height map directory path
        self.MAP_DIR = "D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\height_map"
        # Height map directory path for training data
        self.IMAGE_DIR = "D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\height_save"

        ############## ITERATION NUMBERS ##############
        # The number of samples for one object, i.e, object file
        # in scene_templates/serialized/***.serialized
        self.itr_per_shape = 1000
        if self.mode == "abs":
            self.itr_per_shape = (self.res**2) * 6

        # The number of images of training data in subdirectory
        self.num_per_subdir = 100

        # The number of samples for one object, i.e., an object is rescaled
        # after sampling of this number of times, if scale_fix is false
        if(self.scale_fix):
            self.scene_batch_size = 1
        else:
            self.scene_batch_size = 1

        ############## RANDOM NUMBER SEED ##############
        self.seed = 98


        ############## TAIN DATA TAG ##############
        self.tag = ["eff_albedo", "albedo", "g", "eta", "sigma_n", "p_in_x", "p_in_y", "p_in_z",
                    "p_out_x", "p_out_y", "p_out_z", "d_in_x", "d_in_y", "d_in_z",
                    "abs_prob", "height_max", "model_id", "id"]
        self.coeff_range = 6
        self.im_size = 128

        self.scale = 1


        if(self.DEBUG):
            print("\033[31m" + "This execution is in DEBUG mode" + "\033[0m")
