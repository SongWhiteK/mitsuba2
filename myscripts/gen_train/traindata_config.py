import sys

"""
Configuration of generating train data
"""


class TrainDataConfiguration:
    def __init__(self):
        self.DEBUG = True
        self.mode = "sample" # visualizing medium appearance
        self.medium_fix = False # generating fixed medium parameters
        self.scale_fix = False # do not scale shape objects

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

        # TO DO
        # - glob multiple serialized path in a directory
        self.SERIALIZED_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\scene_templates\\meshes\\leather1.serialized"

        self.SAMPLE_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\sample_files"
        self.TRAIN_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_paths"
        self.MAP_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\height_map"
        self.IMAGE_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\train_images"

        self.itr = 10
        if(self.scale_fix):
            self.scene_batch_size = 1
        else:
            self.scene_batch_size = 1

        self.seed = 12


config = TrainDataConfiguration()
if(config.DEBUG):
    print("\033[31m" + "This execution is in DEBUG mode" + "\033[0m")
