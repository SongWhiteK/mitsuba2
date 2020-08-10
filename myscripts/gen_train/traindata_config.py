import sys

"""
Configuration of generating train data
"""


class TrainDataConfiguration:
    def __init__(self):
        self.DEBUG = True
        self.mode = "visual" # visualizing medium appearance
        self.medium_fix = True # generating fixed medium parameters
        self.scale_fix = True # do not scale shape objects

        self.OUT_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\csv_files"
        if (self.mode is "visual"):
            self.XML_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\scene_templates\\visual_template.xml"
            self.spp = 256
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
        self.SERIALIZED_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\scene_templates\\meshes\\leather2.serialized"

        if(self.scale_fix or self.mode is "visual"):
            self.scene_batch_size = 1
        else:
            self.scene_batch_size = 1

        self.seed = 12


config = TrainDataConfiguration()
if(config.DEBUG):
    print("\033[31m" + "This execution is in DEBUG mode" + "\033[0m")
