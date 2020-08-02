"""
Configuration of generating train data
"""


class TrainDataConfiguration:
    def __init__(self):
        self.visualize = False
        self.mfix = True

        self.OUT_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\csv_files"
        if (self.visualize):
            self.XML_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\visual_template.xml"
            self.spp = 64
        else:
            self.XML_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\sample_template.xml"
            self.spp = 1024
        # TO DO
        # - glob multiple serialized path in a directory
        self.SERIALIZED_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\meshes\\leather_m.serialized"
