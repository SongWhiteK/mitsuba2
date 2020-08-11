"""
Processing trainingdata
"""

import glob
import numpy as np
import pandas as pd
import cv2
import utils


class DataHandler:
    """
    csv data handler for post proccessing
    """

    def __init__(self, config):
        self.file_list = glob.glob(f"{config.SAMPLE_DIR}\\*")
        self.train_image_dir_path = f"{config.IMAGE_DIR}"
        self.map_dir_path = f"{config.MAP_DIR}"
        self.train_sample_dir_path = f"{config.TRAIN_DIR}"

    def generate_train_data(self, offset=0):
        """
        Generating height map w.r.t incident point and medium parameters
        Args:
            offset: ID offset
        """
        height_map = None
        id_data = 0

        # Process with given csv files
        for i, file_name in enumerate(self.file_list):
            # get id number and map number
            data = pd.read_csv(file_name)
            data["id"] = data.index + id_data + offset

            # Generate training data with id
            data.to_csv(f"{self.train_sample_dir_path}\\train_path{i:02}.csv", index=False)

            # Get entire height map for a training data
            height_map_path = glob.glob(f"{self.map_dir_path}\\height_map{i:02}.png")
            height_map = cv2.imread(height_map_path[0])
        
            # Process with each training data 
            for row in data.itertuples():
                image = gen_train_image(row, height_map)

                # Save height map image with id
                cv2.imwrite(f"{self.image_dir_path}\\train_image{row.id:06}.png", image)
                id_data += 1


def join_scale_factor(path, scale):
    """
    Add scale factor of shape objects to the sampled data from given path

    Args:
    path: Path of a file which contains sampled data
    scale: Scale factor of shape objects
    """
    data = pd.read_csv(path)
    data["scale_x"] = scale["scale_x"]
    data["scale_y"] = scale["scale_y"]
    data["scale_z"] = scale["scale_z"]

    data.to_csv(path, index=False)
