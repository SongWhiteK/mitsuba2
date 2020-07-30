"""
Processing trainingdata
"""

import glob
import numpy as np
import pandas as pd
import json

class DataHandler:
    def __init__(self, dir_path):
        self.path = dir_path
        self.update()

    def update(self):
        self.csv_list = glob.glob(self.path)

    def csv_2_json(self, json_file):
        """
        Integrate some csv files in one file, and convert to json format
        Assume tags in csv files are wi_x, wi_y, wi_z, in_x, in_y, in_z, out_x, out_y, out_z
        """
        for csv_name in self.csv_list:
            n_sample = sum([1 for _ in open(csv_name)]) - 1
            data = {"wi":np.zeros([1,3]), "in":np.zeros([1,3]), "out":np.zeros([1,3])}
            sample = [data] * n_sample

            # get path data from csv files
            with open(csv_name) as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    sample[i]["wi"] = row["wi"]
                    sample[i]["in"] = row["in"]
                    sample[i]["out"] = row["out"]
            
            # Integrate to one csv file
            