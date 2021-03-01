"""Generate training data, rendering image and so on"""

import os
import datetime
import pandas as pd
import numpy as np
import random as rand
import time
import mitsuba
import enoki as ek
import utils
import scene_handler
from utils import ParamGenerator
from data_handler import DataHandler
from traindata_config import TrainDataConfiguration


mitsuba.set_variant("scalar_rgb")

from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file


# Configuration
config = TrainDataConfiguration()

np.random.seed(seed=config.seed)

d_handler = DataHandler(config)

# Delete Existing smaple file with conforming
d_handler.delete_sample_files()

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(config.XML_PATH))

# Number of iteration
itr = config.itr_per_shape

start = time.time()
print("Sampling start")

# render the scene
scene_handler.render(itr, config)

process_time = time.time() - start
print("Sampling end (took {} s)".format(process_time))

if config.mode is "visual":
    data  = pd.DataFrame([[str(datetime.datetime.now()), process_time]],
                         columns=["date", "time"])

    data.to_csv("log_pt.csv", mode="a", header=(not os.path.exists("log_pt.csv")))

print("Process with result data")
d_handler.generate_train_data()





