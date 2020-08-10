import os
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

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(config.XML_PATH))

# Number of iteration
itr = config.itr

start = time.time()
print("Sampling start")

# render the scene
scene_handler.render(itr, config)

process_time = time.time() - start
print("Sampling end (took {} s)".format(process_time))





