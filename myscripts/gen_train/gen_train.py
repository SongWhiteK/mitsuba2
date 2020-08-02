import os
import numpy as np
import random as rand
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

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(config.XML_PATH))

# Generate scene object from scene generator
scene = scene_handler.generate_scene(config)

# Number of iteration
itr = 1

# render the scene
scene_handler.render(scene, itr, config.visualize)




