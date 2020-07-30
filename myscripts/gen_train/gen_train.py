import os
import numpy as np
import random as rand
import mitsuba
import enoki as ek
import utils
import scene_handler
from utils import ParamGenerator
from data_handler import DataHandler


mitsuba.set_variant("scalar_rgb")

from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file


visualize = False
if (visualize):
    XML_PATH = "C:\\Users\mineg\mitsuba2\myscripts\gen_train\\visual_template.xml"
    spp = 256
else:
    XML_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\sample_template.xml"
    spp = 1024


CSV_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\csv_files"
# TO DO
# - glob multiple serialized path in a directory
SERIALIZED_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\leather.serialized"

# Add the scene directory to the FileResolver's search path

Thread.thread().file_resolver().append(os.path.dirname(XML_PATH))

# Generate scene object from scene generator
scene = scene_handler.generate_scene(XML_PATH, SERIALIZED_PATH, out_dir=CSV_DIR, visual=visualize, spp=spp)

# render the scene
scene.integrator().render(scene, scene.sensors()[0])

# After rendering, the rendered data is stored in the film
film = scene.sensors()[0].film()

if(visualize):
    # Write out rendering as high dynamic range OpenEXR file
    film.set_destination_file('output.exr')
    film.develop()


