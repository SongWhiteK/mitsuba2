import os
import time
import mitsuba
import integrate
import render_config as config

mitsuba.set_variant(config.variant)

from mitsuba.core import Thread
from mitsuba.core.xml import load_file

##### Setting scene #####
# Scene path
scenepath = "C:/Users/mineg/mitsuba2/myscripts/render/cbox/cbox.xml"

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(scenepath))

#Load the scene
scene = load_file(scenepath)

# Rendering settings
spp = 2048
sample_per_pass = 64

print("Rendering start")

start = time.time()
integrate.render(scene, spp, sample_per_pass)
process_time = time.time() - start

print(f"Rendering end (took {process_time}s)")