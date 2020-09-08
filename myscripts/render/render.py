import os
import mitsuba
import enoki as ek
import integrate
import render_config as config

mitsuba.set_variant(config.variant)

from mitsuba.core import Float, UInt32, UInt64, Vector2f, Vector3f
from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file
from mitsuba.render import ImageBlock

##### Setting scene #####
# Scene path
scenepath = "C:/Users/mineg/mitsuba2/myscripts/render/cbox/cbox.xml"

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(scenepath))

#Load the scene
scene = load_file(scenepath)

# Rendering settings
spp = 64
sample_per_pass = 64

integrate.render(scene, spp, sample_per_pass)