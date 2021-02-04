"""Basic scene file format as dict type"""

import sys
sys.path.append("myscripts/render")

import mitsuba
import render_config as config

mitsuba.set_variant(config.variant)

from mitsuba.core import ScalarTransform4f

origin = [35, 70, 130]
target = [-5, -10, 40]
up = [0, 0, 1]

if config.zoom:
    origin = [10, 10, 75]
    target = [0, 0, 60]

scene_dict = {
    "type": "scene",

    "sensor":{
        "type": "perspective",
        "fov": 39.3077,
        "near_clip": 0.1,
        "far_clip": 1000,
        "fov_axis": "smaller",

        "to_world": ScalarTransform4f.look_at(origin=origin,
                                            target=target,
                                            up=up),

        "sampler": {
            "type": "independent",
            "sample_count": 16,
            "seed": config.seed
        },

        "film": {
            "type": "hdrfilm",
            "width": config.film_width,
            "height": config.film_height,
            "rfilter": {
                "type": "gaussian"
            }
        }
    },

    "emitter": {
        "type": "envmap",
        "filename": "C:/Users/mineg/mitsuba2/myscripts/render/dict/envmap.exr",
        "to_world": ScalarTransform4f.rotate([1,0,0], angle=90)
    },

    "checker": {
        "type": "checkerboard",
        "color0": 0.4,
        "color1": 0.2,
        "to_uv": ScalarTransform4f.scale(2)
    },

    "planemat": {
        "type": "diffuse",
        "reflectance": {
            "type": "ref",
            "id": "checker"
        }
    },

    "floor": {
        "type": "rectangle",
        "to_world": ScalarTransform4f.scale(150),
        "floor_bsdf": {
            "type": "ref",
            "id": "planemat"
        }
    }

    # "wall1": {
    #     "type": "rectangle",
    #     "to_world": ScalarTransform4f.translate([-150, 0, 150])
    #                 * ScalarTransform4f.rotate([0,1,0], angle=90)
    #                 * ScalarTransform4f.scale(150)
    # },

    # "wall2": {
    #     "type": "rectangle",
    #     "to_world": ScalarTransform4f.translate([0, -150, 150])
    #                 * ScalarTransform4f.rotate([1,0,0], angle=-90)
    #                 * ScalarTransform4f.scale(150)
    # }
    
}