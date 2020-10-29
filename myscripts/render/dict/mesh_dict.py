"""Basic scene file format of meshes for bssrdf"""

import sys
import numpy as np

from PIL import Image
sys.path.append("myscripts/render")

import mitsuba
import render_config as config
import data_pipeline

mitsuba.set_variant(config.variant)

from mitsuba.core import ScalarTransform4f, ScalarVector3f


class meshes:
    """Super class for BSSRDF meshes"""

    def __init__(self):
        self.n_mesh = 6
        self.type = [None for i in range(self.n_mesh)]
        self.filename = [None for i in range(self.n_mesh)]
        self.translate = [None for i in range(self.n_mesh)]
        self.rotate = [None for i in range(self.n_mesh)]
        self.scale = [None for i in range(self.n_mesh)]
        self.height_max = [None for i in range(self.n_mesh)]
        self.map = [None for i in range(self.n_mesh)]

    def register_all_mesh(self, bdata):
        """Register whole meshes in this class to BSSRDF_Data"""

        for i in range(self.n_mesh):
            bdata.register_mesh(i+1, self.type, self.height_max, filename=self.filename[i],
                                translate=self.translate[i], rotate=self.rotate[i],
                                scale=self.scale[i]
                                )

    def register_params(self, bdata, ior=1.5, scale=1.0,
                        sigma_t=1.0, albedo=0.5, g=0.25):
        """Register same medium parameters to meshes of one object"""

        for i in range(self.n_mesh):
            i += 1
            bdata.register_medium(i, ior=ior, scale=scale,
                                  sigma_t=sigma_t, albedo=albedo, g=g)


class meshes_cube(meshes):
    """
    Rectangle meshes for BSSRDF
    Mesh ID indicates each mesh below,
    1: bottom
    2: side for positive y
    3: side for positive x
    4: side for negative y
    5: side for negative x
    6: top
    """

    def __init__(self):
        super(meshes_cube, self).__init__()
        self.n_mesh = 6
        self.type = "rectangle"
        self.translate = [
            ScalarVector3f([0, 0, 0.01]),
            ScalarVector3f([0, 30, 30.01]),
            ScalarVector3f([30, 0, 30.01]),
            ScalarVector3f([0, -30, 30.01]),
            ScalarVector3f([-30, 0, 30.01]),
            ScalarVector3f([0, 0, 60.01])
        ]
        self.rotate = [
            {"axis": "x", "angle": 180},
            {"axis": "x", "angle": -90},
            {"axis": "y", "angle": 90},
            {"axis": "x", "angle": 90},
            {"axis": "y", "angle": -90},
            {"axis": "x", "angle": 0}
        ]

        for i in range(self.n_mesh):
            self.scale[i] = ScalarVector3f(30)
            self.height_max[i] = 0
            self.map[i] = np.ones([512, 512], dtype="uint8") * 63



    