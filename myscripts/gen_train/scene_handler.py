"""
Generate scene format in python
"""
import sys
import glob
import numpy as np
import pandas as pd
import utils
import mitsuba

mitsuba.set_variant("scalar_rgb")

from mitsuba.core import Transform4f
from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file, load_string
from mitsuba.python.util import traverse
from data_handler import join_scale_factor, join_model_id

class SceneGenerator:
    def __init__(self, config):
        self.spp = config.spp
        self.seed = 10
        self.xml_path = config.XML_PATH
        self.out_dir = config.SAMPLE_DIR
        self.scale_m = 1  # In mitsuba, world unit distance is [mm]
        self.serialized = None
        if(self.out_dir is None):
            print("\033[31m" + "Please set out put directory" + "\033[0m")
            self.out_dir = "."


        # set initial fixed medium
        medium_ini = utils.FixedParamGenerator().sample_params()
        self.set_medium(medium_ini)

        # set initial transform matrix
        self.mat = utils.scale_mat_2_str(np.eye(4))

    def set_medium(self, medium):
        self.eta = medium["eta"]
        self.g = medium["g"]
        self.albedo = medium["albedo"]
        self.sigmat = medium["sigma_t"]

    def set_serialized_path(self, serialized_path):
        self.serialized = serialized_path

    def set_out_path(self, model_id):
        self.out_path = f"{self.out_dir}\\sample{model_id:02}.csv"

    def set_transform_matrix(self, mat):
        self.mat = utils.scale_mat_2_str(mat)

    def random_set_transform_matrix(self, config):
        scale_factor = np.ones(3) * 0.25 + np.random.rand(3) * 2.75
        scale_mat = np.diag(scale_factor)
        self.set_transform_matrix(scale_mat)

        if(config.DEBUG):
            print(scale_mat)

        return scale_factor

    def get_scene(self, config):
        """
        Generate scene object from attributes
        """
        if (self.serialized is None):
            sys.exit("Please set serialized file path before generating scene")


        # Generate scene object
        if (config.mode is "sample"):
            scene = load_file(self.xml_path,
                              out_path=self.out_path, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              g=self.g, eta=self.eta,
                              serialized=self.serialized, mat=self.mat)

        elif (config.mode is "visual"):
            scene = load_file(self.xml_path, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              g=self.g, eta=self.eta,
                              serialized=self.serialized, mat=self.mat)

        elif (config.mode is "test"):
            scene = load_file(self.xml_path,
                              out_path=self.out_path, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              g=self.g, eta=self.eta)

        return scene


def render(itr, config):
    spp = config.spp

    # Generate scene and parameter generator
    param_gen = utils.ParamGenerator()
    scene_gen = SceneGenerator(config)
    cnt = 0

    # Ready for recording scale factor
    scale_rec = np.ones([itr, 3])

    # Get serialized file name list from serialized file directory
    serialized_list = glob.glob(f"{config.SERIALIZED_DIR}\\*.serialized")
    if(config.DEBUG):
        print(serialized_list)

    if(itr % config.scene_batch_size != 0):
        sys.exit("Please set ite_per_shape to be a multiple of scene_batch_size")

    # Render with given params generator and scene generator
    for model_id, serialized in enumerate(serialized_list):
        # Set serialized model path and output csv file path to scene generator
        scene_gen.set_serialized_path(serialized)
        scene_gen.set_out_path(model_id)

        scale_rec = np.ones([itr, 3])
    
        for i in range(itr // config.scene_batch_size):
            if (not config.scale_fix):
                # Sample scaling matrix and set
                scale_rec_v = scene_gen.random_set_transform_matrix(config)
                scale_rec[config.scene_batch_size * i:config.scene_batch_size * i + config.scene_batch_size] *= scale_rec_v

            
            # Generate scene object
            scene = scene_gen.get_scene(config)

            # Render the scene with scene_batch_size iteration
            for j in range(config.scene_batch_size):
                if(config.mode is "visual"):
                    sensor = scene.sensors()[0]
                else:
                    # Set sampler's seed and generate new sensor object
                    seed = np.random.randint(1000000)
                    sensor = get_sensor(spp, seed)

                # If medium parameters are not fixed, sample medium parameters again and update
                if not config.medium_fix:
                    medium = param_gen.sample_params()
                    update_medium(scene, medium)

                # Render the scene with new sensor and medium parameters
                scene.integrator().render(scene, sensor)

                # If visualize is True, develop the film
                if (config.mode is "visual"):
                    film = scene.sensors()[0].film()
                    bmp = film.bitmap(raw=True)
                    bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8,
                                srgb_gamma=True).write('visualize_{}.jpg'.format(i))

                cnt += 1

        # Join scale factors and model id to the sampled data
        df_scale_rec = pd.DataFrame(scale_rec, columns=["scale_x", "scale_y", "scale_z"])
        join_scale_factor(scene_gen.out_path, df_scale_rec)
        df_model_id = pd.DataFrame(np.ones([itr,1], dtype="uint8") * model_id, columns=["model_id"])
        join_model_id(scene_gen.out_path, df_model_id)





            



def get_sensor(spp, seed):
    """
    Generate new sensor object

    Args:
        spp: Sample per pixel of sensor's sampler
        seed: Seed of sensor's sampler

    Returns:
        sensor: Sensor object with a sampler including given spp and seed
    """
    sensor = load_string(
                        """<sensor version='2.2.1' type='perspective'>
                            <float name="fov" value="22.8952"/>
                            <float name="near_clip" value="0.01"/>
                            <float name="far_clip" value="100"/>

                            <sampler type="independent">
                                <integer name="sample_count" value="{}"/>
                                <integer name="seed" value="{}"/>
                            </sampler>
                        </sensor >""".format(spp, seed)
                         )

    return sensor



def update_medium(scene, medium):
    """
    Update medium parameters in a scene object

    Args:
        scene: Scene object including medium
        medium: List including medium parameters, i.e.,
                - albedo
                - eta (refractive index)
                - g (anisotropic index)
    """


    params = traverse(scene)
    params["Plane_001-mesh_0.interior_medium.albedo.color.value"] = medium["albedo"]
    params["medium_bsdf.eta"] = medium["eta"]
    params["myphase.g"] = medium["g"]
    params.update()



