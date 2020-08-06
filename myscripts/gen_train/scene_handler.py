"""
Generate scene format in python
"""
import sys
import numpy as np
import utils
import mitsuba

mitsuba.set_variant("scalar_rgb")

from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file, load_string
from mitsuba.python.util import traverse


class SceneGenerator:
    def __init__(self, xml_path, out_dir, serialized_path, spp):
        self.spp = spp
        self.seed = 10
        self.xml_path = xml_path
        self.out_dir = out_dir
        self.scale_m = 1  # In mitsuba, world unit distance is [mm]
        self.seriarized = serialized_path

    def set_medium(self, medium):
        self.eta = medium["eta"]
        self.g = medium["g"]
        self.albedo = medium["albedo"]
        self.sigmat = medium["sigma_t"]

    def set_serialized_path(self, serialized_path):
        self.serialized = serialized_path

    def get_scene(self, config):
        """
        Generate scene object from attributes  
        For now, 
            - medium parameters
            - sampler's seed
            - sample per pixel
            - output directory path
        can be changed
        """
        if (self.seriarized is None):
            sys.exit("Please set serialized obj path before generating scene")

        # Generate scene object
        if (config.mode is "sample"):
            scene = load_file(self.xml_path,
                              out_dir=self.out_dir, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              g=self.g, eta=self.eta,
                              serialized=self.seriarized)

        elif (config.mode is "visual"):
            scene = load_file(self.xml_path, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              g=self.g, eta=self.eta,
                              serialized=self.seriarized)

        elif (config.mode is "test"):
            scene = load_file(self.xml_path,
                              out_dir=self.out_dir, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              g=self.g, eta=self.eta)

        return scene


def generate_scene(config):
    """
    Sample medium parameters randomly, and
    generate scene object
    """
    scene = None
    out_dir = config.OUT_DIR
    xml_path = config.XML_PATH
    serialized_path = config.SERIALIZED_PATH
    spp = config.spp

    if (out_dir is None):
        out_dir = ".\\"

    # Instanciate scene generator
    scene_gen = SceneGenerator(xml_path, out_dir, serialized_path, spp)

    if (not config.mfix):
        # Sample medium parameters and get medium dictionary
        param_gen = utils.ParamGenerator(seed=10)
    else:
        param_gen = utils.FixedParamGenerator()

    medium = param_gen.sample_params()

    scene_gen.set_medium(medium)

    scene = scene_gen.get_scene(config)

    return scene


def render(scene, itr, config):
    np.random.seed(seed=10)
    spp = scene.sensors()[0].sampler().sample_count()
    param_gen = utils.ParamGenerator()

    for i in range(itr):
        if(config.mode is "visual"):
            sensor = scene.sensors()[0]
        else:
            # Set sampler's seed and generate new sensor object
            seed = np.random.randint(1000000)
            sensor = get_sensor(spp, seed)

        # If medium parameters are not fixed, sample medium parameters again and update
        if not config.mfix:
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


def get_sensor(spp, seed):
    """
    Generate new sensor object

    Args:
        spp: Sample per pixel of sensor's sampler
        seed: Seed of sensor's sampler

    Returns:
        sensor: Sensor object with a sampler including given spp and seed
    """
    sensor = load_string("""<sensor version='2.2.1' type='perspective'>
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

