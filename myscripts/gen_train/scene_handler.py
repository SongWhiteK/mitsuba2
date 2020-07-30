"""
Generate scene format in python
"""
import sys
import utils
import mitsuba

mitsuba.set_variant("scalar_rgb")

from mitsuba.core.xml import load_file

class SceneGenerator:
    def __init__(self, xml_path, out_dir, serialized_path, spp):
        self.eta = 1.
        self.g = 0.
        self.albedo = 1.
        self.sigmat = 10.
        self.spp = spp
        self.seed = 4
        self.xml_path = xml_path
        self.out_dir = out_dir
        self.scale_m = 1000  # In mitsuba, world unit distance is [m]
        self.seriarized = serialized_path


    def set_medium(self, medium):
        self.eta = medium["eta"]
        self.g = medium["g"]
        self.albedo = medium["albedo"]
        self.sigmat = medium["sigma_t"]

    def seed(self, seed):
        self.seed = seed
        
    def set_serialized_path(self, serialized_path):
        self.serialized = serialized_path

    def set_spp(self, spp):
        self.spp = spp




    def get_scene(self, visual=False):
        """
        Generate scene object from attributes  
        For now, 
            - medium parameters
            - sampler seed
            - sample per pixel
            - output directory path
        can be changed
        """
        if (self.seriarized is None):
            sys.exit("Please set serialized obj path before generating scene")

        # Generate scene object
        if (not visual):
            scene = load_file(self.xml_path,
                            out_dir=self.out_dir, spp=self.spp, seed=self.seed,
                            scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                            g=self.g, eta=self.eta,
                            serialized=self.seriarized)

        elif (visual):
            scene = load_file(self.xml_path, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              g=self.g, eta=self.eta,
                              serialized=self.seriarized)

        return scene


def generate_scene(xml_path, serialized_path, out_dir=None, visual=False, spp=1024):
    """
    Sample medium parameters randomly, and
    generate scene object
    """
    scene = None

    if (out_dir is None):
        out_dir = ".\\"

    # Instanciate scene generator
    scene_gen = SceneGenerator(xml_path, out_dir, serialized_path, spp)

    # Sample medium parameters and get medium dictionary
    param_gen = utils.ParamGenerator(seed=10)
    medium = param_gen.sample_params()
    print(medium)
    print("sigma_n of the medium: {}".format(utils.get_sigman(medium)))

    scene_gen.set_medium(medium)

    scene = scene_gen.get_scene(visual)

    return scene





    



