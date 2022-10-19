"""
Generate scene format in python
"""
from random import random
from re import A
import sys
import glob
from turtle import position
import numpy as np
import pandas as pd
import utils
import mitsuba
import enoki as ek
import time

mitsuba.set_variant("scalar_rgb")

from mitsuba.core import Transform4f, Vector3f
from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file, load_string
from mitsuba.python.util import traverse
from data_handler import delete_file, join_scale_factor, join_model_id

class SceneGenerator:
    """Scene object generator"""
    
    def __init__(self, config):
        self.spp = config.spp
        self.seed = 100
        self.xml_path = config.XML_PATH
        self.out_dir = config.SAMPLE_DIR
        self.scale_m = config.scale  # In mitsuba, world unit distance is [mm]
        self.serialized = None
        if(self.out_dir is None):
            print("\033[31m" + "Please set out put directory" + "\033[0m")
            self.out_dir = "."


        # set initial fixed medium
        medium_ini = utils.FixedParamGenerator().sample_params()
        self.set_medium(medium_ini)

        # set initial transform matrix
        self.mat = utils.scale_mat_2_str(np.eye(4))

        self.init_d = None
        if(config.mode is "abs"):
            self.init_d = utils.get_d_in(config.res)

        self.cnt_d = 0

    def set_medium(self, medium):
        self.eta = medium["eta"]
        self.g = medium["g"]
        self.albedo = medium["albedo"]
        self.sigmat = medium["sigma_t"]

    def set_serialized_path(self, serialized_path):
        self.serialized = serialized_path

    def set_out_path(self, model_id ,sample_num=-1):
        if(sample_num==-1):
            self.out_path = f"{self.out_dir}\\sample{model_id:02}.csv"
        else:
            self.out_path = f"{self.out_dir}\\sample{model_id:02}_{sample_num:05}.csv"

    def set_transform_matrix(self, mat):
        self.mat = utils.scale_mat_2_str(mat)

    def random_set_transform_matrix(self, config, plane=False):
        scale_factor = np.ones(3) * 0.25 + np.random.rand(3) * 2.75
        scale_mat = np.diag(scale_factor)
        self.set_transform_matrix(scale_mat)

        if(config.DEBUG):
            print(scale_mat)

        if(plane):
            scale_factor[2] = 0

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
                              out_path=self.out_path, coeff_range=config.coeff_range, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              g=self.g, eta=self.eta,
                              serialized=self.serialized, mat=self.mat)

        elif (config.mode is "visual"):
            scene = load_file(self.xml_path, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              g=self.g, eta=self.eta,
                              serialized=self.serialized, mat=self.mat)

        elif (config.mode is "test"):
            init_d = config.init_d
            scene = load_file(self.xml_path,
                              out_path=self.out_path, init_d = init_d, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              serialized=self.serialized,g=self.g, eta=self.eta,mat=self.mat)

        elif (config.mode is "abs"):
            init_d = f"{self.init_d[self.cnt_d, 0]:.5f} {self.init_d[self.cnt_d, 1]:.5f} {self.init_d[self.cnt_d, 2]:.5f}"
            scene = load_file(self.xml_path,
                              out_path=self.out_path, init_d = init_d, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              g=self.g, eta=self.eta)
            self.cnt_d += 1

        elif (config.mode is "sample_per_d"):
            init_d = "0, 0, 1"
            # This may be deleted.
            scene = load_file(self.xml_path,
                              out_path=self.out_path, coeff_range=config.coeff_range, spp=self.spp, seed=self.seed,
                              scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                              g=self.g, eta=self.eta,
                              serialized=self.serialized, mat=self.mat, random_sample="true", constant_sample="false", init_p=init_d, init_d=init_d)


        return scene

    def get_same_direction_scene(self, direction, position):

        init_d = "{},{},{}".format(direction[0],direction[1],direction[2])
        init_p = "{},{},{}".format(position[0],position[1],position[2])
        print(init_d)

        scene = load_file(self.xml_path,
                            out_path=self.out_path, init_d=init_d, init_p=init_p, spp=self.spp, seed=self.seed,
                            scale_m=self.scale_m, sigma_t=self.sigmat, albedo=self.albedo,
                            serialized=self.serialized,g=self.g, eta=self.eta,mat=self.mat,random_sample="false", constant_sample="true")

    
        return scene

def render(itr, config):
    if(config.mode is not "sample_per_d"):
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
                    print(scene_gen.albedo)
                    
                    if (not config.scale_fix):
                        # Sample scaling matrix and set
                        scale_rec_v = scene_gen.random_set_transform_matrix(config, plane=config.plane[model_id])
                        scale_rec[config.scene_batch_size * i:config.scene_batch_size * i + config.scene_batch_size] *= scale_rec_v


                    
                    scene = scene_gen.get_scene(config)

                    # Render the scene with scene_batch_size iteration
                    for j in range(config.scene_batch_size):
                        if(config.mode is "visual"):
                            sensor = scene.sensors()[0]
                        else:
                            # Set sampler's seed and generate new sensor object
                            seed = np.random.randint(100000000)
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
    else:
        render_spd(itr, config, 1 ,True)
    

def render_spd(itr, config, roop_num=0,test_mode=False):
    spp = config.spp

    # Generate scene and parameter generator
    param_gen = utils.ParamGenerator()
    scene_gen = SceneGenerator(config)
    cnt = 0
    time_seed = int(time.time())
    if(test_mode):
        np.random.seed(seed=time_seed)
        albedo_rand = np.random.rand()
    # Ready for recording scale factor
    scale_rec = np.ones([itr, 3])

    # Get serialized file name list from serialized file directory
    serialized_list = glob.glob(f"{config.SERIALIZED_DIR}\\*.serialized")
    if(config.DEBUG):
        print(serialized_list)

    if(itr % config.scene_batch_size != 0):
        sys.exit("Please set ite_per_shape to be a multiple of scene_batch_size")

    
    if((time_seed % 2 == 0) and test_mode==False):
        serialized_list.reverse()
        reverse = True
    else:
        reverse = False

    # Render with given params generator and scene generator
    for model_id, serialized in enumerate(serialized_list):

        if((test_mode)and(model_id!=3)):
            continue
        if (reverse):
            model_id = 5 - model_id
        print(model_id,serialized)

        # Set serialized model path and output csv file path to scene generator
        scene_gen.set_serialized_path(serialized)
        scene_gen.set_out_path(model_id)
    
        scale_rec = np.ones([itr, 3])
        time_seed = int(time.time())

        np.random.seed(seed=time_seed)
        rand_x = 30 * np.random.rand() + -15
        rand_y = 30 * np.random.rand() + -15
        rand_int = np.random.randint(0,1800,4)
        
        for i in range(itr // config.scene_batch_size):
            
            if (not config.scale_fix):
                # Sample scaling matrix and set
                scale_rec_v = scene_gen.random_set_transform_matrix(config, plane=config.plane[model_id])
                scale_rec[config.scene_batch_size * i:config.scene_batch_size * i + config.scene_batch_size] *= scale_rec_v
            
            # Generate scene for sampling same direction and position
            csv_input = pd.read_csv(filepath_or_buffer=config.SAMPLE_CSV_DIR, encoding="ms932", sep=",")

            scene_gen.set_out_path(model_id,roop_num)
            if(test_mode):
                scene_gen.set_out_path(model_id,albedo_rand)

            d_in_x = csv_input["d_in_x"]
            d_in_y = csv_input["d_in_y"]
            d_in_z = csv_input["d_in_z"]
            p_in_x = csv_input["p_in_x"]
            p_in_y = csv_input["p_in_y"]
            p_in_z = csv_input["p_in_z"]
            same_medium = {}
            same_medium["eta"] = csv_input["eta"][rand_int[0]]
            same_medium["g"] = csv_input["g"][rand_int[1]]
            same_medium["albedo"] = csv_input["albedo"][rand_int[2]]


                
            same_medium["sigma_t"] = 1
            direction = [d_in_x[rand_int[3]],d_in_y[rand_int[3]],d_in_z[rand_int[3]]]
            position = [rand_x, rand_y, 0]
            print(direction,position)
            if(test_mode):
                same_medium["albedo"] = round(albedo_rand,4)
                direction = [0,0,1]
                position = [0,0,0]
            scene = scene_gen.get_same_direction_scene(direction,position)
            scene_gen.set_medium(same_medium)
            update_medium(scene, same_medium)
            

            # Render the scene with scene_batch_size iteration
            for j in range(config.scene_batch_size):
                if(config.mode is "visual"):
                    sensor = scene.sensors()[0]
                else:
                    # Set sampler's seed and generate new sensor object
                    seed = np.random.randint(100000000)
                    sensor = get_sensor(spp, seed)

                # If medium parameters are not fixed, sample medium parameters again and update
                if (not (config.medium_fix) and  not (i >= 1 and config.mode is "sample_per_d")):
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
            
        
#        if config.mode is "abs" or  config.mode is "test":
#           break






            



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