import sys
import time
import mitsuba
import integrate
import render_config as config
import data_pipeline
import utils_render
from multiprocessing import freeze_support

sys.path.append("myscripts/render/dict")
from scene_dict import scene_dict
from mesh_dict import meshes_cube, meshes_leather

mitsuba.set_variant(config.variant)

from mitsuba.core import Thread
from mitsuba.core.xml import load_file, load_dict

if __name__ == "__main__":
    if config.multi_process:
        freeze_support()
        
    ##### Setting scene #####
    bdata = data_pipeline.BSSRDF_Data()
    mesh = meshes_leather(1, 6)

    mesh.register_params(bdata, ior=1.5, scale=config.scale, sigma_t = 1.0, albedo = 0.9, g = 0.5)

    mesh.register_all_mesh(bdata)

    scene_dict = bdata.add_object(scene_dict)
    scene = load_dict(scene_dict)

    # Rendering settings
    spp = config.spp
    sample_per_pass = config.sample_per_pass

    print("Rendering start")

    start = time.time()
    integrate.render(scene, spp, sample_per_pass, bdata)
    process_time = time.time() - start

    print(f"Rendering end (took {process_time}s)")

    utils_render.write_log(config, process_time)
    