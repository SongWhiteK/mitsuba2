"""
Data pipeline for rendering
Handle some data, such as, mesh transform matrix for BSSRDF
and create scene format
"""
import sys
sys.path.append("./myscripts/vae")
sys.path.append("./myscripts/gen_train")

import mitsuba
import render_config as config
import numpy as np
from multiprocessing import Pool
from data_handler import clip_scaled_map



mitsuba.set_variant(config.variant)

from mitsuba.core import ScalarTransform4f, ScalarVector3f


class BSSRDF_Data:
    """  Class for handling BSSRDF data """

    def __init__(self):
        self.bssrdf = {}
        self.mesh = {}
        self.mesh_map = {}
        self.mesh_range = {}
        self.mesh_minmax = {}

    def register_medium(self, mesh_id, ior=1.5, scale=1.0,
                        sigma_t=1.0, albedo=0.5, g=0.25):
        """
        Register medium with mesh id

        Args:
            id: Mesh id, which must start from 1
            ior, scale, sigma_t, albedo, g: Medium parameters
        """

        if mesh_id in self.bssrdf:
            print("This ID has already used. Do you want to overwrite? [y/n]")
            while(True):
                key = input()
                if key == "y":
                    print("ID " +"\"" + str(mesh_id) + "\"" " is overwritten")
                    break

                elif key == "n":
                    print("Register canceled")
                    return

                else:
                    print("Input valid key")
                    continue

        self.bssrdf[mesh_id] = {
            "type": "bssrdf",
            "int_ior": ior,
            "ext_ior": 1.0,
            "scale": scale,
            "sigma_t": sigma_t,
            "albedo": albedo,
            "g": g,
            "mesh_id": mesh_id
            }

        

    def register_mesh(self, mesh_id, mesh_type, height_max, mesh_map, range,
                      minmax, filename=None, translate=[0,0,0],
                      rotate={"axis": "x", "angle": 0.0}, scale=[1,1,1]):
        """
        Register mesh data
        """
        self.mesh[mesh_id] = {
            "type": mesh_type,
            "filename": filename,
            "translate": translate,
            "rotate": rotate,
            "scale": scale,
            "height_max": height_max
        }

        self.mesh_map[mesh_id] = mesh_map
        self.mesh_range[mesh_id] = range
        self.mesh_minmax[mesh_id] = minmax

    def add_object(self, scene_dict):
        """
        Add registered meshes to given scene file format
        """

        if len(self.bssrdf) != len(self.mesh):
            exit("The number of registerd mesh and bssrdf are different!")

        num_obj = len(self.bssrdf)

        for i in range(num_obj):
            i += 1
            bssrdf = self.bssrdf[i]
            mesh = self.mesh[i]

            axis = None
            if(mesh["rotate"]["axis"] == "x"):
                axis = [1, 0, 0]
            elif(mesh["rotate"]["axis"] == "y"):
                axis = [0, 1, 0]
            elif(mesh["rotate"]["axis"] == "z"):
                axis = [0, 0, 1]
            angle = mesh["rotate"]["angle"]


            if self.mesh[i]["type"] == "rectangle":
                scene_dict[str(i)] = {
                    "type": mesh["type"],
                    "to_world": ScalarTransform4f.translate(mesh["translate"])
                                * ScalarTransform4f.rotate(axis, angle)
                                * ScalarTransform4f.scale(mesh["scale"]),
                }
        
            else:
                scene_dict[str(i)] = {
                    "type": mesh["type"],
                    "filename": mesh["filename"],
                    "to_world": ScalarTransform4f.translate(mesh["translate"])
                                * ScalarTransform4f.rotate(axis, angle)
                                * ScalarTransform4f.scale(mesh["scale"]),
                }

            bssrdf["trans"] = mesh["translate"]
            if(mesh["rotate"]["axis"] == "x"):
                bssrdf["rotate_x"] = angle
            elif(mesh["rotate"]["axis"] == "y"):
                bssrdf["rotate_y"] = angle
            elif(mesh["rotate"]["axis"] == "z"):
                bssrdf["rotate_z"] = angle

            bssrdf["height_max"] = mesh["height_max"]

            bsdf = {
                "bsdf_" + str(i): bssrdf
            }

            scene_dict[str(i)].update(bsdf)
            

        return scene_dict

    def get_medium_dict(self, mesh_id):
        medium = {}
        medium["sigma_t"] = self.bssrdf[mesh_id]["sigma_t"]
        medium["albedo"] = self.bssrdf[mesh_id]["albedo"]
        medium["g"] = self.bssrdf[mesh_id]["g"]

        return medium


    def get_height_map(self, in_pos, mesh_id):
        
        num_objects = range(len(mesh_id))
        self.mesh_id = mesh_id.torch().cpu()
        self.in_pos = in_pos.torch().cpu()

        with Pool(processes=8) as p:
            result = p.map(func=self.call_map, iterable=num_objects)

        result = [x for x in result if x is not None]

        return result

    def call_map(self, i):
        ref_id = int(self.mesh_id[i])

        if(ref_id == 0):
            return

        mesh_map = self.mesh_map[ref_id]
        medium = self.get_medium_dict(ref_id)
        ref_in = self.in_pos[i, :]
        x_range, y_range = self.mesh_range[ref_id]
        x_min, y_max = self.mesh_minmax[ref_id]

        height_map = clip_scaled_map(mesh_map, ref_in, medium, x_range, y_range, x_min, y_max)

        return height_map
