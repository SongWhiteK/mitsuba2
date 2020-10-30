"""
Data pipeline for rendering
Handle some data, such as, mesh transform matrix for BSSRDF
and create scene format
"""

import mitsuba
import render_config as config
import numpy as np

mitsuba.set_variant(config.variant)

from mitsuba.core import ScalarTransform4f, ScalarVector3f


class BSSRDF_Data:
    """  Class for handling BSSRDF data """

    def __init__(self):
        self.bssrdf = {}
        self.mesh = {}

    def register_medium(self, id, ior=1.5, scale=1.0,
                        sigma_t=1.0, albedo=0.5, g=0.25):
        """
        Register medium with mesh id

        Args:
            id: Mesh id, which must start from 1
            ior, scale, sigma_t, albedo, g: Medium parameters
        """

        if id in self.bssrdf:
            print("This ID has already used. Do you want to overwrite? [y/n]")
            while(True):
                key = input()
                if key == "y":
                    print("ID " +"\"" + str(id) + "\"" " is overwritten")
                    break

                elif key == "n":
                    print("Register canceled")
                    return

                else:
                    print("Input valid key")
                    continue

        self.bssrdf[id] = {
            "type": "bssrdf",
            "int_ior": ior,
            "ext_ior": 1.0,
            "scale": scale,
            "sigma_t": sigma_t,
            "albedo": albedo,
            "g": g,
            "mesh_id": id
            }

        

    def register_mesh(self, id, mesh_type, height_max, mesh_map, filename=None,
                      translate=ScalarVector3f([0,0,0]),
                      rotate={"axis": "x", "angle": 0.0},
                      scale=ScalarVector3f([1,1,1])):
        """
        Register mesh data
        """
        self.mesh[id] = {
            "type": mesh_type,
            "filename": filename,
            "translate": translate,
            "rotate": rotate,
            "scale": scale,
            "height_max": height_max,
            "mesh_map": mesh_map
        }

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

                

            


