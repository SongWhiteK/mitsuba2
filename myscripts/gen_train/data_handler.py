"""
Processing trainingdata
"""
import os
from os import makedirs
import re
import shutil
import glob
import datetime
from statistics import mode
import time
import pickle
import numpy as np
import pandas as pd
import cv2
import utils
import matplotlib.pyplot as plt
from traindata_config import TrainDataConfiguration
import tqdm

from mitsuba.heightmap import HeightMap

class DataHandler:
    """
    csv data handler for post proccessing
    """

    def __init__(self, config):
        self.sample_path = config.SAMPLE_DIR
        self.list_update()
        self.train_image_dir_path = f"{config.IMAGE_DIR}"
        self.map_dir_path = f"{config.MAP_DIR}"
        self.train_sample_dir_path = f"{config.TRAIN_DIR}"
        self.debug = config.DEBUG
        self.tag = config.tag
        self.im_size = config.im_size
        self.plane = config.plane
        self.num_per_subdir = config.num_per_subdir

    def list_update(self):
        self.sample_list = glob.glob(f"{self.sample_path}\\*")
        

    def generate_train_data(self, offset=0):
        """
        Generating height map w.r.t incident point and medium parameters
        Args:
            offset: ID offset
        """
        height_map = None
        id_data = 0
        height_map_list = glob.glob(f"{self.map_dir_path}\\height_map*.png")
        df_all = pd.DataFrame()

        # File operation
        key_input = None
        file_list = glob.glob(self.train_image_dir_path)
        if(os.path.exists(self.train_image_dir_path)):
            print(f"Delete sample files in {self.train_image_dir_path}? [y/n]")
            while(True):
                #key_input = input()
                key_input = "n"
                if(key_input == "y"):
                    shutil.rmtree(self.train_image_dir_path)
                    time.sleep(3)
                    os.mkdir(self.train_image_dir_path)
                    break
                elif(key_input == "n"):
                    break
                print("Please input valid letter")
        else:
            os.makedirs(self.train_image_dir_path)

        # Instantiate HeitMap class
        heightmap_pybind = HeightMap(self.im_size, interpolation=HeightMap.Interpolation.NEAREST)
        
        # Process with given csv files
        for i, file_name in enumerate(self.sample_list):
            # get id number and map number
            data = pd.read_csv(file_name)
            data["id"] = data.index + id_data + offset
            df_model_id = pd.DataFrame(np.ones([len(data), 1]) * i, columns=["model_id"])
            data["model_id"] = df_model_id["model_id"]
            print(i , self.plane)
            if(self.plane[i]):
                df_scale_z = pd.DataFrame(np.zeros([len(data), 1]), columns=["scale_z"])
                data["scale_z"] = df_scale_z["scale_z"]

            # join train data into one file
            df_all = df_all.append(data)

            # Get entire height map for a training data
            height_map = cv2.imread(height_map_list[i], cv2.IMREAD_GRAYSCALE)

            # Make training images directory for each height map
            if(key_input == "y"):
                os.makedirs(f"{self.train_image_dir_path}\\map_{i:03}")

            file_path = None
        
            # Process with each training data
            for row in data.itertuples():
                
                if(row.id % self.num_per_subdir == 0):
                    file_path = f"{self.train_image_dir_path}\\map_{i:03}\\images{row.id}_{row.id + self.num_per_subdir-1}"
                    os.makedirs(file_path)

                image = gen_train_image(row, height_map, self.im_size, self.debug, pybind=heightmap_pybind)

                # Save height map image with id
                cv2.imwrite(f"{file_path}\\train_image{row.id:08}.png", image)
                id_data += 1

                if(row.id % 10000 == 0):
                    print(f"{datetime.datetime.now()} -- Log: Processed {row.id}")

        # refine and output sampled path data
        time.sleep(2)
        print(df_all)
        self.refine_data(df_all)

    def generate_spd_train_data(self):
        """
        Generating height map w.r.t incident point and medium parameters for spd mode.

        """
        height_map = None
        id_data = 0
        height_map_list = glob.glob(f"{self.map_dir_path}\\height_map*.png")
        df_all = pd.DataFrame()

        # Instantiate HeitMap class
        heightmap_pybind = HeightMap(self.im_size, interpolation=HeightMap.Interpolation.NEAREST)
        
        # get id number and map number
        data = pd.read_csv("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\train_sample.csv")

        # join train data into one file
        df_all = df_all.append(data)

        file_path = f"{self.train_image_dir_path}"

        # Process with each training data
        for row in data.itertuples():

            height_map = cv2.imread(height_map_list[row.model_id], cv2.IMREAD_GRAYSCALE)

            # sigman = 15
            image = gen_train_image(row, height_map, self.im_size, self.debug, pybind=heightmap_pybind,spd_mode=True)
            
            # Save height map image with id
            cv2.imwrite(f"{file_path}\\train_image{row.Index:08}.png", image)
            id_data += 1

            if(row.Index % 100 == 0):
                print(f"{datetime.datetime.now()} -- Log: Processed {row.Index}")

        # refine and output sampled path data
        time.sleep(2)
        print(df_all)



    def refine_data(self, df):
        """
        Refine and Save training data.
        Refining includes select, add and scale training data

        Args:
            df: row training data
        """
        # Get new params
        df["eff_albedo"] = utils.reduced_albedo_to_effective_albedo(
            utils.get_reduced_albedo(df["albedo"], df["g"], df["sigma_t"])
            )
        df["sigma_n"] = utils.get_sigman(df)
        df["height_max"] = df["scale_z"] / df["sigma_n"]
        df["p_in_x"] /= df["sigma_n"]
        df["p_in_y"] /= df["sigma_n"]
        df["p_in_z"] /= df["sigma_n"]
        df["p_out_x"] /= df["sigma_n"]
        df["p_out_y"] /= df["sigma_n"]
        df["p_out_z"] /= df["sigma_n"]

        df = df[self.tag]
        if(not os.path.exists(f"{self.train_sample_dir_path}")):
            os.makedirs(f"{self.train_sample_dir_path}")
        df.to_csv(f"{self.train_sample_dir_path}\\train_path.csv",
                  index=False, float_format="%.6g")


        
    def delete_sample_files(self):
        file_list = glob.glob(f"{self.sample_path}\\*")
        if(os.path.exists(self.sample_path)):
            print(f"Delete sample files in {self.sample_path}? [y/n]")
            while(True):
                key_input = input()
                if(key_input == "y"):
                    shutil.rmtree(self.sample_path)
                    time.sleep(0.5)
                    os.makedirs(self.sample_path)
                    break
                elif(key_input == "n"):
                    break
                print("Please input valid letter")
        else:
            os.makedirs(self.sample_path)

            

def delete_file(filepath):
    if(os.path.exists(filepath)):
        os.remove(filepath)

def join_scale_factor(path, scale):
    """
    Add scale factor of shape objects to the sampled data from given path

    Args:
        path: Path of a file which contains sampled data
        scale: Scale factor of shape objects
    """
    data = pd.read_csv(path)
    data["scale_x"] = scale["scale_x"]
    data["scale_y"] = scale["scale_y"]
    data["scale_z"] = scale["scale_z"]

    data.to_csv(path, index=False)

def join_model_id(path, model_id):
    """
    Add scale model id to the sampled data from given path

    Args:
    path: Path of a file which contains sampled data
    id: Model id number of shape objects
    """
    data = pd.read_csv(path)
    data["model_id"] = model_id["model_id"]

    data.to_csv(path, index=False)


def gen_train_image(data, height_map, im_size, debug, pybind=None,spd_mode=False):
    """
    Generate training image data from path sample data and entire height map.
    Generated image is centered by incident location.
    Sigma_n of the target medium defines a range of the resulting image.
    Assuming the unscaled original size of 3D model is 50 * 50 * 21 [mm]

    Args:
        data: pandas DataFrame for one path sampling data
        height_map: Entire height map for sampled 3D model
    Return:
        train_image: Generated height map centered by incident location. The size is 255*255 [pxl]
    """

    # Get medium parameters, incident loaction, and scale factors
    medium = {}
    medium["sigma_t"] = data.sigma_t
    medium["albedo"] = data.albedo
    medium["g"] = data.g

    x_in = data.p_in_x
    y_in = data.p_in_y
    z_in = data.p_in_z
    scale_x = data.scale_x
    scale_y = data.scale_y
    scale_z = data.scale_z

    # xy coordinates of u-v origin, 50 is size of 3D model assumed   
    x_min = -25 * scale_x
    y_max = 25 * scale_y

    x_range = -x_min * 2
    y_range = y_max * 2


    # Scale height map with scale factors as sampling
    map_scaled = cv2.resize(height_map, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)

    if(not spd_mode):
        sigma_n = utils.get_sigman(medium)

    else:
        medium_for_uniform_range = {}
        medium_for_uniform_range["sigma_t"] = 1
        medium_for_uniform_range["albedo"] = 0.9
        medium_for_uniform_range["g"] = 1
        sigma_n = utils.get_sigman(medium_for_uniform_range)
    if(pybind != None):
        return pybind.clip_scaled_map(map_scaled, x_in, y_in, sigma_n, x_range, y_range, x_min, y_max)

    return clip_scaled_map(map_scaled, [x_in, y_in], sigma_n, x_range, y_range, x_min, y_max, im_size)

def clip_scaled_map(map_scaled, pos_in, sigma_n, x_range, y_range, x_min, y_max, im_size):
    """
        Clip height map for training data
        This method should not be used and you should use mitsuba.heightmap.Heightmap.clip_scaled_map()
    """
    
    height_scaled, width_scaled = map_scaled.shape

    x_in = pos_in[0]
    y_in = pos_in[1]

    # Length of a pixel edge
    px_len = x_range / width_scaled
    # The number of pixels in 6 sigma_n

    r_px_range = np.ceil(6 * sigma_n / px_len).astype(np.uint32)

    u_c = int((y_max - y_in) * height_scaled / y_range)
    v_c = int((x_in - x_min) * width_scaled / x_range)

    # Clip map_full_scaled in specified range from incident position
    distance_u_n = np.min([r_px_range, u_c])              # distance of u for negative direction
    distance_u_p = np.max([np.min([r_px_range, height_scaled - u_c - 1]), 0]) # distance of u for positive direction
    distance_v_n = np.min([r_px_range, v_c])              # distance of v for negative direction
    distance_v_p = np.max([np.min([r_px_range, width_scaled - v_c - 1]), 0])  # distance of v for positive direction

    u_range = [u_c - distance_u_n, u_c + distance_u_p + 1]
    v_range = [v_c - distance_v_n, v_c + distance_v_p + 1]
    map_clip = map_scaled[u_range[0]: u_range[1], v_range[0]:v_range[1]]

    height_clip, width_clip = map_clip.shape

    scale_map = im_size / (12 * sigma_n / px_len)
    map_clip = cv2.resize(map_clip, None, fx=scale_map, fy=scale_map, interpolation=cv2.INTER_AREA)
    height_clip, width_clip = map_clip.shape

    u_c = int(height_clip * distance_u_n / (distance_u_n + distance_u_p))
    v_c = int(width_clip * distance_v_n / (distance_v_n + distance_v_p))

    r_px = im_size // 2

    distance_u_n = np.min([r_px, u_c])              
    distance_u_p = np.max([np.min([r_px, height_clip - u_c - 1]), 0])
    distance_v_n = np.min([r_px, v_c])
    distance_v_p = np.max([np.min([r_px, width_clip - v_c - 1]), 0])
    

    # Paste map_clip to a canvas ranging 255*255 [px]
    canvas = np.ones([2 * r_px + 1, 2 * r_px + 1], dtype="uint8") * 31
    canvas_c = r_px + 1
    canvas_size = 2 * r_px + 1

    canvas[r_px-distance_u_n:r_px+distance_u_p, r_px-distance_v_n:r_px+distance_v_p] = map_clip[u_c-distance_u_n:u_c+distance_u_p, v_c-distance_v_n:v_c+distance_v_p]

    # If pixels are out of range 6 sigma_n, set pixel value to 0 by masking
    uv_range = np.arange(1, canvas_size + 1)
    V, U = np.meshgrid(uv_range, uv_range)
    mask = (V - canvas_c)**2 + (U - canvas_c)**2 <= r_px**2
    canvas[np.logical_not(mask)] = 0

    return canvas

def spd_datahandler(sample_filepath):
    """
    Data handler of sampling in spd mode.
    The spd mode simulates multiple light incidence from the same direction at the same location.
    """

    print("Train data generating")
    sample_file_list = os.listdir(sample_filepath)
    colmuns = ["sigma_t", "eta", "g", "albedo", "r", "r_std", "p_in_x", "p_in_y", "p_in_z", "p_out_x", "p_out_y", "p_out_z", "d_in_x", "d_in_y", "d_in_z", "scale_x", "scale_y", "scale_z", "model_id", "id"]
    output = pd.DataFrame(columns=colmuns)
    cnt = [0,0,0,0] #count [correct,edge,fixed edge,invalid]
    model_cnt = [0,0,0,0,0,0]
    edge_list=[]
    invalid_list=[]
    for file in tqdm.tqdm(sample_file_list):
        path_now = sample_filepath + "\\" + file
        sample_file = os.listdir(path_now)


        for sample_csv in sample_file:
            model_id = check_model_id(sample_csv)
            model_cnt[model_id] += 1
            csv_path = path_now + "\\" + sample_csv
            csv_input = pd.read_csv(filepath_or_buffer=csv_path, encoding="ms932", sep=",")

            if (check_edge_data(csv_input) and check_number_of_data(csv_input)):
                r, r_std = get_r_and_r_std(csv_input)
                sigma_t = csv_input["sigma_t"][1]
                eta = csv_input["eta"][1]
                g = csv_input["g"][1]
                albedo = csv_input["albedo"][1]
                p_in_x = csv_input["p_in_x"][1]
                p_in_y = csv_input["p_in_y"][1]
                p_in_z = csv_input["p_in_z"][1]
                p_out_x = csv_input["p_in_x"][1]
                p_out_y = csv_input["p_in_y"][1]
                p_out_z = csv_input["p_in_z"][1]
                d_in_x = csv_input["d_in_x"][1]
                d_in_y = csv_input["d_in_y"][1]
                d_in_z = csv_input["d_in_z"][1]
                scale_x = csv_input["scale_x"][1]
                scale_y = csv_input["scale_y"][1]
                scale_z = csv_input["scale_z"][1]
                id = cnt[0] + cnt[2]
                r, r_std = round(r,6), round(r_std,6)
                list = [[sigma_t, eta, g, albedo, r, r_std, p_in_x, p_in_y, p_in_z, p_out_x, p_out_y, p_out_z, d_in_x, d_in_y, d_in_z, scale_x, scale_y, scale_z, model_id, id]]
                list = pd.DataFrame(data=list,columns=colmuns)
                output = pd.concat([output,list], ignore_index= True)
                cnt[0] += 1
            elif(not check_edge_data(csv_input)):
                cnt[1] += 1
                edge_list.append(csv_path)
                csv_input, edge_mask = edge_processing(csv_input)
                if (edge_mask):
                    r, r_std = get_r_and_r_std(csv_input)
                    sigma_t = csv_input["sigma_t"][1]
                    eta = csv_input["eta"][1]
                    g = csv_input["g"][1]
                    albedo = csv_input["albedo"][1]
                    p_in_x = csv_input["p_in_x"][1]
                    p_in_y = csv_input["p_in_y"][1]
                    p_in_z = csv_input["p_in_z"][1]
                    p_out_x = csv_input["p_in_x"][1]
                    p_out_y = csv_input["p_in_y"][1]
                    p_out_z = csv_input["p_in_z"][1]
                    d_in_x = csv_input["d_in_x"][1]
                    d_in_y = csv_input["d_in_y"][1]
                    d_in_z = csv_input["d_in_z"][1]
                    scale_x = csv_input["scale_x"][1]
                    scale_y = csv_input["scale_y"][1]
                    scale_z = csv_input["scale_z"][1]
                    id = cnt[0] + cnt[2]
                    r, r_std = round(r,6), round(r_std,6)
                    list = [[sigma_t, eta, g, albedo, r, r_std, p_in_x, p_in_y, p_in_z, p_out_x, p_out_y, p_out_z, d_in_x, d_in_y, d_in_z, scale_x, scale_y, scale_z, model_id, id]]
                    list = pd.DataFrame(data=list,columns=colmuns)
                    output = pd.concat([output,list], ignore_index= True)
                    cnt[2] += 1

            elif(not check_number_of_data(csv_input)):
                cnt[3] += 1
                invalid_list.append(csv_path)

    print(f"Training data:{cnt[0]+cnt[2]}\nCorrect data:{cnt[0]} Edge data:{cnt[1]} Fixed edge data:{cnt[2]} Invalid number data:{cnt[3]}")
    print("Number of each model",end=" ")
    for modelnum in range(6):
        print(f"{modelnum}:{model_cnt[modelnum]}",end=" ")

    output.to_string(index=False)
    delete_file("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data" + "\\" + "train_sample.csv")
    output.to_csv("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data" + "\\" + "train_sample.csv", index = False)
    print("\n-------Process finished-------")
    while(True):
        key = input("Show csv path e:edge i:invalid number z:end \n>>")
        if key == "z":
            break
        elif key == "e":
            for i in range(len(edge_list)):
                print(edge_list[i])
        elif key == "i":
            for i in range(len(invalid_list)):
                print(invalid_list[i])



def get_r_and_r_std(df):
    # calculation R and R_std

    p_in_x = df["p_in_x"]
    p_in_y = df["p_in_y"]
    p_out_x = df["p_out_x"]
    p_out_y = df["p_out_y"]
    r = ((p_out_x - p_in_x)**2 + (p_out_y - p_in_y)**2)**0.5
    r_mean = r.mean()
    r_std = r.std()

    return r_mean,r_std


def check_model_id(sample_csv_path):
    """
    Identify model id from file name

    Args:
        sample_csv : csv_path(e.g. ~\\sample00.csv)
    return:
        model_id : model id of serialized model
    """

    for i in [1,2,3,4,5]:
        if (str(i) in sample_csv_path):
            return i
    return 0


def check_edge_data(df):
    """
    If df has not edge data, return True.
    Integrate with edge_processing if I have time.
    """

    p_out_x = df["p_out_x"]
    p_out_y = df["p_out_y"]
    if (25 in p_out_x.values or -25 in p_out_x.values):
        return False
    elif (25 in p_out_y.values or -25 in p_out_y.values):
        return False
    return True

def edge_processing(df):
    """
    Function to process data frames with edge data.
    If there are more than criterion(75) other than edge data, return the removed edge data df.

    Args:
        df : sample csv file(pd.DataFrame)

    Return:
        df : the removed edge data df
        bool : If there are more than criterion(75) other than edge data, return True.Else return False.
    """

    df = df[df.p_out_x != 25]
    df = df[df.p_out_y != 25]
    df_len = len(df)
    df = df.reset_index(drop=True)
    if(df_len >= 90):
        return df ,True
    else:
        return 0 ,False

    
def check_number_of_data(df):
    """
    Check if the number of data in df is correct.
    This is largely meant to confirm that the sampling is working.(e.g. If other spd's are merged with each other and become a lot of data)
    """
    df_len = len(df)
    if(df_len == 100):
        return True
    else:
        return False
    

if __name__ == "__main__":
    config = TrainDataConfiguration()
    d_handler = DataHandler(config)
    if(config.mode != "sample_per_d"):
        d_handler.generate_train_data()
    else:
        while(True):
            key = input("1:csv file generate 2:image generate 3:both\n>>")
            if(key == "1"):    
                spd_datahandler(config.SAMPLE_MAIN_DIR)
                break
            elif(key == "2"):
                d_handler.generate_spd_train_data()
                break
            elif(key == "3"):
                spd_datahandler(config.SAMPLE_MAIN_DIR)
                time.sleep(3)
                d_handler.generate_spd_train_data()
                break
            else:
                print("Please input valid letter")