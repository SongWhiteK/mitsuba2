import os
import sys
import glob
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.transforms import ToTensor
from PIL import Image
from sklearn.model_selection import train_test_split

class CNNDatasets_train(Dataset):
    def __init__(self, transform=None):
        self.data = pd.read_csv("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\csv_for_NN\\train.csv")
        self.im_dir = "D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\height_save"

        self.transform = transform

    def __getitem__(self, index):

        # Get csv data
        data = self.data.iloc[index]

        idx_props = ["g", "eta", "albedo",
                     "d_in_x", "d_in_y", "d_in_z"]
        props = pd.Series(data=data, index=idx_props).values
        props = props.astype(np.float32)
        props = torch.tensor(props)
        idx_r = ["r", "r_std"]
        r_and_rstd = pd.Series(data=data, index=idx_r).values
        r_and_rstd = r_and_rstd.astype(np.float32)
        r_and_rstd = torch.tensor(r_and_rstd)

        sample_id = int(data["id"])
        model_id = int(data["model_id"])

        # Get processed height map from index (~= id)
        im_path = f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\height_save\\train_image{sample_id:08}.png"

        sample = {}
        sample["props"] = props
        sample["r_and_rstd"] = r_and_rstd

        return im_path, sample

    def __len__(self):
        return len(self.data)

class CNNDatasets_test(Dataset):
    def __init__(self, transform=None):
        self.data = pd.read_csv("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\csv_for_NN\\test.csv")
        self.im_dir = "D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\height_save"

        self.transform = transform

    def __getitem__(self, index):

        # Get csv data
        data = self.data.iloc[index]

        idx_props = ["g", "eta", "albedo",
                     "d_in_x", "d_in_y", "d_in_z"]
        props = pd.Series(data=data, index=idx_props).values
        props = props.astype(np.float32)
        props = torch.tensor(props)
        idx_r = ["r", "r_std"]
        r_and_rstd = pd.Series(data=data, index=idx_r).values
        r_and_rstd = r_and_rstd.astype(np.float32)
        r_and_rstd = torch.tensor(r_and_rstd)

        sample_id = int(data["id"])
        model_id = int(data["model_id"])

        # Get processed height map from index (~= id)
        im_path = f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\height_save\\train_image{sample_id:08}.png"

        sample = {}
        sample["props"] = props
        sample["r_and_rstd"] = r_and_rstd

        return im_path, sample

    def __len__(self):
        return len(self.data)



def image_generate(im_path, im_size):
    """
        Load training image with path from dataset

        Args:
            im_path: Image path list from dataset
            im_size: image size of training image

        Return:
            im_tensor: Image tensor
    """
    batch_size = len(im_path)

    im = np.zeros([batch_size, 1, im_size, im_size])

    for i, path in enumerate(im_path):
        im[i, 0, :, :] = Image.open(path)

    im = im.astype(np.float32)
    im_tensor = torch.from_numpy(im).clone()

    return im_tensor

def train_data_split(train_csv_path):
    """
        Split dataset
        train:val:test=7:1:2

        Args:
            im_path: Image path list from dataset
            im_size: image size of training image

        Return:
            im_tensor: Image tensor
    """
    csv_input = pd.read_csv(train_csv_path)

    # random_state means seed of sampler
    train_data, test_data = train_test_split(csv_input, test_size=0.3, random_state=0)
    test_data, val_data = train_test_split(test_data, test_size=1/3, random_state=0)
    train_data = train_data.sort_values("id")
    test_data = test_data.sort_values("id")
    val_data = val_data.sort_values("id")
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    val_data = val_data.reset_index()
    print(val_data)

    train_data.to_csv("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\csv_for_NN\\train.csv",index=False)
    test_data.to_csv("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\csv_for_NN\\test.csv",index=False)
    val_data.to_csv("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\csv_for_NN\\val.csv",index=False)


if __name__ == "__main__":
    train_data_split("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\train_sample.csv")