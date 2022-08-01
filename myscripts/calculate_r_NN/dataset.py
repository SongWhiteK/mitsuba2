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

class CNNDatasets(Dataset):
    def __init__(self, transform=None):
        self.data = pd.read_csv("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\train_sample.csv")
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
        # データセットのサンプル数が要求されたときに返す処理を実装
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


if __name__ == "__main__":
    dataset = CNNDatasets()
    train_loader = DataLoader(dataset, batch_size=3)
    for batch_idx, (im_path, sample) in enumerate(train_loader):
        props = sample["props"]
        r_and_rstd = sample["r_and_rstd"]
        im = image_generate(im_path, 128)