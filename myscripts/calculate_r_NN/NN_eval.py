import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from zmq import device
from dataset import CNNDatasets_test, CNNDatasets_train, image_generate
from net import Net
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import time

def eval(net, test_dataset_loader, device):
    criterion = nn.MSELoss()
    running_loss = 0

    for i, (im_path, sample) in enumerate(test_dataset_loader):
        
        props = sample["props"]
        r_and_rstd = sample["r_and_rstd"]
        im = image_generate(im_path, 128)

        props, r_and_rstd, im = props.to(device), r_and_rstd.to(device), im.to(device)

        outputs = net(im, props)

    return outputs, r_and_rstd




if __name__ == "__main__":
    device = torch.device("cuda")
    net = Net().to(device)

    model_path = "D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\model_0.002_Adam.pth"
    net.load_state_dict(torch.load(model_path))

    test_dataset = CNNDatasets_test()
    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False)
    outputs, r_and_std =  eval(net, test_loader, device)