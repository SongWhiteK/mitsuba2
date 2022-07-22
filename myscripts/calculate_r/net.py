"""
NN module for calculating R and R_std(range of image)
"""
from datetime import datetime
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from vae_config import VAEConfiguration
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            #### conv2d(in_ch, out_ch, conv_range)
            # Input: 127x127 image 
            # Output: 128x1 vector
            nn.Conv2d(3,64,3),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(64,128,3,padding=[4//2, 4// 2]),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=[4//2, 4// 2]),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,padding=[4//2, 4// 2]),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(128,256,3,padding=[4//2, 4// 2]),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=[4//2, 4// 2]),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,3,padding=[4//2, 4// 2]),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(256,512,3,padding=[4//2, 4// 2]),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=[4//2, 4// 2]),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,padding=[4//2, 4// 2]),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.25),
        )
        self.dence = nn.Sequential(

            nn.Linear(512*7*7, 512*7*7),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512*7*7, 512*7*7),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512*7*7 , 10),
        )
    def check_cnn_size(self, size_check):
        out = self.conv_layers(size_check)
            
        return out

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0),-1)
        x = self.dence(x)

        return(x)