import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from dataset import CNNDatasets_test, CNNDatasets_train, image_generate
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = 64
        self.conv_layers = nn.Sequential(
            #input : 128*128 image
            #output : 128*1 vector
            nn.Conv2d(1,32,3),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(32,64,3),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(64,128,3),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(128,128,3),
            nn.ReLU(),
        )
        self.feature = nn.Sequential(
            nn.Linear(6,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
        )

        self.dence = nn.Sequential(

            nn.Linear(144, self.hidden_layer),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(self.hidden_layer , 2),
        )

    def check_cnn_size(self, size_check):
        out = self.conv_layers(size_check)
            
        return out

    def feature_conversion(self, im, props):
        """Conversion from height map and properties to feature vector"""
        im_feature = self.conv_layers(im)
        im_feature = im_feature.view(-1,128)

        props_feature = self.feature(props)

        feature = torch.cat([props_feature, im_feature], dim=1)

        return feature


    def forward(self, im, props):
        feature = self.feature_conversion(im, props)
        r = self.dence(feature)

        return r