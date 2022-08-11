import sys
import os
sys.path.append(os.path.abspath("..\.."))
from data_handler import DataHandler, delete_file
from traindata_config import TrainDataConfiguration
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
import optuna

class Net_test(nn.Module):
    def __init__(self,hidden_dim,p_drop=0.25):
        super(Net_test, self).__init__() 
        self.p_drop = p_drop
        self.hidden_dim = hidden_dim

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

            nn.Linear(144, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(self.hidden_dim , 2),
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

def objective(trial):
    # TODO:epoch automation

    max_epoch = 500
    device = "cuda"    
    config = TrainDataConfiguration()
    d_handler = DataHandler(config)
    

    learning_rate = 0.0003
    optimizer = trial.suggest_categorical("optimizer", ["Adam"])
    hidden_dim = 64
    image_scale = trial.suggest_discrete_uniform("image_scale", 0.1, 0.6, q=0.01)

    delete_file("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\height_save")
    time.sleep(1)
    os.mkdir("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\height_save")
    d_handler.generate_spd_train_data(image_scale=image_scale)

    net = Net_test(hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer)(net.parameters(), lr=learning_rate)

    train_dataset = CNNDatasets_train()
    test_dataset = CNNDatasets_test()
    train_loader = DataLoader(train_dataset, batch_size=16,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16,shuffle=False)
    loss_rec = []

    # train 
    net.train()
    for epoch in range(max_epoch):
        for i, (im_path, sample) in enumerate(train_loader):
            props = sample["props"]
            r_and_rstd = sample["r_and_rstd"]
            im = image_generate(im_path, 128)

            props, r_and_rstd, im = props.to(device), r_and_rstd.to(device), im.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(im, props)
            loss = criterion(outputs, r_and_rstd)

            loss.backward()
            optimizer.step()

    # validation
        net.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for i, (im_path, sample) in enumerate(test_loader):
                props = sample["props"]
                r_and_rstd = sample["r_and_rstd"]
                im = image_generate(im_path, 128)

                props, r_and_rstd, im = props.to(device), r_and_rstd.to(device), im.to(device)

                # forward + backward + optimize
                outputs = net(im, props)
                validation_loss += criterion(outputs, r_and_rstd)
            loss_rec.append(validation_loss)
    print(loss_rec)
    return min(loss_rec)

def study_params(n_trials):
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)
    print(f"=====best param=====\n{study.best_params}")

if __name__ == "__main__":
    study_params(n_trials=200)

    #n=100 {'learning_rate': 0.0009500000000000001, 'optimizer': 'Adam', 'hidden_dim': 112, 'image_scale': 0.63}
    #image_scale = 0.1