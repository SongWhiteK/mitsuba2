import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torch.nn as nn
import torch
import torch.nn.functional as F
from dataset import CNNDatasets, image_generate
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

            nn.Linear(144, 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64 , 2),
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

def train(lr=0.0001, opt="Adam"):
    device = torch.device("cuda")
    dataset = CNNDatasets()
    net = Net().to(device)
    if(opt == "Adam"):
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)

    criterion = nn.MSELoss()

    train_loader = DataLoader(dataset, batch_size=16,shuffle=True)

    #n_samples = len(train_loader) 
    #train_size = int(len(train_loader) * 0.8) 
    #val_size = n_samples - train_size 

    #train_dataset, val_dataset = torch.utils.data.random_split(train_loader, [train_size, val_size])

    net.train()
    epochs = 600
    colmuns = ["loss"]
    loss_record_forward = pd.DataFrame(columns=colmuns)
    #loss_record_val = pd.DataFrame(columns=colmuns)

    for epoch in range(epochs):
        print(f"start epoch {epoch+1}")
        running_loss = 0.0
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

            # print statistics
            running_loss += loss.item()

            if (i+1) % 10 == 0:
                print('[{:d}, {:5d}] loss: {:.3f}'
                        .format(epoch + 1, i + 1, running_loss))
                
                if((i+1) % 110 == 0):
                    loss_temp = [[running_loss]]
                    loss_temp = pd.DataFrame(data=loss_temp,columns=colmuns)
                    loss_record_forward = pd.concat([loss_record_forward,loss_temp], ignore_index= True)

                    #loss_val = val(net, val_dataset)
                    #loss_temp = [[loss_val]]
                    #loss_temp = pd.DataFrame(data=loss_temp,columns=colmuns)
                    #loss_record_val = pd.concat([loss_record_val,loss_temp], ignore_index= True)

                running_loss = 0.0
            
    loss_record_forward.to_string(index=False)
    loss_record_forward.to_csv(f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\gen_train\\NN_loss_save\\loss_record_f_{lr}_{opt}.csv")


    #loss_record_val.to_string(index=False)
    #loss_record_val.to_csv(f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\gen_train\\NN_loss_save\\loss_record_v_{lr}_{opt}.csv")


    model_path = f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\gen_train\\NN_loss_save\\model_{lr}_{opt}.pth"
    torch.save(net.state_dict(), model_path)
    print('Finished Training')

def val(net, val_dataset):
    net.eval()
    criterion = nn.MSELoss()
    running_loss = 0

    for i, (im_path, sample) in enumerate(val_dataset):
        props = sample["props"]
        r_and_rstd = sample["r_and_rstd"]
        im = image_generate(im_path, 128)

        outputs = net(im, props)
        loss = criterion(outputs, r_and_rstd)
     
        running_loss += loss.data[0]

    return running_loss
    
    
if __name__ == "__main__":
    for opt in ["Adam", "SGD"]:
        for i in range(4):
            lr = 1/(10**(i+4))
            train(lr,opt)