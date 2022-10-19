"""General NN Training Scripts"""


import os
import random
from tkinter import HIDDEN
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
from net import Net
import copy

class Net_optuna(nn.Module):
    def __init__(self, hidden_dim, p_drop):
        super(Net_optuna, self).__init__()
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
            nn.Dropout(p=self.p_drop),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.p_drop),
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

def train(lr=0.0001, opt="Adam", hidden_dim=64):
    device = torch.device("cuda")
    net = Net().to(device)
    if(opt == "Adam"):
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)

    criterion = nn.MSELoss()
    train_dataset = CNNDatasets_train()
    test_dataset = CNNDatasets_test()
    train_loader = DataLoader(train_dataset, batch_size=16,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16,shuffle=False)

    #n_samples = len(train_loader) 
    #train_size = int(len(train_loader) * 0.8) 
    #val_size = n_samples - train_size 

    #train_dataset, val_dataset = torch.utils.data.random_split(train_loader, [train_size, val_size])

    best_loss = 10000
    epochs = 1000
    colmuns = ["loss"]
    loss_record_forward = pd.DataFrame(columns=colmuns)
    loss_record_test = pd.DataFrame(columns=colmuns)
    #loss_record_val = pd.DataFrame(columns=colmuns)

    for epoch in range(epochs):
        print(f"start epoch {epoch+1}")
        net.train()
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
                
                if((i+1) % 280 == 0):
                    loss_temp = [[running_loss]]
                    loss_temp = pd.DataFrame(data=loss_temp,columns=colmuns)
                    loss_record_forward = pd.concat([loss_record_forward,loss_temp], ignore_index= True)

                    #loss_val = val(net, val_dataset)
                    #loss_temp = [[loss_val]]
                    #loss_temp = pd.DataFrame(data=loss_temp,columns=colmuns)
                    #loss_record_val = pd.concat([loss_record_val,loss_temp], ignore_index= True)

                running_loss = 0.0

        loss_test = test_NN(net, test_loader, device)
        loss_temp = [[loss_test]]
        loss_temp = pd.DataFrame(data=loss_temp,columns=colmuns)
        loss_record_test = pd.concat([loss_record_test,loss_temp], ignore_index= True)
        print(loss_test)
        if best_loss > loss_test:
            best_model = copy.deepcopy(net)
            best_loss = loss_test
            print("updated best model")
        else:
            print(f"best model:{best_loss}")

            
    loss_record_forward.to_string(index=False)
    loss_record_forward.to_csv(f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\loss_record_f_0901_{lr}_{opt}.csv")

    loss_record_test.to_string(index=False)
    loss_record_test.to_csv(f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\loss_record_test_0901_{lr}_{opt}.csv")


    #loss_record_val.to_string(index=False)
    #loss_record_val.to_csv(f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\gen_train\\NN_loss_save\\loss_record_v_{lr}_{opt}.csv")


    model_path = f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\best_model_0901.pth"
    torch.save(best_model.state_dict(), model_path)
    print('Finished Training')

def val(net, val_dataset_loader, device):
    net.eval()
    criterion = nn.MSELoss()
    running_loss = 0

    for i, (im_path, sample) in enumerate(val_dataset_loader):

        props = sample["props"]
        r_and_rstd = sample["r_and_rstd"]
        im = image_generate(im_path, 128)

        props, r_and_rstd, im = props.to(device), r_and_rstd.to(device), im.to(device)

        outputs = net(im, props)
        loss = criterion(outputs, r_and_rstd)
     
        running_loss += loss.item()

    return running_loss

def test_NN(net, test_dataset_loader, device):
    net.eval()
    criterion = nn.MSELoss()
    running_loss = 0

    for i, (im_path, sample) in enumerate(test_dataset_loader):
        
        props = sample["props"]
        r_and_rstd = sample["r_and_rstd"]
        im = image_generate(im_path, 128)

        props, r_and_rstd, im = props.to(device), r_and_rstd.to(device), im.to(device)

        outputs = net(im, props)
        loss = criterion(outputs, r_and_rstd)
     
        running_loss += loss.item()

    return running_loss

def objective(trial):
    # TODO:epoch automation

    max_epoch = 300
    device = "cuda"

    learning_rate = trial.suggest_discrete_uniform("learning_rate", 1e-5, 1e-2, q=1e-5)
    optimizer = trial.suggest_categorical("optimizer", ["Adam"])
    hidden_dim = trial.suggest_int("hidden_dim", 8, 200, 8)
    p_drop = trial.suggest_discrete_uniform("p_drop", 1e-2, 0.5, q=1e-2)

    print(learning_rate,optimizer,hidden_dim)
    net = Net_optuna(hidden_dim=hidden_dim,p_drop=p_drop).to(device)
    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer)(net.parameters(), lr=learning_rate)

    train_dataset = CNNDatasets_train()
    test_dataset = CNNDatasets_test()
    train_loader = DataLoader(train_dataset, batch_size=16,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16,shuffle=False)

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


        return validation_loss

def study_params(n_trials):
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)
    print(f"=====best param=====\n{study.best_params}")

def retrain(lr=0.0001, opt="Adam", hidden_dim=64):
    device = torch.device("cuda")
    net = Net().to(device)
    net.load_state_dict(torch.load("D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\best_model_0814.pth"))

    if(opt == "Adam"):
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)

    criterion = nn.MSELoss()
    train_dataset = CNNDatasets_train()
    test_dataset = CNNDatasets_test()
    train_loader = DataLoader(train_dataset, batch_size=16,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16,shuffle=False)

    #n_samples = len(train_loader) 
    #train_size = int(len(train_loader) * 0.8) 
    #val_size = n_samples - train_size 

    #train_dataset, val_dataset = torch.utils.data.random_split(train_loader, [train_size, val_size])

    best_loss = 10000
    epochs = 1000
    colmuns = ["loss"]
    loss_record_forward = pd.DataFrame(columns=colmuns)
    loss_record_test = pd.DataFrame(columns=colmuns)
    #loss_record_val = pd.DataFrame(columns=colmuns)

    for epoch in range(epochs):
        print(f"start epoch {epoch+1}")
        net.train()
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

        loss_test = test_NN(net, test_loader, device)
        loss_temp = [[loss_test]]
        loss_temp = pd.DataFrame(data=loss_temp,columns=colmuns)
        loss_record_test = pd.concat([loss_record_test,loss_temp], ignore_index= True)
        print(loss_test)
        if best_loss > loss_test:
            best_model = copy.deepcopy(net)
            best_loss = loss_test
            print("updated best model")

            
    loss_record_forward.to_string(index=False)
    loss_record_forward.to_csv(f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\loss_record_f_{lr}_{opt}.csv")

    loss_record_test.to_string(index=False)
    loss_record_test.to_csv(f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\loss_record_test_{lr}_{opt}.csv")


    #loss_record_val.to_string(index=False)
    #loss_record_val.to_csv(f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\gen_train\\NN_loss_save\\loss_record_v_{lr}_{opt}.csv")


    model_path = f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\best_model_0815.pth"
    torch.save(best_model.state_dict(), model_path)
    print('Finished Training')


if __name__ == "__main__":

    start = time.time()
    train(lr=0.00031,hidden_dim=64)

    # study_params(n_trials=200)

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    #{'learning_rate': 0.00031000000000000005, 'optimizer': 'Adam', 'hidden_dim': 72}
    #'learning_rate': 0.0006600000000000001, 'optimizer': 'Adam', 'hidden_dim': 200, 'p_drop': 0.17}. Best is trial 26 with value: 3.196924924850464.