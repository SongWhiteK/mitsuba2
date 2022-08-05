import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from dataset import CNNDatasets_test, CNNDatasets_train, image_generate
from net import Net
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import time

def train(lr=0.0001, opt="Adam"):
    device = torch.device("cuda")
    train_dataset = CNNDatasets_train()
    test_dataset = CNNDatasets_test()

    net = Net().to(device)
    if(opt == "Adam"):
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)

    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=16,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16,shuffle=False)

    #n_samples = len(train_loader) 
    #train_size = int(len(train_loader) * 0.8) 
    #val_size = n_samples - train_size 

    #train_dataset, val_dataset = torch.utils.data.random_split(train_loader, [train_size, val_size])

    net.train()
    epochs = 1000
    colmuns = ["loss"]
    loss_record_forward = pd.DataFrame(columns=colmuns)
    loss_record_test = pd.DataFrame(columns=colmuns)
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

        loss_test = test_NN(net, test_loader, device)
        loss_temp = [[loss_test]]
        loss_temp = pd.DataFrame(data=loss_temp,columns=colmuns)
        loss_record_test = pd.concat([loss_record_test,loss_temp], ignore_index= True)
        print(loss_test)
        

            
    loss_record_forward.to_string(index=False)
    loss_record_forward.to_csv(f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\loss_record_f_{lr}_{opt}.csv")

    loss_record_test.to_string(index=False)
    loss_record_test.to_csv(f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\loss_record_test_{lr}_{opt}.csv")


    #loss_record_val.to_string(index=False)
    #loss_record_val.to_csv(f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\gen_train\\NN_loss_save\\loss_record_v_{lr}_{opt}.csv")


    model_path = f"D:\\kenkyu\\mine\\mitsuba2\\myscripts\\train_data\\NN_loss_save\\model_{lr}_{opt}.pth"
    torch.save(net.state_dict(), model_path)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_layers = trial.suggest_int("n_layers", 1, 5)
    hidden_dim = trial.suggest_int("hidden_dim", 8, 256, 8)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_discrete_uniform("learning_rate", 1e-5, 1e-1, q=1e-5)

    net = Net(input_dim=331, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = getattr(torch.optim, optimizer)(net.parameters(), lr=learning_rate)

    # train 
    net.train()
    for epoch in range(max_epoch):
        for batch in train_dataloader:
            x, t = batch
            x.to(device)
            t.to(device)

            optimizer.zero_grad()
            y = net(x)
            loss = criterion(y, t)
            loss.backward()
            optimizer.step()

    # validation
    net.eval()
    validation_loss = 0.0
    with torch.no_grad():
    for batch in test_dataloader:
        x, t = batch
        x.to(device)
        t.to(device)
        y = net(x)
        loss = criterion(y, t)
        validation_loss += loss.item() * x.size(0)

    validation_loss = validation_loss / len(test_dataloader.dataset)

    return validation_loss


if __name__ == "__main__":
    start = time.time()
    for i in range(9):
        train(0.001*(i+1))
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")