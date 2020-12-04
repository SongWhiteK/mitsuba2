import glob
import datetime
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.transforms import ToTensor
from vae_config import VAEConfiguration
from PIL import Image
from sklearn.model_selection import train_test_split

class VAEDatasets(Dataset):
    def __init__(self, config, transform=None):
        self.data = pd.read_csv(config.SAMPLE_PATH)
        self.im_dir = config.MAP_DIR
        self.transform = transform
        self.n_per_subdir = config.n_per_subdir

    def __getitem__(self, index):

        # Get csv data
        data = self.data.iloc[index]

        idx_props = ["eff_albedo", "g", "eta",
                     "d_in_x", "d_in_y", "d_in_z", "height_max"]
        props = pd.Series(data=data, index=idx_props).values
        props = props.astype(np.float32)
        props = torch.tensor(props)
        idx_in_pos = ["p_in_x", "p_in_y", "p_in_z"]
        idx_out_pos = ["p_out_x", "p_out_y", "p_out_z"]
        in_pos = pd.Series(data=data, index=idx_in_pos).values
        in_pos = in_pos.astype(np.float32)
        in_pos = torch.tensor(in_pos)
        out_pos = pd.Series(data=data, index=idx_out_pos).values
        out_pos = out_pos.astype(np.float32)
        out_pos = torch.tensor(out_pos)
        idx_abs = ["abs_prob"]
        abs_prob = pd.Series(data=data, index=idx_abs).values
        abs_prob = abs_prob.astype(np.float32)
        abs_prob = torch.tensor(abs_prob)

        idx_sigma_n = ["sigma_n"]
        sigma_n = pd.Series(data=data, index=idx_sigma_n).values
        sigma_n = sigma_n.astype(np.float32)
        sigma_n = torch.tensor(sigma_n)

        sample_id = int(data["id"])
        model_id = int(data["model_id"])

        # Get processed height map from index (~= id)
        num_subdir = (sample_id // self.n_per_subdir) * self.n_per_subdir
        im_path = f"{self.im_dir}\\map_{model_id:03}\\images{num_subdir}_{num_subdir+9999}\\train_image{sample_id:08}.png"

        sample = {}
        sample["props"] = props
        sample["in_pos"] = in_pos
        sample["out_pos"] = out_pos
        sample["abs"] = abs_prob
        sample["sigma_n"] = sigma_n

        return im_path, sample

    def __len__(self):
        return len(self.data)

from vae import loss_function


def train(config, model, device, dataset):
    torch.manual_seed(config.seed)
    print(f"{datetime.datetime.now()} -- Training Start")

    # Input model name at this training
    model_name = input("Input model neme at this training: ")
    model_path = f"myscripts/vae/model/{model_name}.pt"

    print(f"{datetime.datetime.now()} -- Data split start")
    train_data, test_data = train_test_split(dataset, test_size=0.2)
    print(f"{datetime.datetime.now()} -- Data split end")

    train_loader = DataLoader(train_data, **config.loader_args)
    test_loader = DataLoader(test_data, **config.loader_args)

    init_lr = config.lr
    decay_rate = 0.8
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    scheduler = ExponentialLR(optimizer, gamma=decay_rate)

    # Writer instanse for logging with TensorboardX
    writer = SummaryWriter(f"{config.LOG_DIR}\\{model_name}")

    for epoch in range(1, config.epoch + 1):
        print(f"epoch {epoch} start")
        train_epoch(epoch, config, model, device,
                    train_loader, optimizer, writer)
        test(epoch, config, model, device, test_loader, writer)
        scheduler.step()

    writer.close()
    torch.save(model.state_dict(), model_path)


def train_epoch(epoch, config, model, device, train_loader, optimizer, writer):
    model.train()

    for batch_idx, (im_path, sample) in enumerate(train_loader):
        props = sample["props"].to(device)
        in_pos = sample["in_pos"].to(device)
        out_pos = sample["out_pos"].to(device)
        abs_prob = sample["abs"].to(device)
        sigma_n = sample["sigma_n"].to(device)

        in_pos_scaled = in_pos / sigma_n
        out_pos_scaled = out_pos / sigma_n

        im = image_generate(im_path, config.im_size)
        
        optimizer.zero_grad()
        recon_pos_scaled, recon_abs, mu, logvar = model(props, im.to(device), in_pos_scaled, out_pos_scaled, is_training=True)
        recon_pos = recon_pos_scaled * sigma_n
        loss_total, losses = loss_function(recon_pos, out_pos, recon_abs, abs_prob, mu, logvar, config)

        loss_total.backward()
        optimizer.step()

        if(batch_idx % 750 == 0):
            day_time = datetime.datetime.now()
            n_data = batch_idx * config.loader_args["batch_size"]
            print(f"{day_time} -- Log: data {n_data} / {2000000 * 0.8}")

        # Logging with TensorboardX
        writer.add_scalar("train/total_loss", loss_total, (epoch-1) * len(train_loader) + batch_idx)
        writer.add_scalars("train/loss",
                           {
                               "latent": losses["latent"] * config.loss_weight_latent,
                               "position": losses["pos"] * config.loss_weight_pos,
                               "absorption": losses["abs"] * config.loss_weight_abs
                           },
                           (epoch - 1) * len(train_loader) + batch_idx)
        writer.add_scalars("train/loss_average",
                           {
                               "latent": (losses["latent"] / config.loader_args["batch_size"]),
                               "position": (losses["pos"] / config.loader_args["batch_size"]),
                               "absorption": (losses["abs"] / config.loader_args["batch_size"])
                           },
                           (epoch - 1) * len(train_loader) + batch_idx)
        


def test(epoch, config, model, device, test_loader, writer):
    day_time = datetime.datetime.now()
    print(f"{day_time} -- Test Start")
    model.eval()
    test_loss_total = 0
    test_loss_latent = 0
    test_loss_pos = 0
    test_loss_abs = 0

    im_show = True

    cnt_test = 0

    with torch.no_grad():
        for im_path, sample in test_loader:
            props = sample["props"].to(device)
            in_pos = sample["in_pos"].to(device)
            out_pos = sample["out_pos"].to(device)
            abs_prob = sample["abs"].to(device)
            sigma_n = sample["sigma_n"].to(device)

            in_pos_scaled = in_pos / sigma_n
            out_pos_scaled = out_pos / sigma_n

            im = image_generate(im_path, config.im_size)

            recon_pos_scaled, recon_abs, mu, logvar = model(props, im.to(device), in_pos_scaled, out_pos_scaled)
            recon_pos = recon_pos_scaled * sigma_n
            loss_total, losses = loss_function(recon_pos, out_pos, recon_abs, abs_prob, mu, logvar, config)

            test_loss_total += loss_total
            test_loss_latent += losses["latent"]
            test_loss_pos += losses["pos"]
            test_loss_abs += losses["abs"]

            cnt_test += len(im_path)

            if(im_show):
                print("recon pos diff: " + str(recon_pos[0:5, :] - in_pos[0:5, :]))
                print("ref pos diff: " + str(out_pos[0:5, :] - in_pos[0:5, :]))
                print("recon_abs: " + str(recon_abs[0:5]))
                print("ref_abs: " + str(abs_prob[0:5]))
                im_show = False

    writer.add_scalar("test/total_loss", test_loss_total, epoch)
    writer.add_scalars("test/loss",
                       {
                           "latent": test_loss_latent * config.loss_weight_latent,
                           "position": test_loss_pos * config.loss_weight_pos,
                           "absorption": test_loss_abs * config.loss_weight_abs
                       },
                       epoch)
    writer.add_scalars("test/loss_average",
                       {
                           "latent": (test_loss_latent / cnt_test),
                           "position": (test_loss_pos / cnt_test),
                           "absorption": (test_loss_abs / cnt_test)
                       },
                       epoch)

    day_time = datetime.datetime.now()
    print(f"{day_time} -- Test End")


def image_generate(im_path, im_size):
    batch_size = len(im_path)

    im = np.zeros([batch_size, 1, im_size, im_size])

    for i, path in enumerate(im_path):
        im[i, 0, :, :] = Image.open(path)

    im = im.astype(np.float32)
    im_tensor = torch.from_numpy(im).clone()

    return im_tensor


if __name__ == "__main__":
    # Load sample randomly, and show images and parameters
    config = VAEConfiguration()
    dataset = VAEDatasets(config, ToTensor())

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    i = 0

    for im_batch, sample_batch in dataloader:
        if (i % 5 == 0):
            props_batch = sample_batch["props"]
            in_pos_batch = sample_batch["in_pos"]
            out_pos_batch = sample_batch["out_pos"]
            abs_batch = sample_batch["abs"]
            print(im_batch.shape, props_batch.shape, in_pos_batch.shape, out_pos_batch.shape, abs_batch.shape)
            print(props_batch)
            print(in_pos_batch)

            im = im_batch.numpy()
            im = np.transpose(im[0],[1,2,0])[:,:,0]
            plt.imshow(im, cmap="gray")
            plt.show()
        i += 1



        
