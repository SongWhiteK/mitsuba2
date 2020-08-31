import glob
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from vae_config import VAEConfiguration
from PIL import Image
from sklearn.model_selection import train_test_split

class VAEDatasets(Dataset):
    def __init__(self, config, transform=None):
        self.data = pd.read_csv(config.SAMPLE_PATH)
        self.im_list = glob.glob(f"{config.MAP_DIR}\\*")
        self.transform = transform

    def __getitem__(self, index):
        # Get processed height map from index (~= id)
        path = self.im_list[index]
        im = Image.open(path)

        if self.transform is not None:
            im = self.transform(im)

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


        sample = {}
        sample["props"] = props
        sample["in_pos"] = in_pos
        sample["out_pos"] = out_pos
        sample["abs"] = abs_prob

        return im, sample

    def __len__(self):
        return len(self.data)

from vae import loss_function


def train(config, model, device, dataset):
    torch.manual_seed(config.seed)

    train_data, test_data = train_test_split(dataset, test_size=0.2)

    train_loader = DataLoader(train_data, **config.loader_args)
    test_loader = DataLoader(test_data, **config.loader_args)

    init_lr = config.lr
    min_lr = init_lr / 8.0
    decay_rate = 0.8
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    scheduler = ExponentialLR(optimizer, gamma=decay_rate)

    # Writer instanse for logging with TensorboardX
    writer = SummaryWriter(config.LOG_DIR)

    for epoch in range(1, config.epoch + 1):
        train_epoch(epoch, config, model, device, train_loader, optimizer, writer)
        test(epoch, config, model, device, test_loader, writer)
    
    writer.close()


def train_epoch(epoch, config, model, device, train_loader, optimizer, writer):
    model.train()

    for batch_idx, (im, sample) in enumerate(train_loader):
        props = sample["props"].to(device)
        in_pos = sample["in_pos"].to(device)
        out_pos = sample["out_pos"].to(device)
        abs_prob = sample["abs"].to(device)

        optimizer.zero_grad()
        recon_pos, recon_abs, mu, logvar = model(props, im.to(device), in_pos, out_pos)
        loss_total, losses = loss_function(recon_pos, out_pos, recon_abs, abs_prob, mu, logvar, config)

        loss_total.backward()
        optimizer.step()

        # Logging with TensorboardX
        writer.add_scalar("train/total_loss", loss_total, (epoch-1) * len(train_loader) + batch_idx)
        writer.add_scalars("train/loss",
                           {
                               "latent": losses["latent"],
                               "position": losses["pos"],
                               "absorption": losses["abs"]
                           },
                           (epoch - 1) * len(train_loader) + batch_idx)

        
def test(epoch, config, model, device, test_loader, writer):
    model.eval()
    test_loss_total = 0
    test_loss_latent = 0
    test_loss_pos = 0
    test_loss_abs = 0

    with torch.no_grad():
        for im, sample in test_loader:
            props = sample["props"].to(device)
            in_pos = sample["in_pos"].to(device)
            out_pos = sample["out_pos"].to(device)
            abs_prob = sample["abs"].to(device)

            recon_pos, recon_abs, mu, logvar = model(props, im.to(device), in_pos, out_pos)

            loss_total, losses = loss_function(recon_pos, out_pos, recon_abs, abs_prob, mu, logvar, config)

            test_loss_total += loss_total
            test_loss_latent += losses["latent"]
            test_loss_pos += losses["pos"]
            test_loss_abs += losses["abs"]
            
    writer.add_scalar("test/total_loss", loss_total, epoch)
    writer.add_scalars("test/loss",
                       {
                           "latent": losses["latent"],
                           "position": losses["pos"],
                           "absorption": losses["abs"]
                       },
                       epoch)


    


        
        
        


if __name__ == "__main__":
    # Load sample randomly, and show images and parameters
    config = VAEConfiguration()
    dataset = VAEDatasets(config, ToTensor())

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    i = 0

    for im_batch, props_batch in dataloader:
        if (i % 5 == 0):
            material_batch = props_batch["material"]
            pos_batch = props_batch["pos"]
            abs_batch = props_batch["abs"]
            print(im_batch.shape, material_batch.shape, pos_batch.shape, abs_batch.shape)
            print(material_batch)
            print(pos_batch)

            im = im_batch.numpy()
            im = np.transpose(im[0],[1,2,0])[:,:,0]
            plt.imshow(im, cmap="gray")
            plt.show()
        i += 1



        
