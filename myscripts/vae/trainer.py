import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.transforms import ToTensor
from vae_config import VAEConfiguration
from PIL import Image

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
        props = torch.tensor(props)
        idx_in_pos = ["p_in_x", "p_in_y", "p_in_z"]
        idx_out_pos = ["p_out_x", "p_out_y", "p_out_z"]
        in_pos = pd.Series(data=data, index=idx_in_pos).values
        in_pos = torch.tensor(in_pos)
        out_pos = pd.Series(data=data, index=idx_out_pos).values
        out_pos = torch.tensor(out_pos)

        abs_prob = torch.tensor(data["abs_prob"])

        sample = {}
        sample["props"] = props
        sample["in_pos"] = in_pos
        sample["out_pos"] = out_pos
        sample["abs"] = abs_prob

        return im, sample

    def __len__(self):
        return len(self.data)


def train(config, model, device, dataset):
    torch.manual_seed(config.seed)

    train_loader = DataLoader(dataset, **config.loader_args)

    init_lr = config.lr
    min_lr = init_lr / 8.0
    decay_rate = 0.8
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    scheduler = ExponentialLR(optimizer, gamma=decay_rate)

    for epoch in range(1, config.epoch + 1):
        train_epoch(epoch, config, model, device, train_loader, optimizer)

    


        
        
        


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

        
