import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
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

        idx_material = ["albedo", "g", "eta"]
        material = pd.Series(data=data, index=idx_material).values
        material = torch.tensor(material)
        idx_pos = ["p_out_x", "p_out_y", "p_out_z",
                   "d_in_x", "d_in_y", "d_in_z"]
        pos = pd.Series(data=data, index=idx_pos).values
        pos = torch.tensor(pos)

        abs_prob = torch.tensor(data["abs_prob"])

        sample = {}
        sample["material"] = material
        sample["pos"] = pos
        sample["abs"] = abs_prob

        return im, sample

    def __len__(self):
        return len(self.data)
        
        
        


if __name__ == "__main__":
    # Load sample randomly, and show images and parameters
    config = VAEConfiguration()
    dataset = VAEDatasets(config, ToTensor())

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    i = 0

    for im_batch, sample_batch in dataloader:
        if (i % 5 == 0):
            material_batch = sample_batch["material"]
            pos_batch = sample_batch["pos"]
            abs_batch = sample_batch["abs"]
            print(im_batch.shape, material_batch.shape, pos_batch.shape, abs_batch.shape)
            print(material_batch)
            print(pos_batch)

            im = im_batch.numpy()
            im = np.transpose(im[0],[1,2,0])[:,:,0]
            plt.imshow(im, cmap="gray")
            plt.show()
        i += 1

        
