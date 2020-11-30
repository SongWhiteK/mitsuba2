"""
NN module for VAE
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


def conv5x5(in_ch, out_ch, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(in_ch, out_ch, 5, stride)

def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(in_ch, out_ch, 3, stride)


class VAE(nn.Module):
    """
    VAE for predicting textured subsurface scattering
    """

    def __init__(self, config):
        super(VAE, self).__init__()

        ###### CONV #####
        # Input: 127x127 image
        # Output: 128x1 vector
        self.conv1 = conv5x5(1, config.ch1, config.stride)
        self.conv2 = conv3x3(config.ch1, config.ch2)
        self.conv3 = conv3x3(config.ch2, config.ch3)
        self.drop = nn.Dropout2d()
        self.pool = config.pool

        ##### Feature Network #####
        # Input: 7 properties
        # Output: 16x1 vector
        self.fn1 = nn.Linear(7, config.n_fn)
        self.fn2 = nn.Linear(config.n_fn, config.n_fn)
        self.fn3 = nn.Linear(config.n_fn, config.n_fn)

        ##### Encoder #####
        # Input: outgoing position (xyz vector)
        # Output: 4x1 normal distribution vector
        self.enc1 = nn.Linear(3, config.n_enc)
        self.enc21 = nn.Linear(config.n_enc, 4)
        self.enc22 = nn.Linear(config.n_enc, 4)

        ##### Scatter Network #####
        # Input: 148x1 (144 + 4) feature vector and random numbers from normal distribution
        # Output: outgoing position (xyz vector)
        self.scatter1 = nn.Linear(148, config.n_dec1)
        self.scatter2 = nn.Linear(config.n_dec1, config.n_dec1)
        self.scatter3 = nn.Linear(config.n_dec1, config.n_dec2)
        self.scatter4 = nn.Linear(config.n_dec2, 3)

        ##### Absorption Network #####
        # Input: 144x1 feature vector
        # Output: scalar absorption
        self.abs1 = nn.Linear(144, config.n_dec1)
        self.abs2 = nn.Linear(config.n_dec1, 1)
        self.sigmoid = nn.Sigmoid()

    def feature_conversion(self, im, props):
        """Conversion from height map and properties to feature vector"""

        ##### Height map conversion #####
        im_feature = F.relu(F.max_pool2d(self.conv1(im), self.pool))
        im_feature = F.relu(F.max_pool2d(self.conv2(im_feature), self.pool))
        im_feature = F.relu(F.max_pool2d(self.drop(self.conv3(im_feature)), self.pool))
        im_feature = im_feature.view(-1, 128)

        ##### Feature conversion #####
        feature = F.relu(self.fn1(props))
        feature = F.relu(self.fn2(feature))
        
        feature = torch.cat([feature, im_feature], dim=1)

        return feature



    def encode(self, x):
        """Encoding from outgoing position to average and standard deviation of normal distribution"""
        x = F.relu(self.enc1(x))
        return self.enc21(x), self.enc22(x)

    def reparameterize(self, mu, logvar):
        """Reparametarization trick for VAE"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, feature, z):
        """
        Decoding from feature vector and latent variables to scatter position and absorption probability

        Args:
            feature: Feature vector from feature network and cnn
            z: Latent variables
        
        Returns:
            scatter: Reconstructed outgoing position
            absorption: Reconstructed absorption probability
        """
        ##### Scatter Network #####
        feature_sc = torch.cat([feature, z], dim=1)

        scatter = F.relu(self.scatter1(feature_sc))
        scatter = F.relu(self.scatter2(scatter))
        scatter = F.relu(self.scatter3(scatter))
        scatter = self.scatter4(scatter) # this outputs outgoing position

        ##### Absorption Network #####
        absorption = F.relu(self.abs1(feature))
        absorption = self.sigmoid(self.abs2(absorption)) # this outputs absorption probability

        return scatter, absorption


    def forward(self, props, im, in_pos, out_pos):
        # position difference between incident and outgoing
        diff_pos = out_pos - in_pos



        # Generate feature vector
        feature = self.feature_conversion(im, props)

        # Encode from position difference to latent variables
        mu, logvar = self.encode(diff_pos)
        z = self.reparameterize(mu, logvar)

        # Decode from latent variables and feature vector to reconstructed position
        recon_pos, recon_abs = self.decode(feature, z)

        # As world coordinates
        recon_pos += in_pos

        return recon_pos, recon_abs, mu, logvar

def loss_function(recon_pos, ref_pos, recon_abs, ref_abs, mu, logvar, config):
    # Latent loss
    loss_latent = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Outgoing position loss
    loss_position = F.smooth_l1_loss(recon_pos, ref_pos, reduction="sum")

    # Absorption loss

    loss_absorption = F.mse_loss(recon_abs, ref_abs, reduction="sum")

    losses = {}
    losses["latent"] = loss_latent
    losses["pos"] = loss_position
    losses["abs"] = loss_absorption

    return config.loss_weight_latent * loss_latent + config.loss_weight_pos * loss_position + config.loss_weight_abs * loss_absorption, losses




if __name__ == "__main__":
    from trainer import VAEDatasets, train
    time1 = time.time()

    config = VAEConfiguration()
    device = torch.device("cuda")
    print(f"{datetime.now()} -- Model generation")
    model = VAE(config).to(device)
    
    print(f"{datetime.now()} -- Dataset generation")
    dataset = VAEDatasets(config, ToTensor())

    # Visualize network in Tensorboard
    # model.eval()
    # writer = SummaryWriter(config.LOG_DIR)
    # im = np.random.randint(0, 255, [1, 1, 127, 127]).astype(np.float32)
    # im = torch.tensor(im).to(device)

    # props = torch.randn([1, 7]).to(device)
    # in_pos = torch.randn([1, 3]).to(device)
    # out_pos = torch.randn([1, 3]).to(device)

    # writer.add_graph(model, (props, im, in_pos, out_pos))
    # writer.close()

    train(config, model, device, dataset)
    print(time.time() - time1)

    




    
