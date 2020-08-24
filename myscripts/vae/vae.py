"""
NN module for VAE
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def conv5x5(in_ch, out_ch, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(in_ch, out_ch, 5, stride)


class VAE(nn.Module):
    """
    VAE for predicting textured subsurface scattering
    """

    def __init__(self, config):
        super(VAE, self).__init__()

        ###### CONV #####
        # Input: 256x256 image
        # Output: 512x1 vector
        self.conv1 = conv5x5(1, config.ch1, self.stirde)
        self.conv2 = conv5x5(config.ch1, config.ch2)
        self.conv3 = conv5x5(config.ch2, config.ch3)
        self.drop = nn.Dropout2d()

        ##### Feature Network #####
        # Input: 6 properties
        # Output: 16x1 vector
        self.fn1 = nn.Linear(6, config.n_fn)
        self.fn2 = nn.Linear(config.n_fn, config.n_fn)
        self.fn3 = nn.Linear(config.n_fn, config.n_fn)

        ##### Encoder #####
        # Input: outgoing position (xyz vector)
        # Output: 4x1 normal distribution vector
        self.enc1 = nn.Linear(3, config.n_enc)
        self.enc21 = nn.Linear(config.n_enc, 4)
        self.enc22 = nn.Linear(config.n_enc, 4)

        ##### Scatter Network #####
        # Input: 1172x1 (1168 + 4) feature vector and random numbers from normal distribution
        # Output: outgoing position (xyz vector)
        self.sn1 = nn.Linear(1172, config.n_dec1)
        self.sn2 = nn.Linear(config.n_dec1, config.n_dec1)
        self.sn3 = nn.Linear(config.n_dec1, config.n_dec2)
        self.sn4 = nn.Linear(config.n_dec2, 3)

        ##### Absorption Network #####
        # Input: 1168x1 feature vector
        # Output: scalar absorption
        self.abs1 = nn.Linear(1168, config.n_dec1)
        self.abs2 = nn.Linear(config.n_dec1, 1)

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

        scatter = F.relu(self.sn1(feature_sc))
        scatter = F.relu(self.sn2(scatter))
        scatter = F.relu(self.sn3(scatter))
        scatter = self.sn4(scatter) # this outputs outgoing position

        ##### Absorption Network #####
        absorption = F.relu(self.abs1(feature))
        absorption = F.sigmoid(self.abs2(feature)) # this outputs absorption probability

        return scatter, absorption



    def forward(self, x, im, config, out_pos=None):
        ##### Height map conversion #####
        im_feature = F.relu(F.max_pool2d(self.conv1(im), config.pool))
        im_feature = F.relu(F.max_pool2d(self.conv2(im_feature), config.pool))
        im_feature = F.relu(F.max_pool2d(self.drop(self.conv3(im_feature), config.pool)))
        im_feature = im_feature.view(1, -1)

        ##### Feature conversion #####
        feature = F.relu(self.fn1(x))
        feature = F.relu(self.fn2(feature))
        feature = F.relu(self.fn3(feature))

        feature = torch.cat([feature, im_feature], dim=1)

        ##### Encoder #####
        mu, logvar = self.encode(out_pos)
        z = self.reparameterize(mu, logvar)

        ##### Decoder #####
        recon_pos, recon_abs = self.decode(feature, z)

        return recon_pos, recon_abs, mu, logvar

def loss_function(recon_pos, ref_pos, recon_abs, ref_abs, mu, logvar, config):
    # Latent loss
    loss_latent = -0.5 * torch.sum(1 + logvar - mu.pow - logvar.exp())

    # Outgoing position loss
    loss_position = config.loss_weight_pos * F.smooth_l1_loss(recon_pos, ref_pos, reduction="sum")

    # Absorption loss
    loss_absorption = config.loss_weight_abs * F.mse_loss(recon_abs, ref_abs, reduction="sum")

    return loss_latent + loss_position + loss_absorption






    
