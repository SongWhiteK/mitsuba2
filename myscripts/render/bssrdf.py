"""Process BSSRDF with VAE"""
import os
import sys
sys.path.append("./myscripts/vae")
sys.path.append("./myscripts/gen_train")

import torch
from torch._C import device
import render_config
import vae_config
import mitsuba
import enoki as ek
import utils
from utils_render import index_spectrum

mitsuba.set_variant(render_config.variant)

from mitsuba.core import (Vector3f, Float, Spectrum)
from mitsuba.render import (BSDF)
from vae import VAE


class BSSRDF:
    """BSSRDF class with VAE"""

    def __init__(self, model_name):
        """Instanciation the VAE class and load a trained model"""

        self.config = vae_config.VAEConfiguration()
        self.device = torch.device("cuda")

        # Instanciate and load trained model
        model_path = f"{self.config.MODEL_DIR}\\{model_name}.pt"
        self.model = VAE(self.config).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))

    def estimate(self, in_pos, im, props, sigma_n, active):
        """
        Estimate output position and absorption with VAE
        Notice that the types of arguments are in pytorch (tensor) except sigma_n, but the ones of returns are in mitsuba (Vector3f, Float)
        Args:
            in_pos: Incident position in local mesh coordinates, and the type of this is tensor
            im: Height map around incident position ranging to multiple of sigma_n.
                This map can be generated by clip_scaled_map() from data_handler, and the type is tensor
            props: Medium properties vector including (and following this order)
                   - effective albedo
                   - g
                   - eta
                   - incident angle (xyz)
                   - max height
                   , and the type of this argument is tensor
            sigma_n: Standard deviation of the range of medium scattering.
                     In this method, this value is used as scale factor of coordinates in vae
            active: Boolean mask which indicates whether a ray is applied VAE or not

        Return:
            recon_pos: estimated outgoing position (Vector3f)
            recon_abs: estimated absorption probability (Spectrum)
        """

        n_sample, _, _, _ = im.shape
        pos = Vector3f(in_pos)
        abs_prob = Float().zero(n_sample)

        self.model.eval()

        with torch.no_grad():
            # Feature conversion
            feature = self.model.feature_conversion(im.to(self.device, dtype=torch.float), props.to(self.device, dtype=torch.float))

            # Sample latent variable from normal distribution
            z = torch.randn(n_sample, 4).to(self.device, dtype=torch.float)

            # Decode and get reconstructed position and absorption
            recon_pos, recon_abs = self.model.decode(feature, z)

        # Convert from tensor to Vector3f and Float
        recon_pos = Vector3f(recon_pos)
        recon_abs[torch.isnan(recon_abs)] = 0
        abs_prob = Spectrum(recon_abs.view(1,-1).squeeze())

        # Reconstruct real scale position in mesh local coordinates
        pos += ek.select(active, sigma_n * recon_pos, 0)

        return pos, abs_prob


    def sample_bssrdf(self, scene, bsdf, bs, si, bdata, channel, active):
        
        # Get ID of BSSDRF mesh for each sur face interactions
        mesh_id = BSDF.mesh_id_vec(bsdf, active)

        # Convert incident position into local coordinates of mesh of interested as tensor
        in_pos = ek.select(active, si.to_mesh_local(bs), Vector3f(0))

        # Get properties, e.g., medium params and incident angle as tensor
        props, sigma_n = get_props(bs, si, channel)

        # Get height map around incident position as tensor
        im = bdata.get_height_map(in_pos, mesh_id)

        # Estimate position and absorption probability with VAE as mitsuba types
        recon_pos_local, abs_recon = self.estimate(in_pos.torch(), im, props, sigma_n, active)

        # Convert from mesh coordinates to world coordinates
        recon_pos_world = si.to_mesh_world(bs, recon_pos_local)

        # Project estimated position onto nearest mesh
        projected_si, proj_suc = si.project_to_mesh_normal(scene, recon_pos_world, bs, channel, active)

        return projected_si, proj_suc, abs_recon


def get_props(bs, si, channel):
    """
    Get property tensor for vae
    
    Args:
        bs: BSDFSample3f
        si: SurfaceInteraction3f
        channel: RGB channel of interest

    Return:
        props: Property tensor including 
               - effective albedo
               - g
               - eta
               - incident angle (xyz)
               - max height
    """

    albedo = index_spectrum(bs.albedo, channel)
    g = bs.g
    sigma_t = index_spectrum(bs.sigma_t, channel)
    medium = {}
    medium["albedo"] = albedo
    medium["g"] = g
    medium["sigma_t"] = sigma_t
    sigma_n = utils.get_sigman(medium)
    eff_albedo = utils.reduced_albedo_to_effective_albedo(
        utils.get_reduced_albedo(albedo, g, sigma_t)
        ).torch().view(-1, 1)

    g = g.torch().view(-1, 1)
    eta = bs.eta.torch().view(-1, 1)

    d_in = si.wi.torch()
    height_max = bs.height_max.torch().view(-1, 1)

    props = torch.cat([eff_albedo, g, eta, d_in, height_max], 1)

    return props, sigma_n

