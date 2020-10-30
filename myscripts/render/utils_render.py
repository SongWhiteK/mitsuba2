""" utility functions for renderign"""

import sys
sys.path.append("myscripts/gen_train")

import utils
import mitsuba
import enoki as ek
import torch

mitsuba.set_variant("gpu_rgb")

def index_spectrum(spec, idx):
    m = spec[0]

    m[ek.eq(idx, 1)] = spec[1]
    m[ek.eq(idx, 2)] = spec[2]

    return m


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
    eff_albedo = utils.reduced_albedo_to_effective_albedo(
        utils.get_reduced_albedo(albedo, g, sigma_t)
        ).torch().view(-1, 1)

    g = g.torch().view(-1, 1)
    eta = bs.eta.torch().view(-1, 1)

    d_in = si.wi.torch()
    height_max = bs.height_max.torch().view(-1, 1)

    props = torch.cat([eff_albedo, g, eta, d_in, height_max], 1)
    print(props)
    print(props.size())
    

