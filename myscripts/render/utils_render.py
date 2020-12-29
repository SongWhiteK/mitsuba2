""" utility functions for renderign"""

import sys
sys.path.append("myscripts/gen_train")

import mitsuba
import enoki as ek
import torch

mitsuba.set_variant("gpu_rgb")

from mitsuba.core import Float, warp
from mitsuba.render import ImageBlock

def index_spectrum(spec, idx):
    m = spec[0]

    m[ek.eq(idx, 1)] = spec[1]
    m[ek.eq(idx, 2)] = spec[2]

    return m


def resample_wo(sampler, active):
    d_out_local = warp.square_to_cosine_hemisphere(sampler.next_2d(active))
    d_out_pdf = warp.square_to_cosine_hemisphere_pdf(d_out_local)

    return d_out_local, d_out_pdf


def check_zero_scatter(sampler, si, bs, channel, active):

    sigma_t = index_spectrum(bs.sigma_t, channel)

    # Ray passes through medium w/o scattering?
    is_zero_scatter = (sampler.next_1d(active) > Float(1) - ek.exp(-sigma_t * si.t)) & active

    return is_zero_scatter

def reduced_albedo_to_effective_albedo(reduced_albedo):
    return -ek.log(1.0 - reduced_albedo * (1.0 - ek.exp(-8.0))) / 8.0


def gen_blocks(crop_size, channel_count=5,
               filter=film.reconstruction_filter(), border=False):

    block = ImageBlock(
            crop_size,
            channel_count=channel_count,
            filter=filter,
            border=border
    )
    block.clear()

    block_scatter = ImageBlock(
            crop_size,
            channel_count=channel_count,
            filter=filter,
            border=border
    )
    block_scatter.clear()

    block_nonscatter = ImageBlock(
            crop_size,
            channel_count=channel_count,
            filter=filter,
            border=border
    )
    block_nonscatter.clear()

    blocks = [block, block_scatter, block_nonscatter]

    return blocks
