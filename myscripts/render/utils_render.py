""" utility functions for renderign"""

import sys
sys.path.append("myscripts/gen_train")

import mitsuba
import enoki as ek
import torch

mitsuba.set_variant("gpu_rgb")

from mitsuba.core import Float, warp, Color3f, srgb_to_xyz
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


def gen_blocks(crop_size, filter, channel_count=5, border=False):

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


def postprocess_render(results, weights, blocks, pos):

    result = results[0]
    valid_rays = results[1]
    scatter = results[2]
    non_scatter = results[3]

    result *= weights
    scatter *= weights
    non_scatter *= weights

    xyz = Color3f(srgb_to_xyz(result))
    xyz_scatter = Color3f(srgb_to_xyz(scatter))
    xyz_nonscatter = Color3f(srgb_to_xyz(non_scatter))

    aovs = [xyz[0], xyz[1], xyz[2],
            ek.select(valid_rays, Float(1.0), Float(0.0)),
            1.0]
    aovs_scatter = [xyz_scatter[0], xyz_scatter[1], xyz_scatter[2],
            ek.select(valid_rays, Float(1.0), Float(0.0)),
            1.0]
    aovs_nonscatter = [xyz_nonscatter[0], xyz_nonscatter[1], xyz_nonscatter[2],
            ek.select(valid_rays, Float(1.0), Float(0.0)),
            1.0]

    block = blocks[0]
    block_scatter = blocks[1]
    block_nonscatter = blocks[2]

    block.put(pos, aovs)
    block_scatter.put(pos, aovs_scatter)
    block_nonscatter.put(pos, aovs_nonscatter)




