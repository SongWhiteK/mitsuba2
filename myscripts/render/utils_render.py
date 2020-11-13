""" utility functions for renderign"""

import sys
sys.path.append("myscripts/gen_train")

import utils
import mitsuba
import enoki as ek
import torch

mitsuba.set_variant("gpu_rgb")

from mitsuba.core import Float, warp

def index_spectrum(spec, idx):
    m = spec[0]

    m[ek.eq(idx, 1)] = spec[1]
    m[ek.eq(idx, 2)] = spec[2]

    return m


def resample_wo(si, sampler, active):
    d_out_local = warp.square_to_cosine_hemisphere(sampler.next_2d(active))
    d_out_pdf = warp.square_to_cosine_hemisphere_pdf(d_out_local)

    return d_out_local, d_out_pdf


def check_zero_scatter(sampler, si, bs, channel, active):

    sigma_t = index_spectrum(bs.sigma_t, channel)

    # Ray passes through medium w/o scattering?
    is_zero_scatter = (sampler.next_1d(active) > Float(1) - ek.exp(-sigma_t * si.t)) & active

    return is_zero_scatter

