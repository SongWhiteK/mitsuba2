""" utility functions for renderign"""

import sys
sys.path.append("myscripts/gen_train")

import utils
import mitsuba
import enoki as ek
import torch

mitsuba.set_variant("gpu_rgb")

from mitsuba.core import warp

def index_spectrum(spec, idx):
    m = spec[0]

    m[ek.eq(idx, 1)] = spec[1]
    m[ek.eq(idx, 2)] = spec[2]

    return m


def resample_wo(si, sampler, active):
    d_out_local = warp.square_to_cosine_hemisphere(sampler.next_2d(active))
    d_out_pdf = warp.square_to_cosine_hemisphere_pdf(d_out_local)
    d_out = si.to_world(d_out_local)

    return d_out, d_out_pdf


