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


