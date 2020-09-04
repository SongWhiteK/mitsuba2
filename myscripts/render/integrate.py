import sys
import numpy
import enoki as ek
import render_config as config
import mitsuba

mitsuba.set_variant(config.variant)

from mitsuba.core import Spectrum, Float, UInt32, UInt64, Vector2f, Vector3f
from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file
from mitsuba.render import ImageBlock
from mitsuba.render import Emitter


def mis_weight(pdf_a, pdf_b):
    pdf_a *= pdf_a
    pdf_b *= pdf_b
    return ek.select(pdf_a > 0.0, pdf_a / (pdf_a + pdf_b), Float(0.0))


def render(scene, spp, sample_per_pass):
    """
    Generate basic objects and rays.
    Throw render job

    Args:
        scene: Target scene object
        spp: Sample per pixel
        sample_per_pass: a number of sample per pixel in one vectored calculation with GPU
    """

    if spp % sample_per_pass != 0:
        sys.exit("please set spp as multilple of sample_per_pass")

    # Generate basic objects for rendering
    sensor = scene.sensors()[0]
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()

    result = 0

    # Sampling and integrating RTE
    n_ite = int(spp / sample_per_pass)

    total_sample_count = ek.hprod(film_size) * sample_per_pass

    # Seed the sampler
    if sampler.wavefront_size() != total_sample_count:
        sampler.seed(0, total_sample_count)

    for ite in range(n_ite):
        # Get sampling position in film uv space
        pos = ek.arange(UInt32, total_sample_count)

        pos //= sample_per_pass
        scale = Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
        pos = Vector2f(Float(pos % int(film_size[0])),
                       Float(pos // int(film_size[0])))

        pos += sampler.next_2d()

        rays, weights = sensor.sample_ray_differential(
            time=0,
            sample1=sampler.next_1d(),
            sample2=pos * scale,
            sample3=0
        )

        result += render_sample(scene, sampler, rays)



