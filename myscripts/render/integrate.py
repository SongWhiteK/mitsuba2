from time import time
import sys
import numpy as np
import torch
import mitsuba
import enoki as ek
import render_config as config
import utils_render
from bssrdf import BSSRDF

mitsuba.set_variant(config.variant)

from mitsuba.core import (Spectrum, Float, UInt32, Vector2f, Vector3f,
                          Frame3f, Color3f, RayDifferential3f, srgb_to_xyz)
from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file
from mitsuba.render import ImageBlock
from mitsuba.render import (Emitter, BSDF, BSDFContext, BSDFFlags, has_flag,
                            DirectionSample3f, SurfaceInteraction3f)


def mis_weight(pdf_a, pdf_b):
    pdf_a *= pdf_a
    pdf_b *= pdf_b
    return ek.select(pdf_a > 0.0, pdf_a / (pdf_a + pdf_b), Float(0.0))


def render(scene, spp, sample_per_pass, bdata):
    """
    Generate basic objects and rays.
    Throw render job

    Args:
        scene: Target scene object
        spp: Sample per pixel
        sample_per_pass: a number of sample per pixel in one vectored
                         calculation with GPU
    """


    # Generate basic objects for rendering
    sensor = scene.sensors()[0]
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    blocks = utils_render.gen_blocks(
             film.crop_size(),
             channel_count=5,
             filter=film.reconstruction_filter(),
             border=False
    )

    result = 0

    bssrdf = None
    if(config.enable_bssrdf):
        bssrdf = BSSRDF(config.model_name)

    np.random.seed(seed=config.seed)

    # The number of samples in one gpu rendering
    total_sample_count = int(ek.hprod(film_size) * spp)

    if total_sample_count % sample_per_pass != 0:
        sys.exit("total_sample_count is not multilple of sample_per_pass")
    n_ite = int(total_sample_count / sample_per_pass)

    # Seed the sampler
    if sampler.wavefront_size() != sample_per_pass:
        sampler.seed(0, sample_per_pass)

    # pos = ek.arange(UInt32, total_sample_count)
    pos = np.arange(total_sample_count, dtype=np.uint32)

    pos //= spp
    scale = Vector2f(1.0 / film_size[0], 1.0 / film_size[1])

    pos_x = (pos % int(film_size[0])).astype(np.float)
    pos_x += np.random.rand(total_sample_count)
    pos_y = (pos // int(film_size[0])).astype(np.float)
    pos_y += np.random.rand(total_sample_count)

    cnt = 0

    heightmap_pybind = bdata.get_heightmap_pybind()

    for ite in range(n_ite):
        # Get position for this iteration
        pos_ite = Vector2f(pos_x[ite*sample_per_pass:ite*sample_per_pass+sample_per_pass],
                           pos_y[ite*sample_per_pass:ite*sample_per_pass+sample_per_pass])

        # Get rays for path tracing
        rays, weights = sensor.sample_ray_differential(
            time=0,
            sample1=sampler.next_1d(),
            sample2=pos_ite * scale,
            sample3=0
        )

        result, valid_rays = render_sample(scene, sampler, rays, bdata, heightmap_pybind, bssrdf)
        result = weights * result
        xyz = Color3f(srgb_to_xyz(result))
        aovs = [xyz[0], xyz[1], xyz[2],
                ek.select(valid_rays, Float(1.0), Float(0.0)),
                1.0]

        block.put(pos_ite, aovs)
        sampler.advance()
        cnt += sample_per_pass
        print(f"done {cnt} / {total_sample_count}")

    xyzaw_np = np.array(block.data()).reshape([film_size[1], film_size[0], 5])

    bmp = Bitmap(xyzaw_np, Bitmap.PixelFormat.XYZAW)
    bmp = bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
    bmp.write('result.exr')
    bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True).write('result.jpg')



def render_sample(scene, sampler, rays, bdata, heightmap_pybind, bssrdf=None):
    """
    Sample RTE
    TODO: Support multi channel sampling

    Args:
        scene: Target scene object
        sampler: Sampler object for random number
        rays: Given rays for sampling

    Returns:
        result: Sampling RTE result
    """
    eta = Float(1.0)
    emission_weight = Float(1.0)
    throughput = Spectrum(1.0)
    result = Spectrum(0.0)
    scatter = Spectrum(0.0)
    non_scatter = Spectrum(0.0)
    active = True
    is_bssrdf = False

    ##### First interaction #####
    si = scene.ray_intersect(rays, active)
    active = si.is_valid() & active
    valid_rays = si.is_valid()

    emitter = si.emitter(scene, active)

    depth = 0

    # Set channel
    # At and after evaluating BSSRDF, a ray consider only this one channel
    n_channels = 3
    channel = UInt32(ek.min(sampler.next_1d(active) * n_channels, n_channels - 1))

    d_out_local = Vector3f().zero()
    d_out_pdf = Float(0)

    while(True):
        depth += 1

        ##### Interaction with emitters #####
        emission_val = emission_weight * throughput * Emitter.eval_vec(emitter, si, active)
        
        result += ek.select(active, emission_val, Spectrum(0.0))
        scatter += ek.select(active & is_bssrdf, emission_val, Spectrum(0.0))
        non_scatter += ek.select(active & ~is_bssrdf, emission_val, Spectrum(0.0))

        active = active & si.is_valid()

        # Process russian roulette
        if depth > config.rr_depth:
            q = ek.min(ek.hmax(throughput) * ek.sqr(eta), 0.95)
            active = active & (sampler.next_1d(active) < q)
            throughput *= ek.rcp(q)

        # Stop if the number of bouces exceeds the given limit bounce, or
        # all rays are invalid. latter check is done only when the limit
        # bounce is infinite
        if depth >= config.max_depth:
            break

        ##### Emitter sampling #####
        bsdf = si.bsdf(rays)
        ctx = BSDFContext()

        active_e = active & has_flag(BSDF.flags_vec(bsdf), BSDFFlags.Smooth)
        ds, emitter_val = scene.sample_emitter_direction(si, sampler.next_2d(active_e),
                                                         True, active_e)
        active_e &= ek.neq(ds.pdf, 0.0)

        # Query the BSDF for that emitter-sampled direction
        wo = si.to_local(ds.d)
        bsdf_val = BSDF.eval_vec(bsdf, ctx, si, wo, active_e)
        # Determine density of sampling that same direction using BSDF sampling
        bsdf_pdf = BSDF.pdf_vec(bsdf, ctx, si, wo, active_e)

        mis = ek.select(ds.delta, Float(1), mis_weight(ds.pdf, bsdf_pdf))

        emission_val = mis * throughput * bsdf_val * emitter_val

        result += ek.select(active, emission_val, Spectrum(0.0))
        scatter += ek.select(active & is_bssrdf, emission_val, Spectrum(0.0))
        non_scatter += ek.select(active & ~is_bssrdf, emission_val, Spectrum(0.0))

        ##### BSDF sampling #####
        bs, bsdf_val = BSDF.sample_vec(bsdf, ctx, si, sampler.next_1d(active),
                                       sampler.next_2d(active), active)
        
        ##### BSSRDF replacing #####
        if(config.enable_bssrdf):
            # Replace bsdf samples by ones of BSSRDF
            bs.wo = ek.select(is_bssrdf, d_out_local, bs.wo)
            bs.pdf = ek.select(is_bssrdf, d_out_pdf, bs.pdf)
            bs.sampled_component = ek.select(is_bssrdf, UInt32(1), bs.sampled_component)
            bs.sampled_type = ek.select(is_bssrdf, UInt32(+BSDFFlags.DeltaTransmission), bs.sampled_type)
        ############################
        
        throughput *= ek.select(is_bssrdf, Float(1.0), bsdf_val)
        active &= ek.any(ek.neq(throughput, 0))

        eta *= bs.eta

        # Intersect the BSDF ray against the scene geometry
        rays = RayDifferential3f(si.spawn_ray(si.to_world(bs.wo)))
        si_bsdf = scene.ray_intersect(rays, active)


        ##### Checking BSSRDF #####
        if(config.enable_bssrdf):
            # Whether the BSDF is BSS   RDF or not?
            is_bssrdf = (active & has_flag(BSDF.flags_vec(bsdf), BSDFFlags.BSSRDF)
                            & (Frame3f.cos_theta(bs.wo) < Float(0.0))
                            & (Frame3f.cos_theta(si.wi) > Float(0.0)))

            # Decide whether we should use 0-scattering or multiple scattering
            is_zero_scatter = utils_render.check_zero_scatter(sampler, si_bsdf, bs, channel, is_bssrdf)
            is_bssrdf = is_bssrdf & ~is_zero_scatter

            throughput *= ek.select(is_bssrdf, ek.sqr(bs.eta), Float(1.0))
        ###########################

        ###### Process for BSSRDF ######
        if(config.enable_bssrdf and not ek.none(is_bssrdf)):
            # Get projected samples from BSSRDF
            projected_si, project_suc, abs_prob = bssrdf.sample_bssrdf(scene, bsdf, bs, si, bdata, 
                                                                       heightmap_pybind, channel, is_bssrdf)

            if config.visualize_invalid_sample:
                active = active & (~is_bssrdf | project_suc)
                result[(is_bssrdf & (~project_suc))] += Spectrum([100, 0, 0])

            # Sample outgoing direction from projected position
            d_out_local, d_out_pdf = utils_render.resample_wo(sampler, is_bssrdf)
            # Apply absorption probability
            throughput *= ek.select(is_bssrdf, Spectrum(1) - abs_prob, Spectrum(1))
            # Replace interactions by sampled ones from BSSRDF
            si_bsdf = SurfaceInteraction3f().masked_si(si_bsdf, projected_si, is_bssrdf)
        ################################
        

        # Determine probability of having sampled that same
        # direction using emitter sampling
        emitter = si_bsdf.emitter(scene, active)
        ds = DirectionSample3f(si_bsdf, si)
        ds.object = emitter

        delta = has_flag(bs.sampled_type, BSDFFlags.Delta)
        emitter_pdf = ek.select(delta, Float(0.0),
                                scene.pdf_emitter_direction(si, ds))
        emission_weight = mis_weight(bs.pdf, emitter_pdf)

        si = si_bsdf

    return result, valid_rays, scatter, non_scatter
