#include <iostream>
#include <fstream>
#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/core/warp.h>


NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class VolumetricPathSampler : public PathSampler<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(PathSampler, m_max_depth, m_rr_depth, m_random_sample)
    MTS_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr,
                     Medium, MediumPtr, PhaseFunctionContext, Shape)

    VolumetricPathSampler(const Properties &props) : Base(props){
    }

    MTS_INLINE
    Float index_spectrum(const UnpolarizedSpectrum &spec, const UInt32 &idx) const {
        Float m = spec[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            masked(m, eq(idx, 1u)) = spec[1];
            masked(m, eq(idx, 2u)) = spec[2];
        } else {
            ENOKI_MARK_USED(idx);
        }
        return m;
    }

    // Sample position and direction on a object in the input scene
    // Generate a ray which through the sampled position and direction

    std::pair<RayDifferential3f, MediumPtr> sample_path(Scene *scene, Sampler *sampler) const override {
        RayDifferential3f ray = zero<RayDifferential3f>();


        BoundingBox3f bbox = zero<BoundingBox3f>();

        MediumPtr medium = nullptr;

        // Get shapes from the scene and its bounding box
        ref<Shape> target = scene->shapes()[0];
        bbox              = target->bbox();

        if (m_random_sample) {
            // Set range for position sampling
            Vector2f sample_min = Vector2f(bbox.min[0], bbox.min[1]);
            Vector2f sample_range =
                Vector2f(bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1]);
            Log(Info, "sample from x: %f to %f, y: %f to %f", sample_min[0],
                sample_min[0] + sample_range[0], sample_min[1],
                sample_min[1] + sample_range[1]);

            SurfaceInteraction3f si_sample;
            Ray3f ray_sample_xy = zero<Ray3f>();

            // sample position and generate a ray for get intersection
            while (true) {
                Vector2f pos_sample = sample_min + sample_range * sampler->next_2d();

                Vector3f o = Vector3f(pos_sample[0], pos_sample[1], bbox.max[2] + 1);
                ray_sample_xy.o = o;
                ray_sample_xy.d = Vector3f(0, 0, -1);
                ray_sample_xy.mint = math::RayEpsilon<Float>;
                ray_sample_xy.maxt = math::Infinity<Float>;
                ray_sample_xy.update();
                

                si_sample = scene->ray_intersect(ray_sample_xy);

                // Sample direction and convert to world coordinates
                Vector3f d_sample = warp:: square_to_cosine_hemisphere(sampler->next_2d());
                Vector3f d_sample_world = si_sample.to_world(d_sample);
                Vector3f d_sample_world_small = d_sample_world / 1000;

                Log(Info, " sampled position x: %f, y: %f, z: %f",
                            si_sample.p[0], si_sample.p[1], si_sample.p[2]);
                Log(Info, " sampled direction x: %f, y: %f, z: %f",
                            d_sample_world[0], d_sample_world[1], d_sample_world[2]);

                // Generate a ray for tracing
                ray.o = si_sample.p + d_sample_world_small;
                ray.d = -d_sample_world;
                ray.mint = math::RayEpsilon<Float>;
                ray.maxt = math::Infinity<Float>;
                ray.update();

                // Check the ray is valid. If somehow the ray intersects from inside, resampling
                SurfaceInteraction3f si_test = scene->ray_intersect(ray);

                if(any_or<true>(dot(si_test.n, si_test.to_world(si_test.wi)) < 0)){
                    Log(Info, "This sampled position and direction are invalid");
                }else{
                    break;
                }
            }

            // record the target medium
            medium = si_sample.target_medium(ray_sample_xy.d);
        } else {
            ray.o = Vector3f(0, 0, bbox.max[2]+1);
            ray.d    = Vector3f(0, 0, -1); 
            ray.mint = math::RayEpsilon<Float>;
            ray.maxt = math::Infinity<Float>;
            ray.update();

            SurfaceInteraction3f si = scene->ray_intersect(ray);
            medium = si.target_medium(ray.d);
        }

        return { ray, medium };
    }


    PathSampleResult sample(const Scene *scene,
                Sampler *sampler,
                const RayDifferential3f &ray_,
                const Medium *initial_medium) const override {

        PathSampleResult r;
        r.status = PathSampleResult::EStatus::EInvalid;

        Ray3f ray = ray_;

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        Spectrum throughput(1.f), result(0.f);
        MediumPtr medium = initial_medium;
        MediumInteraction3f mi = zero<MediumInteraction3f>();
        mi.t = math::Infinity<Float>;
        Mask active = true;
        Mask specular_chain = active;
        UInt32 depth = 0;

        UInt32 channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
            channel = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
        }

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t = math::Infinity<Float>;
        Mask needs_intersection = true;

        Mask record = false;
        Vector3f pos_out = zero<Vector3f>();
        Vector3f pos_in = zero<Vector3f>();
        Vector3f wi = zero<Vector3f>();
        Vector3f wo = zero<Vector3f>();
        Vector3f n_in = zero<Vector3f>();
        Vector3f n_out = zero<Vector3f>();
        

        for (int bounce = 0;; ++bounce) {
            // ----------------- Handle termination of paths ------------------

            // Russian roulette: try to keep path weights equal to one, while accounting for the
            // solid angle compression at refractive index boundaries. Stop with at least some
            // probability to avoid  getting stuck (e.g. due to total internal reflection)

            active &= any(neq(depolarize(throughput), 0.f));
            Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
            Mask perform_rr = (depth > (uint32_t) m_rr_depth);
            active &= sampler->next_1d(active) < q || !perform_rr;
            masked(throughput, perform_rr) *= rcp(detach(q));

            Mask exceeded_max_depth = depth >= (uint32_t) m_max_depth;
            if (none(active) || all(exceeded_max_depth)){
                if(any_or<true>(record)){
                    r.status = PathSampleResult::EStatus::EAbsorbed;
                }
                break;
            }

            // ----------------------- Sampling the RTE -----------------------
            Mask active_medium  = active && neq(medium, nullptr);
            Mask active_surface = active && !active_medium;
            Mask act_null_scatter = false, act_medium_scatter = false,
                 escaped_medium = false;

            // If the medium does not have a spectrally varying extinction,
            // we can perform a few optimizations to speed up rendering
            Mask is_spectral = active_medium;
            Mask not_spectral = false;
            if (any_or<true>(active_medium)) {
                is_spectral &= medium->has_spectral_extinction();
                not_spectral = !is_spectral && active_medium;
            }

            if (any_or<true>(active_medium)) {
                mi = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = mi.t;
                Mask intersect = needs_intersection && active_medium;
                if (any_or<true>(intersect))
                    masked(si, intersect) = scene->ray_intersect(ray, intersect);
                needs_intersection &= !active_medium;

                masked(mi.t, active_medium && (si.t < mi.t)) = math::Infinity<Float>;
                if (any_or<true>(is_spectral)) {
                    auto [tr, free_flight_pdf] = medium->eval_tr_and_pdf(mi, si, is_spectral);
                    Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                    masked(throughput, is_spectral) *= select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                }

                escaped_medium = active_medium && !mi.is_valid();
                active_medium &= mi.is_valid();

                // Handle null and real scatter events
                Mask null_scatter = sampler->next_1d(active_medium) >= index_spectrum(mi.sigma_t, channel) / index_spectrum(mi.combined_extinction, channel);

                act_null_scatter |= null_scatter && active_medium;
                act_medium_scatter |= !act_null_scatter && active_medium;

                if (any_or<true>(is_spectral && act_null_scatter))
                    masked(throughput, is_spectral && act_null_scatter) *=
                        mi.sigma_n * index_spectrum(mi.combined_extinction, channel) /
                        index_spectrum(mi.sigma_n, channel);

                masked(depth, act_medium_scatter) += 1;
            }

            // Dont estimate lighting if we exceeded number of bounces
            active &= depth < (uint32_t) m_max_depth;
            act_medium_scatter &= active;

            if (any_or<true>(act_null_scatter)) {
                masked(ray.o, act_null_scatter) = mi.p;
                masked(ray.mint, act_null_scatter) = 0.f;
                masked(si.t, act_null_scatter) = si.t - mi.t;
            }

            if (any_or<true>(act_medium_scatter)) {
                if (any_or<true>(is_spectral))
                    masked(throughput, is_spectral && act_medium_scatter) *=
                        mi.sigma_s * index_spectrum(mi.combined_extinction, channel) / index_spectrum(mi.sigma_t, channel);
                if (any_or<true>(not_spectral))
                    masked(throughput, not_spectral && act_medium_scatter) *= mi.sigma_s / mi.sigma_t;

                PhaseFunctionContext phase_ctx(sampler);
                auto phase = mi.medium->phase_function();

                // --------------------- Emitter sampling ---------------------
                Mask sample_emitters = mi.medium->use_emitter_sampling();

                // ------------------ Phase function sampling -----------------
                masked(phase, !act_medium_scatter) = nullptr;
                auto [wo, phase_pdf] = phase->sample(phase_ctx, mi, sampler->next_2d(act_medium_scatter), act_medium_scatter);
                Ray3f new_ray  = mi.spawn_ray(wo);
                new_ray.mint = 0.0f;
                masked(ray, act_medium_scatter) = new_ray;
                needs_intersection |= act_medium_scatter;
            }

            // if escape from medium, record the position
            if(any_or<true>(escaped_medium)){
                pos_out = si.p;
            }

            // --------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium;
            Mask intersect = active_surface && needs_intersection;
            if (any_or<true>(intersect))
                masked(si, intersect) = scene->ray_intersect(ray, intersect);

            active_surface &= si.is_valid();
            if (any_or<true>(active_surface)) {
                // --------------------- Emitter sampling ---------------------
                BSDFContext ctx;
                BSDFPtr bsdf  = si.bsdf(ray);

                // ----------------------- BSDF sampling ----------------------
                auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active_surface),
                                                   sampler->next_2d(active_surface), active_surface);
                bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

                masked(eta, active_surface) *= bs.eta;

                Ray bsdf_ray                = si.spawn_ray(si.to_world(bs.wo));
                masked(ray, active_surface) = bsdf_ray;
                needs_intersection |= active_surface;

                Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                masked(depth, non_null_bsdf) += 1;

                Mask add_emitter = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Delta) &&
                                   any(neq(depolarize(throughput), 0.f)) && (depth < (uint32_t) m_max_depth);

                // Intersect the indirect ray against the scene
                Mask intersect2 = active_surface && needs_intersection && add_emitter;
                SurfaceInteraction3f si_new = si;
                if (any_or<true>(intersect2))
                    masked(si_new, intersect2) = scene->ray_intersect(ray, intersect2);
                needs_intersection &= !intersect2;

                Mask has_medium_trans            = active_surface && si.is_medium_transition();
                masked(medium, has_medium_trans) = si.target_medium(ray.d);

                // If medium change (maybe when dive medium), record it's position
                if(any_or<true>(has_medium_trans)){
                    Float cos_theta = dot(ray.d, si.n);
                    Mask inner = has_medium_trans && (cos_theta < 0);

                    if(any_or<true>(record)){
                        if(any_or<true>(!inner)){
                            masked(record, !inner) = false;
                            r.status = PathSampleResult::EStatus::EValid;
                            wo = ray.d;
                            n_out = si.n;
                        }
                    } else if(any_or<true>(inner)){
                        wi = si.to_world(si.wi);
                        pos_in = si.p;
                        masked(record, inner) = true;
                        n_in = si.n;
                        r.eta = eta;
                    } else {
                        r.status = PathSampleResult::EStatus::EReflect;
                    }
                }

                masked(si, intersect2) = si_new;
            }
            active &= (active_surface | active_medium);

            // When the ray escapes from medium once, tracing ends.
            active = record;
        }
        if(r.status == PathSampleResult::EStatus::EValid){
            r.p_in = pos_in;
            r.d_in = wi;
            r.p_out = pos_out;
            r.d_out = wo;
            r.n_in = n_in;
            r.n_out = n_out;
            r.throughput = throughput;
        }
        return r;
    }



    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(VolumetricPathSampler, PathSampler);
MTS_EXPORT_PLUGIN(VolumetricPathSampler, "Volumetric Path sample integrator");
NAMESPACE_END(mitsuba)