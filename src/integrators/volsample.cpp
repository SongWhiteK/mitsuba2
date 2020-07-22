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
    MTS_IMPORT_BASE(PathSampler, m_max_depth, m_rr_depth, m_output_dir)
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

    RayDifferential3f sample_path(Scene *scene, Sampler *sampler) const override {
        RayDifferential3f ray = zero<RayDifferential3f>();
        Mask active = true;
        BoundingBox3f bbox = zero<BoundingBox3f>();
        
        // Get shapes from the scene and its bounding box
        ref<Shape> target = scene->shapes()[0];
        bbox = target->bbox();

        // Set range for position sampling
        Vector2f sample_min = Vector2f(bbox.min[0], bbox.min[1]);
        Vector2f sample_range = Vector2f(bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1]);
        Log(Info, "sample from x: %f to %f, y: %f to %f",
            sample_min[0], sample_min[0] + sample_range[0],
            sample_min[1], sample_min[1] + sample_range[1]);


        // sample position and generate a ray for get intersection 
        Vector2f pos_sample = sample_min + sample_range * sampler->next_2d();

        Ray3f ray_sample_xy = zero<Ray3f>();
        Vector3f o = Vector3f(pos_sample[0], pos_sample[1], bbox.max[2] + 1);
        ray_sample_xy.o = o;
        ray_sample_xy.d = Vector3f(0, 0, -1);
        ray_sample_xy.mint = 0.f;
        ray_sample_xy.maxt = 10.f;

        SurfaceInteraction3f si_sample = scene->ray_intersect(ray_sample_xy);

        // Sample direction and convert to world coordinates
        Vector3f d_sample = warp:: square_to_uniform_hemisphere(sampler->next_2d());
        Vector3f d_sample_world = si_sample.to_world(d_sample);
        Vector3f d_sample_world_small = d_sample_world / 100000;

        Log(Info, " sampled position x: %f, y: %f, z: %f",
                    si_sample.p[0], si_sample.p[1], si_sample.p[2]);
        Log(Info, " sampled direction x: %f, y: %f, z: %f",
                    d_sample_world[0], d_sample_world[1], d_sample_world[2]);

        // Generate a ray for tracing
        ray.o = si_sample.p + d_sample_world_small;
        ray.d = -d_sample_world;
        ray.mint = 0;
        ray.maxt = 10;

        // Check the ray is valid
        // SurfaceInteraction3f si_test = scene->ray_intersect(ray);
        // std::cout << "is it valid?: " << si_test.is_valid() << std::endl;
        // std::cout << "test position: " << si_test.p << std::endl;
        // std::cout << "test direction: " << si_test.to_world(si_test.wi) << std::endl;
        

        return { ray };
    }


    void sample(const Scene *scene,
                Sampler *sampler,
                const RayDifferential3f &ray_,
                const Medium *initial_medium) const override {
        // Tracing input and output to csv file


        // setup csv file stream
        std::string filename_pos = m_output_dir + "\\output_path.csv";
        std::ofstream ofs(filename_pos, std::ios::app);
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
        Vector3f out_pos = zero<Vector3f>();
        Vector3f in_pos = zero<Vector3f>();
        Vector3f wi = zero<Vector3f>();

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
                specular_chain &= !act_medium_scatter;
                specular_chain |= act_medium_scatter && !sample_emitters;

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
                out_pos = si.p;
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

                masked(throughput, active_surface) *= bsdf_val;
                masked(eta, active_surface) *= bs.eta;

                Ray bsdf_ray                = si.spawn_ray(si.to_world(bs.wo));
                masked(ray, active_surface) = bsdf_ray;
                needs_intersection |= active_surface;

                Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                masked(depth, non_null_bsdf) += 1;

                specular_chain |= non_null_bsdf && has_flag(bs.sampled_type, BSDFFlags::Delta);
                specular_chain &= !(active_surface && has_flag(bs.sampled_type, BSDFFlags::Smooth));

                Mask add_emitter = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Delta) &&
                                   any(neq(depolarize(throughput), 0.f)) && (depth < (uint32_t) m_max_depth);
                act_null_scatter |= active_surface && has_flag(bs.sampled_type, BSDFFlags::Null);

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
                    Mask record_pos = has_medium_trans && (cos_theta < 0);

                    if(any_or<true>(record)){
                        if(any_or<true>(!record_pos)){
                            masked(record, !record_pos) = false;
                            // ofs << wi << "," << in_pos << "," << out_pos << std::endl;
                        }
                    }
                    else if(any_or<true>(record_pos)){
                        wi = si.to_world(si.wi);
                        in_pos = si.p;
                        masked(record, record_pos) = true;
                    }
                }

                masked(si, intersect2) = si_new;
            }
            active &= (active_surface | active_medium);

            // When the ray escapes from medium once, tracing ends.
            active = select(!record, false, true);
        }
    }



    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(VolumetricPathSampler, PathSampler);
MTS_EXPORT_PLUGIN(VolumetricPathSampler, "Volumetric Path sample integrator");
NAMESPACE_END(mitsuba)