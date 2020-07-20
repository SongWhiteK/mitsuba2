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
    MTS_IMPORT_BASE(PathSampler, m_max_depth, m_rr_depth)
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
        ray.maxt = 1;

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
                const Medium *initial_medium,
                Mask active) const override {
        Log(Info, "sample");
    };

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(VolumetricPathSampler, PathSampler);
MTS_EXPORT_PLUGIN(VolumetricPathSampler, "Volumetric Path sample integrator");
NAMESPACE_END(mitsuba)