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

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class VolumetricPathSampler : public PathSampler<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(PathSampler, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr,
                     Medium, MediumPtr, PhaseFunctionContext)

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

    std::pair<RayDifferential3f, Mask>
    sample_path(Scene *scene, Sampler *sampler) const override {
        Log(Info, "sample path");

        RayDifferential3f ray = zero<RayDifferential3f>();
        Mask active = true;

        return { ray, active };
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