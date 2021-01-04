#pragma once

#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/shapegroup.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MTS_EXPORT_RENDER Scene : public Object {
public:
    MTS_IMPORT_TYPES(BSDF, BSDFSample3f, Emitter, EmitterPtr, Film, Sampler, Shape, ShapePtr,
                     ShapeGroup, Sensor, Integrator, Medium, MediumPtr)

    /// Instantiate a scene from a \ref Properties object
    Scene(const Properties &props);

    // =============================================================
    //! @{ \name Ray tracing
    // =============================================================

    /**
     * \brief Intersect a ray against all primitives stored in the scene
     * and return information about the resulting surface interaction
     *
     * \param ray
     *    A 3-dimensional ray data structure with minimum/maximum
     *    extent information, as well as a time value (which matters
     *    when the shapes are in motion)
     *
     * \return
     *    A detailed surface interaction record. Query its \ref
     *    <tt>is_valid()</tt> method to determine whether an
     *    intersection was actually found.
     */
    SurfaceInteraction3f ray_intersect(const Ray3f &ray,
                                       Mask active = true) const;

    SurfaceInteraction3f ray_intersect(const Ray3f &ray,
                                       HitComputeFlags flags,
                                       Mask active = true) const;

    PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray,
                                                        Mask active = true) const;

    /**
     * \brief Ray intersection using brute force search. Used in
     * unit tests to validate the kdtree-based ray tracer.
     *
     * \remark Not implemented by the Embree/OptiX backends
     */
    SurfaceInteraction3f ray_intersect_naive(const Ray3f &ray,
                                             Mask active = true) const;

    /**
     * \brief Intersect a ray against all primitives stored in the scene
     * and \a only determine whether or not there is an intersection.
     *
     * Testing for the mere presence of intersections (as in \ref
     * ray_intersect) is considerably faster than finding an actual
     * intersection, hence this function should be preferred when
     * detailed information is not needed.
     *
     * \param ray
     *    A 3-dimensional ray data structure with minimum/maximum
     *    extent information, as well as a time value (which matterns
     *    when the shapes are in motion)
     *
     * \return \c true if an intersection was found
     */
    Mask ray_test(const Ray3f &ray, Mask active = true) const;

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Sampling interface
    // =============================================================

    /**
     * \brief Direct illumination sampling routine
     *
     * Given an arbitrary reference point in the scene, this method samples a
     * direction from the reference point to towards an emitter.
     *
     * Ideally, the implementation should importance sample the product of the
     * emission profile and the geometry term between the reference point and
     * the position on the emitter.
     *
     * \param ref
     *    A reference point somewhere within the scene
     *
     * \param sample
     *    A uniformly distributed 2D vector
     *
     * \param test_visibility
     *    When set to \c true, a shadow ray will be cast to ensure that the
     *    sampled emitter position and the reference point are mutually visible.
     *
     * \return
     *    Radiance received along the sampled ray divided by the sample
     *    probability.
     */
    std::pair<DirectionSample3f, Spectrum>
    sample_emitter_direction(const Interaction3f &ref,
                             const Point2f &sample,
                             bool test_visibility = true,
                             Mask active = true) const;

    /**
     * \brief Evaluate the probability density of the  \ref
     * sample_emitter_direct() technique given an filled-in \ref
     * DirectionSample record.
     *
     * \param ref
     *    A reference point somewhere within the scene
     *
     * \param ds
     *    A direction sampling record, which specifies the query location.
     *
     * \return
     *    The solid angle density expressed of the sample
     */
    Float pdf_emitter_direction(const Interaction3f &ref,
                                const DirectionSample3f &ds,
                                Mask active = true) const;

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Accessors
    // =============================================================

    /// Return a bounding box surrounding the scene
    const ScalarBoundingBox3f &bbox() const { return m_bbox; }

    /// Return the list of sensors
    std::vector<ref<Sensor>> &sensors() { return m_sensors; }
    /// Return the list of sensors (const version)
    const std::vector<ref<Sensor>> &sensors() const { return m_sensors; }

    /// Return the list of emitters
    host_vector<ref<Emitter>, Float> &emitters() { return m_emitters; }
    /// Return the list of emitters (const version)
    const host_vector<ref<Emitter>, Float> &emitters() const { return m_emitters; }

    /// Return the environment emitter (if any)
    const Emitter *environment() const { return m_environment.get(); }

    /// Return the list of shapes
    std::vector<ref<Shape>> &shapes() { return m_shapes; }
    /// Return the list of shapes
    const std::vector<ref<Shape>> &shapes() const { return m_shapes; }

    /// Return the scene's integrator
    Integrator* integrator() { return m_integrator; }
    /// Return the scene's integrator
    const Integrator* integrator() const { return m_integrator; }

    //! @}
    // =============================================================

    /// Perform a custom traversal over the scene graph
    void traverse(TraversalCallback *callback) override;

    /// Update internal state following a parameter update
    void parameters_changed(const std::vector<std::string> &/*keys*/ = {}) override;

    /// Return whether any of the shape's parameters require gradient
    bool shapes_grad_enabled() const { return m_shapes_grad_enabled; };

    /// Return a human-readable string representation of the scene contents.
    virtual std::string to_string() const override;

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~Scene();

    /// Create the ray-intersection acceleration data structure
    void accel_init_cpu(const Properties &props);
    void accel_init_gpu(const Properties &props);

    /// Updates the ray-intersection acceleration data structure
    void accel_parameters_changed_gpu();

    /// Release the ray-intersection acceleration data structure
    void accel_release_cpu();
    void accel_release_gpu();

    /// Trace a ray and only return a preliminary intersection data structure
    MTS_INLINE PreliminaryIntersection3f ray_intersect_preliminary_cpu(const Ray3f &ray, Mask active) const;
    MTS_INLINE PreliminaryIntersection3f ray_intersect_preliminary_gpu(const Ray3f &ray, Mask active) const;

    /// Trace a ray
    MTS_INLINE SurfaceInteraction3f ray_intersect_cpu(const Ray3f &ray, HitComputeFlags flags, Mask active) const;
    MTS_INLINE SurfaceInteraction3f ray_intersect_gpu(const Ray3f &ray, HitComputeFlags flags, Mask active) const;
    MTS_INLINE SurfaceInteraction3f ray_intersect_naive_cpu(const Ray3f &ray, Mask active) const;

    /// Trace a shadow ray
    MTS_INLINE Mask ray_test_cpu(const Ray3f &ray, Mask active) const;
    MTS_INLINE Mask ray_test_gpu(const Ray3f &ray, Mask active) const;

    using ShapeKDTree = mitsuba::ShapeKDTree<Float, Spectrum>;

protected:
    /// Acceleration data structure (type depends on implementation)
    void *m_accel = nullptr;

    ScalarBoundingBox3f m_bbox;

    host_vector<ref<Emitter>, Float> m_emitters;
    std::vector<ref<Shape>> m_shapes;
    std::vector<ref<ShapeGroup>> m_shapegroups;
    std::vector<ref<Sensor>> m_sensors;
    std::vector<ref<Object>> m_children;
    ref<Integrator> m_integrator;
    ref<Emitter> m_environment;

    bool m_shapes_grad_enabled;
};

/// Dummy function which can be called to ensure that the librender shared library is loaded
extern MTS_EXPORT_RENDER void librender_nop();

// See records.h
template <typename Float, typename Spectrum>
void DirectionSample<Float, Spectrum>::set_query(const Ray3f &ray, const SurfaceInteraction3f &si) {
    p      = si.p;
    n      = si.sh_frame.n;
    uv     = si.uv;
    time   = si.time;
    object = static_cast<ObjectPtr>(si.shape->emitter());
    d      = ray.d;
    dist   = si.t;
}

// See interaction.h
template <typename Float, typename Spectrum>
typename SurfaceInteraction<Float, Spectrum>::EmitterPtr
SurfaceInteraction<Float, Spectrum>::emitter(const Scene *scene, Mask active) const {
    if constexpr (!is_array_v<ShapePtr>) {
        if (is_valid())
            return shape->emitter(active);
        else
            return scene->environment();
    } else {
        return select(is_valid(), shape->emitter(active), scene->environment());
    }
}

template <typename Float, typename Spectrum>
std::pair<typename SurfaceInteraction<Float, Spectrum>::SurfaceInteraction3f, typename SurfaceInteraction<Float, Spectrum>::Mask>
SurfaceInteraction<Float, Spectrum>::project_to_mesh_effnormal(const Scene *scene,
                                                            const Vector3f &sampled_pos,
                                                            const BSDFSample3f &bs,
                                                            const UInt32 &channel,
                                                            Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::SurfaceProjection, active);

    Vector3f dx = Transform4f::rotate(Vector3f(1,0,0), bs.x) *
                Transform4f::rotate(Vector3f(0,1,0), bs.y) *
                Transform4f::rotate(Vector3f(0,0,1), bs.z) * Vector3f(1, 0, 0);
    Vector3f dy = Transform4f::rotate(Vector3f(1,0,0), bs.x) *
                Transform4f::rotate(Vector3f(0,1,0), bs.y) *
                Transform4f::rotate(Vector3f(0,0,1), bs.z) * Vector3f(0, 1, 0);
    Vector3f dz = Transform4f::rotate(Vector3f(1,0,0), bs.x) *
                Transform4f::rotate(Vector3f(0,1,0), bs.y) *
                Transform4f::rotate(Vector3f(0,0,1), bs.z) * Vector3f(0, 0, 1);

    Float kernelEps = get_kernelEps(bs, channel);
    
    return mesh_projection(dx, dy, dz, scene, sampled_pos, kernelEps, active);
}

template <typename Float, typename Spectrum>
std::pair<typename SurfaceInteraction<Float, Spectrum>::SurfaceInteraction3f, typename SurfaceInteraction<Float, Spectrum>::Mask>
SurfaceInteraction<Float, Spectrum>::project_to_mesh_normal(const Scene *scene,
                                                            const Vector3f &sampled_pos,
                                                            const BSDFSample3f &bs,
                                                            const UInt32 &channel,
                                                            Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::SurfaceProjection, active);

    Vector3f dx = sh_frame.to_world(Vector3f(1, 0, 0));
    Vector3f dy = sh_frame.to_world(Vector3f(0, 1, 0));
    Vector3f dz = sh_frame.to_world(Vector3f(0, 0, 1));

    Float kernelEps = get_kernelEps(bs, channel);

    return mesh_projection(dx, dy, dz, scene, sampled_pos, kernelEps, active);
}

template <typename Float, typename Spectrum>
std::pair<typename SurfaceInteraction<Float, Spectrum>::SurfaceInteraction3f, typename SurfaceInteraction<Float, Spectrum>::Mask>
SurfaceInteraction<Float, Spectrum>::mesh_projection(const Vector3f &dx, const Vector3f &dy, const Vector3f &dz,
                const Scene *scene, const Vector3f &sampled_pos,
                const Float kernelEps, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::SurfaceProjection, active);

    SurfaceInteraction3f si_result = zero<SurfaceInteraction3f>();
    Mask si_found = false;
    Vector2f dists;
    dists[0] = 2 * kernelEps;
    dists[1] = math::Infinity<Float>;



    for (int i = 0; i < 2; i++){
        Float projdist_max = dists[i];
        Float point_dist = Float(-1.f);

        for (int j = 0; j < 3; j++){
            Vector3f d;
            switch (j)
            {
            case 0:
                d = dz;
                break;
            case 1:
                d = dy;
                break;
            case 2:
                d = dx;
                break;
            }

            Ray3f ray1 = zero<Ray3f>();
            ray1.o = sampled_pos;
            ray1.d = -d;
            ray1.mint = math::RayEpsilon<Float>;
            ray1.maxt = projdist_max;
            ray1.update();

            SurfaceInteraction3f si1 = scene->ray_intersect(ray1, active);
            if(any_or<true>(si_found)){
                Mask si1_valid = eq(shape, si1.shape) && (point_dist < 0 || point_dist > si1.t) && active;
                si_found |= si1_valid && active;
                masked(point_dist, si1_valid) = si1.t;
                masked(si_result, si1_valid) = si1;
            }

            Float maxT = select(si_found, si1.t, projdist_max);
            Ray3f ray2 = zero<Ray3f>();
            ray2.o = sampled_pos;
            ray2.d = d;
            ray2.mint = math::RayEpsilon<Float>;
            ray2.maxt = maxT;

            SurfaceInteraction3f si2 = scene->ray_intersect(ray2, active);
            if(any_or<true>(si2.is_valid())){
                Mask si2_valid = eq(shape, si2.shape) && (point_dist < 0 || point_dist > si2.t) && active;
                si_found |= si2_valid && active;
                masked(point_dist, si2_valid) = si2.t;
                masked(si_result, si2_valid)  = si2;
            }

        }
        if(none(active ^ si_found)){
            return {si_result, si_found};
        }
    }
    return {si_result, si_found};
}

MTS_EXTERN_CLASS_RENDER(Scene)
NAMESPACE_END(mitsuba)
