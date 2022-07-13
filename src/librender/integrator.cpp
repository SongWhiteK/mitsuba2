#include <thread>
#include <mutex>

#include <enoki/morton.h>
#include <enoki/stl.h>
#include <fstream>
#include <iostream>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/spiral.h>
#include <mutex>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

NAMESPACE_BEGIN(mitsuba)

// -----------------------------------------------------------------------------

MTS_VARIANT SamplingIntegrator<Float, Spectrum>::SamplingIntegrator(const Properties &props)
    : Base(props) {

    m_block_size = (uint32_t) props.size_("block_size", 0);
    uint32_t block_size = math::round_to_power_of_two(m_block_size);
    if (m_block_size > 0 && block_size != m_block_size) {
        Log(Warn, "Setting block size from %i to next higher power of two: %i", m_block_size,
            block_size);
        m_block_size = block_size;
    }

    m_samples_per_pass = (uint32_t) props.size_("samples_per_pass", (size_t) -1);
    m_timeout = props.float_("timeout", -1.f);

    /// Disable direct visibility of emitters if needed
    m_hide_emitters = props.bool_("hide_emitters", false);
}

MTS_VARIANT SamplingIntegrator<Float, Spectrum>::~SamplingIntegrator() { }

MTS_VARIANT void SamplingIntegrator<Float, Spectrum>::cancel() {
    m_stop = true;
}

MTS_VARIANT std::vector<std::string> SamplingIntegrator<Float, Spectrum>::aov_names() const {
    return { };
}

MTS_VARIANT bool SamplingIntegrator<Float, Spectrum>::render(Scene *scene, Sensor *sensor) {
    ScopedPhase sp(ProfilerPhase::Render);
    m_stop = false;

    ref<Film> film = sensor->film();
    ScalarVector2i film_size = film->crop_size();

    size_t total_spp        = sensor->sampler()->sample_count();
    size_t samples_per_pass = (m_samples_per_pass == (size_t) -1)
                               ? total_spp : std::min((size_t) m_samples_per_pass, total_spp);
    if ((total_spp % samples_per_pass) != 0)
        Throw("sample_count (%d) must be a multiple of samples_per_pass (%d).",
              total_spp, samples_per_pass);

    size_t n_passes = (total_spp + samples_per_pass - 1) / samples_per_pass;

    std::vector<std::string> channels = aov_names();
    bool has_aovs = !channels.empty();

    // Insert default channels and set up the film
    for (size_t i = 0; i < 5; ++i)
        channels.insert(channels.begin() + i, std::string(1, "XYZAW"[i]));
    film->prepare(channels);

    m_render_timer.reset();
    if constexpr (!is_cuda_array_v<Float>) {
        /// Render on the CPU using a spiral pattern
        size_t n_threads = __global_thread_count;
        Log(Info, "Starting render job (%ix%i, %i sample%s,%s %i thread%s)",
            film_size.x(), film_size.y(),
            total_spp, total_spp == 1 ? "" : "s",
            n_passes > 1 ? tfm::format(" %d passes,", n_passes) : "",
            n_threads, n_threads == 1 ? "" : "s");

        if (m_timeout > 0.f)
            Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

        // Find a good block size to use for splitting up the total workload.
        if (m_block_size == 0) {
            uint32_t block_size = MTS_BLOCK_SIZE;
            while (true) {
                if (block_size == 1 || hprod((film_size + block_size - 1) / block_size) >= n_threads)
                    break;
                block_size /= 2;
            }
            m_block_size = block_size;
        }

        Spiral spiral(film, m_block_size, n_passes);

        ThreadEnvironment env;
        ref<ProgressReporter> progress = new ProgressReporter("Rendering");
        std::mutex mutex;

        // Total number of blocks to be handled, including multiple passes.
        size_t total_blocks = spiral.block_count() * n_passes,
               blocks_done = 0;

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, total_blocks, 1),
            [&](const tbb::blocked_range<size_t> &range) {
                ScopedSetThreadEnvironment set_env(env);
                ref<Sampler> sampler = sensor->sampler()->clone();
                ref<ImageBlock> block = new ImageBlock(m_block_size, channels.size(),
                                                       film->reconstruction_filter(),
                                                       !has_aovs);
                scoped_flush_denormals flush_denormals(true);
                std::unique_ptr<Float[]> aovs(new Float[channels.size()]);

                // For each block
                for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {
                    auto [offset, size, block_id] = spiral.next_block();
                    Assert(hprod(size) != 0);
                    block->set_size(size);
                    block->set_offset(offset);

                    render_block(scene, sensor, sampler, block,
                                 aovs.get(), samples_per_pass, block_id);

                    film->put(block);

                    /* Critical section: update progress bar */ {
                        std::lock_guard<std::mutex> lock(mutex);
                        blocks_done++;
                        progress->update(blocks_done / (ScalarFloat) total_blocks);
                    }
                }
            }
        );
    } else {
        Log(Info, "Start rendering...");

        ref<Sampler> sampler = sensor->sampler();
        sampler->set_samples_per_wavefront((uint32_t) samples_per_pass);

        ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());
        ScalarUInt32 wavefront_size = hprod(film_size) * (uint32_t) samples_per_pass;
        if (sampler->wavefront_size() != wavefront_size)
            sampler->seed(0, wavefront_size);

        UInt32 idx = arange<UInt32>(wavefront_size);
        if (samples_per_pass != 1)
            idx /= (uint32_t) samples_per_pass;

        ref<ImageBlock> block = new ImageBlock(film_size, channels.size(),
                                               film->reconstruction_filter(),
                                               !has_aovs);
        block->clear();
        block->set_offset(sensor->film()->crop_offset());

        Vector2f pos = Vector2f(Float(idx % uint32_t(film_size[0])),
                                Float(idx / uint32_t(film_size[0])));
        pos += block->offset();

        std::vector<Float> aovs(channels.size());

        for (size_t i = 0; i < n_passes; i++)
            render_sample(scene, sensor, sampler, block, aovs.data(),
                          pos, diff_scale_factor);

        film->put(block);
    }

    if (!m_stop)
        Log(Info, "Rendering finished. (took %s)",
            util::time_string(m_render_timer.value(), true));

    return !m_stop;
}

MTS_VARIANT void SamplingIntegrator<Float, Spectrum>::render_block(const Scene *scene,
                                                                   const Sensor *sensor,
                                                                   Sampler *sampler,
                                                                   ImageBlock *block,
                                                                   Float *aovs,
                                                                   size_t sample_count_,
                                                                   size_t block_id) const {
    block->clear();
    uint32_t pixel_count  = (uint32_t)(m_block_size * m_block_size),
             sample_count = (uint32_t)(sample_count_ == (size_t) -1
                                           ? sampler->sample_count()
                                           : sample_count_);

    ScalarFloat diff_scale_factor = rsqrt((ScalarFloat) sampler->sample_count());

    if constexpr (!is_array_v<Float>) {
        for (uint32_t i = 0; i < pixel_count && !should_stop(); ++i) {
            sampler->seed(block_id * pixel_count + i);

            ScalarPoint2u pos = enoki::morton_decode<ScalarPoint2u>(i);
            if (any(pos >= block->size()))
                continue;

            pos += block->offset();
            for (uint32_t j = 0; j < sample_count && !should_stop(); ++j) {
                render_sample(scene, sensor, sampler, block, aovs,
                              pos, diff_scale_factor);
            }
        }
    } else if constexpr (is_array_v<Float> && !is_cuda_array_v<Float>) {
        // Ensure that the sample generation is fully deterministic
        sampler->seed(block_id);

        for (auto [index, active] : range<UInt32>(pixel_count * sample_count)) {
            if (should_stop())
                break;
            Point2u pos = enoki::morton_decode<Point2u>(index / UInt32(sample_count));
            active &= !any(pos >= block->size());
            pos += block->offset();
            render_sample(scene, sensor, sampler, block, aovs, pos, diff_scale_factor, active);
        }
    } else {
        ENOKI_MARK_USED(scene);
        ENOKI_MARK_USED(sensor);
        ENOKI_MARK_USED(aovs);
        ENOKI_MARK_USED(diff_scale_factor);
        ENOKI_MARK_USED(pixel_count);
        ENOKI_MARK_USED(sample_count);
        Throw("Not implemented for CUDA arrays.");
    }
}

MTS_VARIANT void
SamplingIntegrator<Float, Spectrum>::render_sample(const Scene *scene,
                                                   const Sensor *sensor,
                                                   Sampler *sampler,
                                                   ImageBlock *block,
                                                   Float *aovs,
                                                   const Vector2f &pos,
                                                   ScalarFloat diff_scale_factor,
                                                   Mask active) const {
    Vector2f position_sample = pos + sampler->next_2d(active);

    Point2f aperture_sample(.5f);
    if (sensor->needs_aperture_sample())
        aperture_sample = sampler->next_2d(active);

    Float time = sensor->shutter_open();
    if (sensor->shutter_open_time() > 0.f)
        time += sampler->next_1d(active) * sensor->shutter_open_time();

    Float wavelength_sample = sampler->next_1d(active);

    Vector2f adjusted_position =
        (position_sample - sensor->film()->crop_offset()) /
        sensor->film()->crop_size();

    auto [ray, ray_weight] = sensor->sample_ray_differential(
        time, wavelength_sample, adjusted_position, aperture_sample);

    ray.scale_differential(diff_scale_factor);

    const Medium *medium = sensor->medium();
    std::pair<Spectrum, Mask> result = sample(scene, sampler, ray, medium, aovs + 5, active);
    result.first = ray_weight * result.first;

    UnpolarizedSpectrum spec_u = depolarize(result.first);

    Color3f xyz;
    if constexpr (is_monochromatic_v<Spectrum>) {
        xyz = spec_u.x();
    } else if constexpr (is_rgb_v<Spectrum>) {
        xyz = srgb_to_xyz(spec_u, active);
    } else {
        static_assert(is_spectral_v<Spectrum>);
        xyz = spectrum_to_xyz(spec_u, ray.wavelengths, active);
    }

    aovs[0] = xyz.x();
    aovs[1] = xyz.y();
    aovs[2] = xyz.z();
    aovs[3] = select(result.second, Float(1.f), Float(0.f));
    aovs[4] = 1.f;

    block->put(position_sample, aovs, active);

    sampler->advance();
}

MTS_VARIANT std::pair<Spectrum, typename SamplingIntegrator<Float, Spectrum>::Mask>
SamplingIntegrator<Float, Spectrum>::sample(const Scene * /* scene */,
                                            Sampler * /* sampler */,
                                            const RayDifferential3f & /* ray */,
                                            const Medium * /* medium */,
                                            Float * /* aovs */,
                                            Mask /* active */) const {
    NotImplementedError("sample");
}

// -----------------------------------------------------------------------------

MTS_VARIANT MonteCarloIntegrator<Float, Spectrum>::MonteCarloIntegrator(const Properties &props)
    : Base(props) {
    /// Depth to begin using russian roulette
    m_rr_depth = props.int_("rr_depth", 5);
    if (m_rr_depth <= 0)
        Throw("\"rr_depth\" must be set to a value greater than zero!");

    /*  Longest visualized path depth (``-1 = infinite``). A value of \c 1 will
        visualize only directly visible light sources. \c 2 will lead to
        single-bounce (direct-only) illumination, and so on. */
    m_max_depth = props.int_("max_depth", -1);
    if (m_max_depth < 0 && m_max_depth != -1)
        Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");
}

MTS_VARIANT MonteCarloIntegrator<Float, Spectrum>::~MonteCarloIntegrator() { }

// -----------------------------------------------------------------------------

MTS_VARIANT PathSampler<Float, Spectrum>::PathSampler(const Properties &props)
    : Base(props) {

    m_timeout = props.float_("timeout", -1.f);
    m_samples_per_pass = (uint32_t) props.size_("samples_per_pass", (size_t) -1);

    m_rr_depth = props.int_("rr_depth", 5);
    if (m_rr_depth <= 0)
        Throw("\"rr_depth\" must be set to a value greater than zero!");

    m_max_depth = props.int_("max_depth", -1);
    if (m_max_depth < 0 && m_max_depth != -1)
        Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");

    m_output_path = props.string("output_path", ".\\");

    m_spp_roop = props.bool_("spp_roop", true);
    m_thread_roop = props.bool_("thread_roop", false);
    if(m_thread_roop) m_spp_roop = false;

    m_random_sample = props.bool_("random_sample", true);

    m_constant_sample = props.bool_("constant_sample", false);

    m_init_d = props.vector3f("init_d", ScalarVector3f(0, 0, 1));

    m_init_p = props.vector3f("init_p", ScalarVector3f(0, 0, 1));

    m_size_train_data_batch = props.int_("data_batch", 1);

    m_coeff_sigman = props.int_("coeff_sigman", -1);
}


MTS_VARIANT PathSampler<Float, Spectrum>::~PathSampler() { }

MTS_VARIANT void PathSampler<Float, Spectrum>::cancel() {
    m_stop = true;
}

MTS_VARIANT bool PathSampler<Float, Spectrum>::render(Scene *scene, Sensor *sensor) {
    ScopedPhase sp(ProfilerPhase::Render);
    m_stop = false;

    if constexpr (!is_cuda_array_v<Float>) {
        size_t total_spp = sensor->sampler()->sample_count();
        size_t samples_per_pass = (m_samples_per_pass == (size_t) -1)
                                ? total_spp : std::min((size_t) m_samples_per_pass, total_spp);
        if ((total_spp % samples_per_pass) != 0)
            Throw("sample_count (%d) must be a multiple of samples_per_pass (%d).",
                total_spp, samples_per_pass);

        size_t n_passes = (total_spp + samples_per_pass - 1) / samples_per_pass;

        

        size_t n_threads = __global_thread_count;
        Log(Info, "Starting path sampling job (%i samples, %s %i threads)",
            total_spp,
            n_passes > 1 ? tfm::format(" %d passes,", n_passes) : "",
            n_threads);

        if (m_timeout > 0.f)
                Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

        ThreadEnvironment env;
        ref<ProgressReporter> progress = new ProgressReporter("Rendering");
        std::mutex mutex;

        size_t it_done = 0;

        // Generate initial ray for tracing
        ref<Sampler> sampler_ray = sensor->sampler()->clone();
        sampler_ray->seed(0);
        std::pair<RayDifferential3f, MediumPtr> sample_result = sample_path(scene, sampler_ray);
        RayDifferential3f ray = sample_result.first;
        MediumPtr medium_sample = sample_result.second;
        Float sigman = get_sigma_n(medium_sample);

        Mask active = true;
        const Medium *medium = sensor->medium();

        m_render_timer.reset();

        // Setup 
        size_t n_sample_thread = (total_spp >= n_threads) ? total_spp / n_threads : total_spp;
        size_t size_it_batch = std::min((size_t)32, total_spp);
        size_t size_train_data_batch = m_size_train_data_batch;
        int coeff_sigman = m_coeff_sigman;
        int seed_add = 0;

        TrainingSample s;
        std::vector<TrainingSample> TrainingSamples;

        size_t n_valid = 0, n_absorbed = 0, n_invalid = 0, n_reflect = 0;

        if(m_spp_roop){
            // tracing the same ray "spp" times with spp roops
            // continue to sample until getting enough data
            while(TrainingSamples.size() < size_train_data_batch || n_valid < total_spp){
                seed_add += size_it_batch;

                // if too many samples are invalid, resample
                if(n_invalid > 2 * total_spp){
                    Log(Info, "Too many invalid samples, Resampleing");
                    Log(Info,
                        "Result----> valid: %i, absorbed %i, invalid: %i, reflect: "
                        "%i",
                        n_valid, n_absorbed, n_invalid, n_reflect);
                    it_done = 0;
                    n_valid = 0;
                    n_absorbed = 0;
                    n_invalid = 0;
                    n_reflect = 0;
                    TrainingSamples.clear();

                    sampler_ray->advance();
                    sample_result = sample_path(scene, sampler_ray);
                    ray = sample_result.first;
                    medium_sample = sample_result.second;
                }
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, size_it_batch, 1),
                    [&](const tbb::blocked_range<size_t> &range) {
                        ScopedSetThreadEnvironment set_env(env);
                        ref<Sampler> sampler = sensor->sampler()->clone();
                        scoped_flush_denormals flush_denormals(true);

                        // For each path
                        for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {

                            // Ensure that the sample generation is deterministic?
                            sampler->seed(i + seed_add);
                            PathSampleResult r = sample(scene, sampler, ray, medium); // sample path

                            /* Critical section: update progress bar */ {
                                std::lock_guard<std::mutex> lock(mutex);

                                // Process result data
                                if(TrainingSamples.size() < size_train_data_batch || n_valid < total_spp){
                                    switch (r.status)
                                    {
                                    case PathSampleResult::EStatus::EValid:
                                        n_valid++;
                                        if((sigman * coeff_sigman > enoki::norm(r.p_out - r.p_in)) || coeff_sigman < 0){
                                            if(TrainingSamples.size() < size_train_data_batch && r.n_out[2] >= 0){
                                                s.p_in  = r.p_in;
                                                s.d_in  = r.d_in;
                                                s.p_out = r.p_out;
                                                s.d_out = r.d_out;
                                                s.n_in  = r.n_in;
                                                s.n_out = r.n_out;
                                                s.eta   = r.eta;
                                                s.throughput = r.throughput;
                                                TrainingSamples.push_back(s);
                                            }
                                        }
                                        it_done++;
                                        break;
                                    case PathSampleResult::EStatus::EAbsorbed:
                                        n_valid++;
                                        if(it_done < total_spp) n_absorbed++;
                                        it_done++;
                                        break;
                                    case PathSampleResult::EStatus::EInvalid:
                                        n_invalid++;
                                        it_done++;
                                        break;
                                    case PathSampleResult::EStatus::EReflect:
                                        n_invalid++;
                                        n_reflect++;
                                        break;
                                    }
                                }

                            }
                        }
                    }
                );
            }

            // Calculate absorption probability and contain it
            s.abs_prob = (Float) n_absorbed / (Float) total_spp;
            for(int i = 0; i < TrainingSamples.size(); i++){
                TrainingSamples[i].abs_prob = s.abs_prob;
            }

        }else if(m_thread_roop){
            size_t total_it = (total_spp >= n_threads) ? n_threads : 1;
            // tracing the same ray "spp" times with 8 roops
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, total_it, 1),
                [&](const tbb::blocked_range<size_t> &range) {
                    ScopedSetThreadEnvironment set_env(env);
                    ref<Sampler> sampler = sensor->sampler()->clone();
                    scoped_flush_denormals flush_denormals(true);

                    // For each path
                    for (auto i = range.begin(); i != range.end() && !should_stop(); ++i) {

                        // Ensure that the sample generation is deterministic?
                        sampler->seed(i);
                        sample_thread(scene, sampler, ray, medium, n_sample_thread);

                        /* Critical section: update progress bar */ {
                            std::lock_guard<std::mutex> lock(mutex);
                            it_done++;
                            progress->update(it_done / (ScalarFloat) total_it);
                        }
                    }
                }
            );
        }

        if (!m_stop){
            Log(Info, "Rendering finished. (took %s)",
                util::time_string(m_render_timer.value(), true));
            Log(Info, "Result----> valid: %i, absorbed %i, invalid: %i, reflect: %i",
                n_valid, n_absorbed, n_invalid, n_reflect);

            result_to_csv(TrainingSamples, medium_sample, sigman);
        }
    }
    return !m_stop;
}

MTS_VARIANT void PathSampler<Float, Spectrum>::result_to_csv(const std::vector<TrainingSample> TrainingSamples, const Medium *medium, const Float sigman) const{
    // Setup csv file stream
    std:: string filename = m_output_path;
    bool init = false;

    // get medium parameters
    MediumInteraction3f mi = zero<MediumInteraction3f>();
    std::tuple<Spectrum, Spectrum, Spectrum> m_coef = medium->get_scattering_coefficients(mi);
    Spectrum& sigmas = std::get<0>(m_coef);
    Spectrum& sigmat = std::get<2>(m_coef);
    Spectrum albedo = sigmas / sigmat;

    auto phase = medium->phase_function();
    Float g = phase->get_param();
    
    


    // check a csv-file has already been generated
    {
        std::ifstream ifs;
        ifs.open(filename, std::ios::in);

        if(!ifs){
            init = true;
        }
    }

    std::ofstream ofs(filename, std::ios::app);
    // set dics
    if(init){
        ofs << "sigma_t" << "," << "albedo" << "," << "g" << "," << "eta" << ","
            << "p_in_x" << "," << "p_in_y" << ","<< "p_in_z"<< ","
            << "p_out_x"<< ","<< "p_out_y"<< ","<< "p_out_z"<< ","
            << "d_in_x"<< ","<< "d_in_y"<< ","<< "d_in_z"<< ","
            << "d_out_x"<< ","<< "d_out_y"<< ","<< "d_out_z"<< ","
            << "n_in_x"<< ","<< "n_in_y"<< ","<< "n_in_z"<< ","
            << "n_out_x"<< ","<< "n_out_y"<< ","<< "n_out_z"<< ","
            << "throughput" << "," << "abs_prob" << std::endl;
    }

    for (int i = 0; i < TrainingSamples.size(); i++){
        TrainingSample s = TrainingSamples[i];
        ofs << sigmat[0] << "," << albedo[0] << "," << g << "," << s.eta << ","
            << s.p_in[0] << "," << s.p_in[1] << "," << s.p_in[2] << ","
            << s.p_out[0] << "," << s.p_out[1] << "," << s.p_out[2] << ","
            << s.d_in[0] << "," << s.d_in[1] << "," << s.d_in[2] << ","
            << s.d_out[0] << "," << s.d_out[1] << "," << s.d_out[2] << ","
            << s.n_in[0] << "," << s.n_in[1] << "," << s.n_in[2] << ","
            << s.n_out[0] << "," << s.n_out[1] << "," << s.n_out[2] << ","
            << s.throughput[0] << "," << s.abs_prob << std::endl;
    }
}

MTS_VARIANT Float
PathSampler<Float, Spectrum>::get_sigma_n(const Medium *medium) {
    // get medium parameters
    MediumInteraction3f mi = zero<MediumInteraction3f>();
    std::tuple<Spectrum, Spectrum, Spectrum> m_coef =
        medium->get_scattering_coefficients(mi);
    Spectrum &sigmas = std::get<0>(m_coef);
    Spectrum &sigmat = std::get<2>(m_coef);
    Spectrum albedo  = sigmas / sigmat;
    Spectrum sigmaa = sigmat - sigmas;

    auto phase = medium->phase_function();
    Float g    = phase->get_param();

    Float reduced_albedo = (1 - g) * sigmas[0] / ((1 - g) * sigmas[0] + sigmaa[0]);
    Float reduced_sigmat = (1 - g) * sigmas[0] + sigmaa[0];

    Float eff_albedo = -enoki::log(1 - reduced_albedo * (1 - std::exp(-8))) / 8;

    Float MAD = 0.25 * (g + reduced_albedo) + eff_albedo;

    return 2*MAD / reduced_sigmat;
}

MTS_VARIANT void PathSampler<Float, Spectrum>::sample_thread(const Scene *scene,
                                    Sampler *sampler,
                                    const RayDifferential3f &ray,
                                    const Medium *medium,
                                    size_t n_sample_thread) const {
    for(size_t i = 0; i < n_sample_thread; i++){
        PathSampleResult r = sample(scene, sampler, ray, medium);
    }

}

MTS_VARIANT std::pair <typename PathSampler<Float, Spectrum>::RayDifferential3f,typename PathSampler<Float, Spectrum>::MediumPtr>
PathSampler<Float, Spectrum>::sample_path(Scene * /* scene */,
                                          Sampler * /* sampler */) const {
    NotImplementedError("sample_path");
}

MTS_VARIANT typename PathSampler<Float, Spectrum>::PathSampleResult
PathSampler<Float, Spectrum>::sample(const Scene * /* scene */,
                                    Sampler * /* sampler */,
                                    const RayDifferential3f & /* ray */,
                                    const Medium * /* medium */) const {
    NotImplementedError("sample");
}

MTS_IMPLEMENT_CLASS_VARIANT(Integrator, Object, "integrator")
MTS_IMPLEMENT_CLASS_VARIANT(SamplingIntegrator, Integrator)
MTS_IMPLEMENT_CLASS_VARIANT(MonteCarloIntegrator, SamplingIntegrator)
MTS_IMPLEMENT_CLASS_VARIANT(PathSampler, Integrator)

MTS_INSTANTIATE_CLASS(Integrator)
MTS_INSTANTIATE_CLASS(SamplingIntegrator)
MTS_INSTANTIATE_CLASS(MonteCarloIntegrator)
MTS_INSTANTIATE_CLASS(PathSampler)
NAMESPACE_END(mitsuba)
