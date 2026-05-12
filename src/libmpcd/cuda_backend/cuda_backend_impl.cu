#include <algorithm>
#include <string>

#include <mpcd/api/simulation_parameters.hpp>
#ifdef USE_HDF5
    #include "adapters/hdf5/h5cpp.hpp"
#endif
#include "common/probing.hpp"

#include "cuda_backend_impl.hpp"
#include "simulation_kernels.hpp"

namespace mpcd::cuda {

    CudaBackendImpl::CudaBackendImpl(SimulationParameters const& params)
        : ctx(params), strategy(makeCollisionStrategy(ctx)),
        sampling_state(SAMPLING_COMPLETED), sample_counter(0)
    {}

    CudaBackendImpl::~CudaBackendImpl() {
        cudaDeviceSynchronize();
    }

    /**
    *   @brief Data io, either creating snapshots or averaging and writing to disk.
    */
    void CudaBackendImpl::writeSample()
    {
    #ifndef USE_HDF5
        std::cerr << "Backend compiled without HDF5 support. No output possible." << std::endl;
        throw std::runtime_error("No HDF5 support");
    #endif

        ProbingType probe = what_to_do(ctx.step_counter, ctx.parameters);

        ctx.mpc_cells.set(0); // clear cells.
        sampling::addParticles<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.mpc_cells, ctx.particles);
        error_check("add_particles_reduce_only");

        sampling::averageCells<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.mpc_cells);
        error_check("average_cells_reduce_only");

        switch (probe){
            case ProbingType::snapshots_only:
                ctx.kahan_c.set(0);
                sampling::snapshot<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.mpc_cells, ctx.cell_states);
                error_check( "snapshot" );
                break;
            case ProbingType::start_accumulating:
                sampling::snapshot<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.mpc_cells, ctx.cell_states);
                error_check( "snapshot" );
                break;
            case ProbingType::accumulate:
                sampling::accumulate<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.mpc_cells, ctx.cell_states, ctx.kahan_c);
                error_check( "accumulate" );
                break;
            case ProbingType::finish_accumulation:
                sampling::finish<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.parameters.average_samples / ctx.parameters.sample_every,
                                                                                              ctx.mpc_cells, ctx.cell_states, ctx.kahan_c);
                error_check( "finish" );
                break;
        }

        if (probe == ProbingType::finish_accumulation or probe == ProbingType::snapshots_only) {
            ctx.cell_states.pull();
        #ifdef USE_HDF5
            mpcd::adapters::hdf5::file data( "simulation_data.h5", "a" ); // append
            data.write_float_data(std::string("fluid/") + std::to_string(ctx.step_counter),
                                  reinterpret_cast<float*>(ctx.cell_states.data()),
                                  {4, static_cast<size_t>(ctx.parameters.volume_size[0]),
                                      static_cast<size_t>(ctx.parameters.volume_size[1]),
                                      static_cast<size_t>(ctx.parameters.volume_size[2])});
        #endif
        }
    }

    void CudaBackendImpl::writeBackupFile() {
        // TODO
    }

    /**
    *   @brief Perform SRD streaming step
    */
    void CudaBackendImpl::translationStep() {
        ctx.grid_shift = {ctx.random.genUniformFloat() - mpcd::Float(0.5),
                          ctx.random.genUniformFloat() - mpcd::Float(0.5),
                          ctx.random.genUniformFloat() - mpcd::Float(0.5)};

        ctx.uniform_counter.set(0);
        translateParticles<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(
            ctx.particles, ctx.mpc_cells, ctx.generator, ctx.grid_shift,
            {ctx.parameters.volume_size[0], ctx.parameters.volume_size[1], ctx.parameters.volume_size[2]},
            {ctx.parameters.periodicity[0], ctx.parameters.periodicity[1], ctx.parameters.periodicity[2]},
            ctx.parameters.delta_t, ctx.parameters.experiment,
            ctx.uniform_counter, ctx.uniform_list);
        error_check( "translate_particles" );
    }

    /**
    *   @brief Perform SRD collision step — dispatched via CollisionStrategy
    */
    void CudaBackendImpl::collisionStep() {
        ctx.mpc_cells.set(0);
        strategy->collideParticles();
    }

    void CudaBackendImpl::sortParticles() {
        strategy->sortParticles();
    }

    void CudaBackendImpl::step(int n_steps) {
        for (int i = 0; i < n_steps; i++) {
            translationStep();
            collisionStep();

            if ((ctx.step_counter++ % ctx.resort_rate) == 0)
                sortParticles();
        }
    }

    size_t CudaBackendImpl::getNParticles() {
        return ctx.particles.size();
    }

    void CudaBackendImpl::getParticlePositions(std::vector<float>& positions) {
        ctx.particles.pull();
        positions.resize(ctx.particles.size() * 3);
        for (size_t i = 0; i < ctx.particles.size(); i++) {
            positions[i * 3 + 0] = ctx.particles[i].position.x;
            positions[i * 3 + 1] = ctx.particles[i].position.y;
            positions[i * 3 + 2] = ctx.particles[i].position.z;
        }
    }

    void CudaBackendImpl::getParticleVelocities(std::vector<float>& velocities) {
        ctx.particles.pull();
        velocities.resize(ctx.particles.size() * 3);
        for (size_t i = 0; i < ctx.particles.size(); i++) {
            velocities[i * 3 + 0] = ctx.particles[i].velocity.x;
            velocities[i * 3 + 1] = ctx.particles[i].velocity.y;
            velocities[i * 3 + 2] = ctx.particles[i].velocity.z;
        }
    }

    void CudaBackendImpl::setParticlePositions(std::vector<float> const& positions) {
        ctx.particles.pull();
        for (size_t i = 0; i < positions.size(); i += 3) {
            ctx.particles[i].position = {positions[i], positions[i+1], positions[i+2]};
        }
        ctx.particles.push();
    }

    void CudaBackendImpl::setParticleVelocities(std::vector<float> const& velocities) {
        ctx.particles.pull();
        for (size_t i = 0; i < velocities.size(); i += 3) {
            ctx.particles[i].velocity = {velocities[i], velocities[i+1], velocities[i+2]};
        }
        ctx.particles.push();
    }

    void CudaBackendImpl::setParticles(std::vector<float> const& positions, std::vector<float> const& velocities) {
        ctx.particles.pull();
        for (size_t i = 0; i < positions.size(); i += 3) {
            ctx.particles[i].position = {positions[i], positions[i+1], positions[i+2]};
            ctx.particles[i].velocity = {velocities[i], velocities[i+1], velocities[i+2]};
        }
        ctx.particles.push();
    }

    void CudaBackendImpl::stepAndAccumulateSample(int n_steps) {
        for (int i = 0; i < n_steps; i++) {
            step(1);

            ctx.mpc_cells.set(0); // Clear cells
            sampling::addParticles<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.mpc_cells, ctx.particles);
            error_check( "add particles reduce only" );

            sampling::averageCells<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.mpc_cells);
            error_check( "add particles reduce only" );

            if (sampling_state == SAMPLING_COMPLETED) {
                sampling_state = SAMPLING_IN_PROGRESS;
                sample_counter = 0;
                ctx.kahan_c.set(0);
                sampling::snapshot<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.mpc_cells, ctx.cell_states);
            } else {
                sample_counter++;
                sampling::accumulate<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.mpc_cells, ctx.cell_states, ctx.kahan_c);
            }
            error_check( "sample accumulate" );
        }
    }

    void CudaBackendImpl::getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity) {
        ctx.cell_states.pull();
        mean_density.resize(ctx.cell_states.size());
        mean_velocity.resize(ctx.cell_states.size() * 3);
        auto inverse = mpcd::Float(1) / sample_counter;
        for (size_t i = 0; i < ctx.cell_states.size(); i++) {
            mean_density[i]           = ctx.cell_states[i].density * inverse;
            mean_velocity[i * 3 + 0]  = ctx.cell_states[i].mean_velocity.x * inverse;
            mean_velocity[i * 3 + 1]  = ctx.cell_states[i].mean_velocity.y * inverse;
            mean_velocity[i * 3 + 2]  = ctx.cell_states[i].mean_velocity.z * inverse;
        }
        sampling_state = SAMPLING_COMPLETED;
    }

} // namespace mpcd::cuda
