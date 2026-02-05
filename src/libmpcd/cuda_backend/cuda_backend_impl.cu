#include <algorithm>
#include <string>

#include <mpcd/api/simulation_parameters.hpp>
#ifdef USE_HDF5
    #include "adapters/hdf5/h5cpp.hpp"
#endif
#include "common/probing.hpp"

#include "cuda_backend_impl.hpp"
#include "simulation_kernels.hpp"
#include "extended_collision.hpp"

namespace mpcd::cuda {
    /**
    *   @brief Initialize the GPU and the SRD fluid
    */
    CudaBackendImpl::CudaBackendImpl(SimulationParameters const& params) : parameters([&]{ cudaSetDevice(params.device_id); return params; }()),
                                                                   particles(static_cast<size_t>(params.volume_size[0] * params.volume_size[1] * params.volume_size[2] * params.n)),
                                                                   particles_sorted(0),
                                                                   mpc_cells({params.volume_size[0], params.volume_size[1], params.volume_size[2]}),
                                                                   cell_states(static_cast<size_t>(params.volume_size[0] * params.volume_size[1] * params.volume_size[2])),
                                                                   kahan_c(cell_states.size()),
                                                                   uniform_list(mpc_cells.size() * 4 * params.n),
                                                                   uniform_counter(mpc_cells.size()),
                                                                   step_counter(0),
                                                                   resort_rate(100),
                                                                   sample_counter(0),
                                                                   sampling_state(SAMPLING_COMPLETED)
    {
        // query device properties to decide kernel launch layouts
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, parameters.device_id);
        cuda_config.multiprocessors = properties.multiProcessorCount;
        cuda_config.block_count     = properties.multiProcessorCount;
        cuda_config.shared_bytes    = properties.sharedMemPerMultiprocessor;
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

        if ( properties.major < 7 )
        {
            std::cout << "cuda: pascal architecture ... " << std::endl;

            if ( cuda_config.block_count > 28 )    // p100
            {
                cuda_config.block_size = 64;
                cuda_config.block_count *= 4;  // concurrent kernels

                cuda_config.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 2 * 4 );
                cuda_config.sharing_blocks = properties.multiProcessorCount * 2 * 4;  // 2 warps per SM, 4x occupancy.
            }
            else if ( cuda_config.block_count > 14 )         // gtx 1070 etc.
            {
                cuda_config.block_size = 128;
                cuda_config.block_count *= 4;

                cuda_config.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 4 * 4 );
                cuda_config.sharing_blocks = properties.multiProcessorCount * 4 * 4;  // 4 warps per SM, 4x occupancy.
            }
            else                            // gtx 960
            {
                cuda_config.block_size = 64;
                cuda_config.block_count *= 2;

                cuda_config.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 2 * 2 );
                cuda_config.sharing_blocks = properties.multiProcessorCount * 2 * 2;  // 2 warps per SM, 2x occupancy.
            }
        }
        else // Turing:
        {
            std::cout << "cuda: turing architecture ... " << std::endl;

            cuda_config.block_size = 64;
            cuda_config.block_count *= 4;

            cuda_config.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 2 * 4 );
            cuda_config.sharing_blocks = properties.multiProcessorCount * 2 * 4;  // 4 warps per SM, 4x occupancy.
        }

        // seed the parallel random number generators
        generator.alloc(cuda_config.block_size * cuda_config.sharing_blocks);
        {
            UnifiedVector<uint64_t> seed(2 * generator.size());

            for (auto& item : seed)
                item = std::hash<uint64_t>()(random());

            seed.push();
            initialize::seedRandomNumberGenerators<<<cuda_config.block_count, cuda_config.block_size>>>(generator, seed);
            error_check("initialise_generators");
        }

        // initialize SRD fluid particles
        grid_shift = {random.genUniformFloat() - mpcd::Float(0.5),
                    random.genUniformFloat() - mpcd::Float(0.5),
                    random.genUniformFloat() - mpcd::Float(0.5)};

        initialize::distributeParticles<<<cuda_config.block_count, cuda_config.block_size>>>(particles, mpc_cells, generator, grid_shift,
                                        {parameters.volume_size[0], parameters.volume_size[1], parameters.volume_size[2]},
                                        {parameters.periodicity[0], parameters.periodicity[1], parameters.periodicity[2]},
                                        parameters.thermal_velocity, parameters.experiment, 0);

        cudaDeviceSynchronize();
        error_check( "distribute_particles" );
        std::cout << "gpu initialized ..." << std::endl;

    #ifdef USE_HDF5
        // Create output file
        mpcd::adapters::hdf5::file data( "simulation_data.h5", "rw" );
        data.create_group( "fluid" );
    #else
        // No backend output, use pybind11 frontend
    #endif
    }

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

        ProbingType probe = what_to_do(step_counter, parameters);

        mpc_cells.set( 0 ); // clear cells.
        sampling::addParticles<<<cuda_config.block_count, cuda_config.block_size>>>(mpc_cells, particles);
        error_check( "add_particles_reduce_only" );

        sampling::averageCells<<< cuda_config.block_count, cuda_config.block_size>>>(mpc_cells);
        error_check( "average_cells_reduce_only" );

        switch (probe){
            case ProbingType::snapshots_only:
                kahan_c.set(0);
                sampling::snapshot<<<cuda_config.block_count, cuda_config.block_size>>>(mpc_cells, cell_states);
                error_check( "snapshot" );
                break;
            case ProbingType::start_accumulating:
                sampling::snapshot<<<cuda_config.block_count, cuda_config.block_size>>>(mpc_cells, cell_states);
                error_check( "snapshot" );
                break;
            case ProbingType::accumulate:
                sampling::accumulate<<<cuda_config.block_count, cuda_config.block_size>>>(mpc_cells, cell_states, kahan_c);
                error_check( "accumulate" );
                break;
            case ProbingType::finish_accumulation:
                sampling::finish<<<cuda_config.block_count, cuda_config.block_size>>>(parameters.average_samples / parameters.sample_every,
                                                                                    mpc_cells, cell_states, kahan_c);
                error_check( "finish" );
                break;
        }

        if (probe == ProbingType::finish_accumulation or probe == ProbingType::snapshots_only) {
            cell_states.pull();
        #ifdef USE_HDF5
            mpcd::adapters::hdf5::file data( "simulation_data.h5", "a" ); // append
            data.write_float_data(std::string("fluid/") + std::to_string(step_counter), reinterpret_cast<float*>(cell_states.data()),
                                {4, static_cast<size_t>(parameters.volume_size[0]), static_cast<size_t>(parameters.volume_size[1]),
                                static_cast<size_t>(parameters.volume_size[2])});
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
        grid_shift = {random.genUniformFloat() - mpcd::Float(0.5),
                    random.genUniformFloat() - mpcd::Float(0.5),
                    random.genUniformFloat() - mpcd::Float(0.5)};

        uniform_counter.set(0);
        translateParticles<<<cuda_config.block_count, cuda_config.block_size>>>( particles, mpc_cells, generator, grid_shift,
                                                                                {parameters.volume_size[0], parameters.volume_size[1], parameters.volume_size[2]},
                                                                                {parameters.periodicity[0], parameters.periodicity[1], parameters.periodicity[2]},
                                                                                parameters.delta_t, parameters.experiment,
                                                                                uniform_counter, uniform_list );
        error_check( "translate_particles" );
    }

    /**
    *   @brief Perform SRD collision step
    */
    void CudaBackendImpl::collisionStep() {
        mpc_cells.set(0);

        switch (parameters.algorithm) {
            case MPCDAlgorithm::srd:
                srdCollision<<<cuda_config.sharing_blocks, 32, cuda_config.shared_bytes>>>(particles, mpc_cells, generator.data(), grid_shift,
                                                                                {parameters.volume_size[0], parameters.volume_size[1], parameters.volume_size[2]},
                                                                                {parameters.periodicity[0], parameters.periodicity[1], parameters.periodicity[2]},
                                                                                parameters.delta_t, parameters.drag, parameters.thermal_velocity, parameters.n,
                                                                                uniform_counter, uniform_list, cuda_config.shared_bytes);
                break;
            case MPCDAlgorithm::extended:
                extendedCollision<<<cuda_config.sharing_blocks, 32, cuda_config.shared_bytes>>>(particles, mpc_cells, generator.data(), grid_shift,
                                                                                {parameters.volume_size[0], parameters.volume_size[1], parameters.volume_size[2]},
                                                                                {parameters.periodicity[0], parameters.periodicity[1], parameters.periodicity[2]},
                                                                                parameters.delta_t, parameters.drag, parameters.thermal_velocity, parameters.n,
                                                                                uniform_counter, uniform_list, cuda_config.shared_bytes);
                break;
            default:
                break;
        }
        error_check( "collision_step" );
    }

    // Backend operations overrides

    void CudaBackendImpl::step(int n_steps) {
        for (int i = 0; i < n_steps; i++) {
            // MPC steps:
            translationStep();
            collisionStep();

            // sort praticles array to improve memory access times
            if ((step_counter++ % resort_rate) == 0) {
                particles.pull();
                std::sort(particles.begin(), particles.end(), [] (auto a, auto b) { return a.cell_idx < b.cell_idx; });
                particles.push();
            }
        }
    }

    size_t CudaBackendImpl::getNParticles() {
        return particles.size();
    }

    void CudaBackendImpl::getParticlePositions(std::vector<float>& positions) {
        particles.pull();
        positions.resize(particles.size() * 3);
        for (size_t i = 0; i < particles.size(); i++) {
            positions[i * 3 + 0] = particles[i].position.x;
            positions[i * 3 + 1] = particles[i].position.y;
            positions[i * 3 + 2] = particles[i].position.z;
        }
    }

    void CudaBackendImpl::getParticleVelocities(std::vector<float>& velocities) {
        particles.pull();
        velocities.resize(particles.size() * 3);
        for (size_t i = 0; i < particles.size(); i++) {
            velocities[i * 3 + 0] = particles[i].velocity.x;
            velocities[i * 3 + 1] = particles[i].velocity.y;
            velocities[i * 3 + 2] = particles[i].velocity.z;
        }
    }

    void CudaBackendImpl::setParticlePositions(std::vector<float> const& positions) {
        particles.pull();
        for (size_t i = 0; i < positions.size(); i += 3) {
            particles[i].position = {positions[i], positions[i+1], positions[i+2]};
        }
        particles.push();
    }

    void CudaBackendImpl::setParticleVelocities(std::vector<float> const& velocities) {
        particles.pull();
        for (size_t i = 0; i < velocities.size(); i += 3) {
            particles[i].velocity = {velocities[i], velocities[i+1], velocities[i+2]};
        }
        particles.push();
    }

    void CudaBackendImpl::setParticles(std::vector<float> const& positions, std::vector<float> const& velocities) {
        particles.pull();
        for (size_t i = 0; i < positions.size(); i += 3) {
            particles[i].position = {positions[i], positions[i+1], positions[i+2]};
            particles[i].velocity = {velocities[i], velocities[i+1], velocities[i+2]};
        }
        particles.push();

    }

    void CudaBackendImpl::stepAndAccumulateSample(int n_steps) {
        for (int i = 0; i < n_steps; i++) {
            step(1);

            mpc_cells.set( 0 ); // Clear cells
            sampling::addParticles<<<cuda_config.block_count, cuda_config.block_size>>>(mpc_cells, particles);
            error_check( "add particles reduce only" );

            sampling::averageCells<<<cuda_config.block_count, cuda_config.block_size>>>(mpc_cells);
            error_check( "add particles reduce only" );

            if (sampling_state == SAMPLING_COMPLETED) {
                sampling_state = SAMPLING_IN_PROGRESS;
                sample_counter = 0;
                kahan_c.set(0);
                sampling::snapshot<<<cuda_config.block_count, cuda_config.block_size>>>(mpc_cells, cell_states);
            } else {
                sample_counter++;
                sampling::accumulate<<<cuda_config.block_count, cuda_config.block_size>>>(mpc_cells, cell_states, kahan_c);
            }
            error_check( "sample accumulate" );
        }
    }

    void CudaBackendImpl::getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity) {
        cell_states.pull();
        mean_density.resize(cell_states.size());
        mean_velocity.resize(cell_states.size() * 3);
        auto inverse = Float(1) / sample_counter;
        for (size_t i = 0; i < cell_states.size(); i++) {
            mean_density[i] = cell_states[i].density * inverse;
            mean_velocity[i * 3 + 0] = cell_states[i].mean_velocity.x * inverse;
            mean_velocity[i * 3 + 1] = cell_states[i].mean_velocity.y * inverse;
            mean_velocity[i * 3 + 2] = cell_states[i].mean_velocity.z * inverse;
        }
        sampling_state = SAMPLING_COMPLETED;
    }
} // namespace mpcd::cuda
