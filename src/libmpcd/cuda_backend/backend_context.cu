#include <algorithm>
#include <string>

#include <mpcd/api/simulation_parameters.hpp>
#ifdef USE_HDF5
    #include "adapters/hdf5/h5cpp.hpp"
#endif

#include "backend_context.hpp"
#include "simulation_kernels.hpp"

namespace mpcd::cuda {

    /**
    *   @brief Initialize the GPU and the SRD fluid
    */
    BackendContext::BackendContext(SimulationParameters const& params) :
        parameters([&]{ cudaSetDevice(params.device_id); return params; }()),
        particles(static_cast<size_t>(params.volume_size[0] * params.volume_size[1] * params.volume_size[2] * params.n)),
        particles_sorted(0),
        mpc_cells({params.volume_size[0], params.volume_size[1], params.volume_size[2]}),
        cell_states(static_cast<size_t>(params.volume_size[0] * params.volume_size[1] * params.volume_size[2])),
        kahan_c(cell_states.size()),
        uniform_list(mpc_cells.size() * 4 * params.n),
        uniform_counter(mpc_cells.size()),
        cell_prefix(mpc_cells.size()),
        sort_tmp(mpc_cells.size()),
        use_tmp_sort_buffer(false),
        step_counter(0),
        resort_rate(100)
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
                cuda_config.block_size = 128;  // 4 warps per SM
                cuda_config.block_count *= 4;

                cuda_config.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 4 * 4 );
                cuda_config.sharing_blocks = properties.multiProcessorCount * 4 * 4;  // 4 warps per SM
            }
            else                            // gtx 960
            {
                cuda_config.block_size = 64;
                cuda_config.block_count *= 2;

                cuda_config.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 2 * 2 );
                cuda_config.sharing_blocks = properties.multiProcessorCount * 2 * 2;
            }
        }
        else if ( properties.major < 8 ) // Turing:
        {
            std::cout << "cuda: turing architecture ... " << std::endl;

            cuda_config.block_size = 64;
            cuda_config.block_count *= 4;

            cuda_config.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 2 * 4 );
            cuda_config.sharing_blocks = properties.multiProcessorCount * 2 * 4;
        }
        else // Ampere:
        {
            std::cout << "cuda: ampere architecture ... " << std::endl;

            cuda_config.block_size = 64;
            cuda_config.block_count *= 8;

            cuda_config.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 2 * 8 );
            cuda_config.sharing_blocks = properties.multiProcessorCount * 2 * 8;
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

        // Allocate scatter-sort buffer if VRAM is plentiful (>3× particle array size free).
        // Falls back to in-place cyclic-permutation sort when memory is sparse.
        {
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            use_tmp_sort_buffer = free_mem > particles.size() * sizeof(Particle) * 3;
            if (use_tmp_sort_buffer)
                particles_sorted.alloc(particles.size());
        }

        std::cout << "gpu initialized ..." << std::endl;

    #ifdef USE_HDF5
        // Create output file
        mpcd::adapters::hdf5::file data( "simulation_data.h5", "rw" );
        data.create_group( "fluid" );
    #else
        // No backend output, use pybind11 frontend
    #endif
    }

    BackendContext::~BackendContext() {
        cudaDeviceSynchronize();
    }

} // namespace mpcd::cuda
