#include <mpcd/api/simulation_parameters.hpp>

#include "simulation_kernels.hpp"
#include "common/mechanic.hpp"

namespace mpcd::cuda {
    namespace initialize {
        /**
        *  @brief Initialize GPU random number generators
        */
        __global__ void seedRandomNumberGenerators(DeviceVector<Xoshiro128Plus> generator, DeviceVector<uint64_t> seed) {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
                stride = blockDim.x * gridDim.x;

            for (; idx < generator.size(); idx += stride)
                generator[idx].seed(seed[idx], seed[idx + generator.size()]);
        }

        /**
        *  @brief Initialize the fluid by distributing the SRD fluid particles in the simulation volume
        */
        __global__ void distributeParticles(DeviceVector<Particle> particles, DeviceVolumeContainer<MPCCell> mpc_cells,
                                            DeviceVector<Xoshiro128Plus> generator, mpcd::Vector grid_shift,
                                            mpcd::Vector volume_size, mpcd::IntVector periodicity, mpcd::Float thermal_velocity,
                                            ExperimentType experiment_type,
                                            uint32_t start)
        {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;
            auto   random = generator[idx];
            mpcd::Vector scale = volume_size - 2 * (1 - periodicity);

            float channel_radius2 = (volume_size.z - 2) * (volume_size.z - 2) * 0.25f;

            idx += start;
            for (auto end = particles.size(); idx < end; idx += stride)
            {
                Particle particle = {};
                bool          replace;

                do {
                    replace  = false;
                    particle.position = {random.genUniformFloat(), random.genUniformFloat(), random.genUniformFloat()}; // uniform on the unit cube.
                    particle.position = (particle.position - mpcd::Float(0.5)).scaledWith(scale);  // rescale to the simulation volume

                    if (experiment_type == ExperimentType::channel)
                        replace = replace or ((particle.position.z * particle.position.z + particle.position.y * particle.position.y) > channel_radius2);

                } while (replace);

                particle.velocity = random.maxwellBoltzmann() * thermal_velocity;
                particle.cell_idx = mpc_cells.getIndex(particle.position);

                particles[idx] = particle;
            }

            generator[idx % stride] = random; // store new state of the random number generator
        }

    }  //  namespace initialize

    /**
    *  @brief This function applies the SRD streaming step to the fluid particles
    */
    __global__ void translateParticles(DeviceVector<Particle> particles, DeviceVolumeContainer<MPCCell> mpc_cells,
                                    DeviceVector<Xoshiro128Plus> generator, mpcd::Vector grid_shift,
                                    mpcd::Vector volume_size, mpcd::IntVector periodicity, mpcd::Float delta_t,
                                    ExperimentType experiment_type,
                                    DeviceVector<uint32_t> uniform_counter, DeviceVector<uint32_t> uniform_list)
    {
        size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
               stride = blockDim.x * gridDim.x;
        auto   random = generator[ idx ];

        uint32_t cell_lookup_size = uniform_list.size() / uniform_counter.size();
        float channel_radius2 = ( volume_size.z - 2 ) * ( volume_size.z - 2 ) * 0.25f;

        auto apply_periodic_boundaries = [&] ( auto r )
        {
            r.x = fmodf( r.x + 1.5f * volume_size.x, volume_size.x ) - volume_size.x * 0.5f;
            r.y = fmodf( r.y + 1.5f * volume_size.y, volume_size.y ) - volume_size.y * 0.5f;
            r.z = fmodf( r.z + 1.5f * volume_size.z, volume_size.z ) - volume_size.z * 0.5f; // does not interfere with using walls...
            return r;
        };

        for ( auto end = particles.size(); idx < end; idx += stride )
        {
            auto particle = gpu_utilities::texture_load( particles.data() + idx ); // load via texture load path

            if ( experiment_type != ExperimentType::channel )
            {
                if ( not periodicity.z ) // if walls are present, calculate collisions
                {
                    auto z_wall = ( 0.5f * volume_size.z - 1 ); // distance of the walls, remove one layer for ghost particles
                    auto next_z = particle.position.z + particle.velocity.z * delta_t;

                    if ( fabsf( next_z ) > z_wall ) // this is more safe than just calcualating the time
                    {
                        auto time_left = ( next_z > 0 ? z_wall - particle.position.z : -z_wall - particle.position.z )
                                        / particle.velocity.z;

                        particle.position += particle.velocity * time_left;
                        particle.velocity = -particle.velocity; // bounce back roule, creates no-slip boundary condition
                        particle.position += particle.velocity * ( delta_t - time_left );
                    }
                    else
                        particle.position += particle.velocity * delta_t;
                }
                else
                    particle.position += particle.velocity * delta_t;
            }
            else // channel:
            {
                particle.position += particle.velocity * delta_t; // advance particle

                if ( ( particle.position.z * particle.position.z + particle.position.y * particle.position.y ) > channel_radius2 )
                {
                    particle.position -= particle.velocity * delta_t; // apply correction if it left the channel
                    particle.velocity = -particle.velocity;
                }
            }

            particle.position = apply_periodic_boundaries( particle.position + grid_shift );
            particle.cell_idx = mpc_cells.getIndex( particle.position );

            // make an entry in the index lookup for the cell in which the particle lies
            int slot = atomicAdd( uniform_counter.data() + particle.cell_idx, 1 );
            if ( slot < cell_lookup_size )
                uniform_list[particle.cell_idx + slot * mpc_cells.size()] = idx;
            else
                particle.position = apply_periodic_boundaries( particle.position - grid_shift );
            // Table overflow: the particle keeps its pre-shift position so it is handled correctly in the collision step.
            // The table holds 4n slots; cell occupancy is Poisson(n), so overflow needs >4n particles — 3√n σ above the
            // mean. At n≥5 that is >6.7σ (p≈10⁻¹¹ per cell per step), making this branch physically insignificant.

            assert( particle.position.isFinite() ); // check for error in floating point math
            assert( particle.velocity.isFinite() );

            particles[idx] = particle;
        }

        generator[idx % stride] = random;
    }

    namespace sampling
    {
        /**
        *  @brief 1st step to compute the fluid state on the grid of the SRD collision cell
        */
        __global__ void addParticles(DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<Particle> particles) {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;

            for (auto end = particles.size(); idx < end; idx += stride) {
                auto particle = particles[idx];
                mpc_cells[particle.position].addReduceOnly(particle.velocity);
            }
        }

        /**
        *  @brief 2nd step to compute the fluid state on the grid of the SRD collision cell
        */
        __global__ void averageCells(DeviceVolumeContainer<MPCCell> mpc_cells)
        {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;

            for (; idx < mpc_cells.size(); idx += stride)
                mpc_cells[idx].averageReduceOnly();
        }

        /**
        *  @brief Store one timestep, either to initialize a time average or to store just one time step
        */
        __global__ void snapshot(DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<FluidState> cell_states) {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;

            for (; idx < mpc_cells.size(); idx += stride) {
                cell_states[idx].density       = mpc_cells[idx].density;
                cell_states[idx].mean_velocity = mpc_cells[idx].mean_velocity;
            }
        }

        /**
        *  @brief Add more data when performing an average
        */
        __global__ void accumulate(DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<FluidState> cell_states, DeviceVector<FluidState> kahan_c) {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;

            for (; idx < mpc_cells.size(); idx += stride) {
                {
                    auto y = mpc_cells[idx].density - kahan_c[idx].density;
                    auto t = cell_states[idx].density + y;
                    kahan_c[idx].density = (t - cell_states[idx].density) - y;
                    cell_states[idx].density = t;
                }
                auto y = mpc_cells[idx].mean_velocity - kahan_c[idx].mean_velocity;
                auto t = cell_states[idx].mean_velocity + y;
                kahan_c[idx].mean_velocity = (t - cell_states[idx].mean_velocity) - y;
                cell_states[idx].mean_velocity = t;
            }
        }

        /**
        *  @brief Finish performing the average
        */
        __global__ void finish(size_t n_samples, DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<FluidState> cell_states, DeviceVector<FluidState> kahan_c) {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
                stride = blockDim.x * gridDim.x;

            double inverse = 1.0 / n_samples;

            for (; idx < mpc_cells.size(); idx += stride) {
                {
                    auto y = mpc_cells[idx].density - kahan_c[idx].density;
                    auto t = cell_states[idx].density + y;
                    kahan_c[idx].density = (t - cell_states[idx].density) - y;
                    cell_states[idx].density = t;
                }
                auto y = mpc_cells[idx].mean_velocity - kahan_c[idx].mean_velocity;
                auto t = cell_states[idx].mean_velocity + y;
                kahan_c[idx].mean_velocity = (t - cell_states[idx].mean_velocity) - y;
                cell_states[idx].mean_velocity = t;

                cell_states[idx].density       *= inverse;
                cell_states[idx].mean_velocity *= inverse;
            }
        }
    } // namespace sampling
} // namespace mpcd::cuda
