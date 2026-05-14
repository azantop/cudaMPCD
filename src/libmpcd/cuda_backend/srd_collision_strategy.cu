#include <mpcd/api/simulation_parameters.hpp>

#include "srd_collision_strategy.hpp"
#include "gpu_error_check.hpp"
#include "common/mechanic.hpp"

namespace mpcd::cuda {

    /**
    *  @brief This function applies the SRD collision step to the fluid particles
    */
    __global__ void __launch_bounds__(32) srdCollision(DeviceVector<Particle> particles, DeviceVolumeContainer<MPCCell> mpc_cells,
                                                            Xoshiro128Plus* generator,
                                                            mpcd::Vector grid_shift,
                                                            mpcd::Vector volume_size,
                                                            mpcd::IntVector periodicity,
                                                            mpcd::Float delta_t, mpcd::Float drag, mpcd::Float thermal_velocity,
                                                            uint32_t n_density,
                                                            DeviceVector<uint32_t> uniform_counter,
                                                            DeviceVector<uint32_t> uniform_list, uint32_t const shared_bytes )
    {
        extern __shared__ uint32_t shared_mem [];

        // asign shared memory for particle positions and velocities
        uint32_t const max_particles      = shared_bytes / ( sizeof( uint32_t ) + 2 * sizeof( mpcd::Vector ) ); // this is the per particle memory
        uint32_t     * particle_idx       = shared_mem;                                                         // storing the indices of loaded particles
        mpcd::Vector * particle_position  = reinterpret_cast< mpcd::Vector* >( particle_idx + max_particles );  // 1st vector
        mpcd::Vector * particle_velocity  = particle_position + max_particles;                                  // 2nd vector

        uint32_t       cell_idx           = blockIdx.x * blockDim.x + threadIdx.x,
                    stride             = blockDim.x * gridDim.x;

        auto           random             = generator[ cell_idx ];
        uint32_t const cell_lookup_size   = uniform_list.size() / uniform_counter.size();
        auto     const regular_particles  = particles.size();

        auto const     shift              = fabs( grid_shift.z ); // create wall's ghost particles on the fly
        unsigned const sign               = grid_shift.z > 0;
        float const    sin_alpha          = sinf( ( M_PI * 120 ) / 180 ), // these are used to represent the SRD rotation matrix
                    cos_alpha             = cosf( ( M_PI * 120 ) / 180 );

        drag *= delta_t; // pressure gradient that accelerates the flow

        auto apply_periodic_boundaries = [&] ( auto r ) // does not interfere with using walls...
        {
            r.x = fmodf( r.x + 1.5f * volume_size.x, volume_size.x ) - volume_size.x * 0.5f;
            r.y = fmodf( r.y + 1.5f * volume_size.y, volume_size.y ) - volume_size.y * 0.5f;
            r.z = fmodf( r.z + 1.5f * volume_size.z, volume_size.z ) - volume_size.z * 0.5f;
            return r;
        };

        for ( uint32_t end = mpc_cells.size(); __any_sync( 0xFFFFFFFF, cell_idx < end ); cell_idx += stride )
        {
            random.syncPhase();

            MPCCell cell         = {};
            uint32_t      n_particles  = (cell_idx < mpc_cells.size()) ? ::min(cell_lookup_size, uniform_counter[cell_idx]) : 0; // load the table size
            bool const    layer        = (mpc_cells.getZIdx(cell_idx) == (volume_size.z - (sign ? 1 : 2)));
            bool const    add_ghosts   = (not periodicity.z) and ((mpc_cells.getZIdx(cell_idx) == sign) or layer); // wall layer?
            uint32_t      added_ghosts = {};

            if (add_ghosts) // create wall's ghost particles on the fly; prepare number of ghosts
                for (int i = 0; i < n_density; ++i)
                    if ((random.genUniformFloat() > shift) xor layer xor sign)
                        ++added_ghosts;

            n_particles += added_ghosts;

            // arrange shared memory and decides how many cells can be fit into memory and handeled at once
            uint32_t const prefix        = gpu_utilities::warp_prefix_sum( n_particles );
            uint32_t const active_cells  = __popc( __ballot_sync( 0xFFFFFFFF, prefix + n_particles < max_particles and cell_idx < mpc_cells.size() ) );
            uint32_t const sum           = __shfl_sync( 0xFFFFFFFF, prefix + n_particles, active_cells - 1 );
            bool     const thread_active = prefix + n_particles < max_particles and cell_idx < mpc_cells.size();

            if (thread_active) {
                for (uint32_t i = 0; i < n_particles - added_ghosts; ++i)
                    particle_idx[prefix + i] = cell_idx + i * mpc_cells.size(); // write to shared mem which lookup table positions need to be loaded.

                if (add_ghosts) { // create wall's ghost particles on the fly
                    auto pos = mpc_cells.getPosition(cell_idx);

                    for (int i = n_particles - added_ghosts; i < n_particles; ++i)
                    {
                        float z;

                        do {
                            z = random.genUniformFloat();
                        } while ( ( ( z < shift ) xor layer ) xor sign );

                        particle_idx     [prefix + i] = -1u; // this means that this particle should not be loaded
                        particle_position[prefix + i] = mpcd::Vector(random.genUniformFloat() - 0.5f, random.genUniformFloat() - 0.5f, z - 0.5f) + pos;
                        particle_velocity[prefix + i] = random.maxwellBoltzmann() * thermal_velocity;
                    }
                }
            }

            __syncwarp();

            for (uint32_t i = threadIdx.x; i < sum; i += 32) // load the entries of the lookup table uniformly without binding threads to SRD cells
                if (particle_idx[i] != -1u)
                    particle_idx[i] = gpu_utilities::texture_load(uniform_list.data() + particle_idx[i]);

            __syncwarp();

            for (uint32_t i = threadIdx.x; i < sum; i += 32) { // load the SRD fluid particles uniformly without binding threads to SRD cells
                if (particle_idx[i] != -1u) {
                    if (particle_idx[i] < regular_particles) {
                        auto particle = gpu_utilities::texture_load(particles.data() + particle_idx[i]);

                        particle_position[i] = particle.position;
                        particle_velocity[i] = particle.velocity;
                    }
                }
            }

            if (thread_active) { // SRD collision step, each thread one SRD cell
                if (n_particles > 1) {
                    for (uint32_t i = 0; i < n_particles; ++i)
                        cell.unlockedAdd({0, 0, particle_position[prefix + i], particle_velocity[prefix + i]});

                    cell.average();
                    random.syncPhase();

                    auto  axis = random.genUnitVector();

                    for (uint32_t i = 0; i < n_particles; ++i) { // rotation step:
                        auto v      = particle_velocity[prefix + i] - cell.mean_velocity;
                        auto v_para = axis * (v.dotProduct(axis));
                        auto v_perp = v - v_para;

                        particle_velocity[prefix + i] = v_para + cos_alpha * v_perp + sin_alpha * v_perp.crossProduct(axis);
                    }

                    for (uint32_t i = 0; i < n_particles; ++i) { // finilize step:
                        particle_velocity[prefix + i] += cell.getCorrection(particle_position[prefix + i]);
                        particle_position[prefix + i] = apply_periodic_boundaries(particle_position[prefix + i] - grid_shift);
                        particle_velocity[prefix + i].x += drag;
                    }
                }
            }
            else
                cell_idx -= stride;

            cell_idx = __shfl_sync( 0xFFFFFFFF, cell_idx, threadIdx.x + active_cells ); // cyclic shift so that theads' cell_index remains uniform

            for (uint32_t i = threadIdx.x; i < sum; i += 32) { // write the SRD fluid particles to memory uniformly without binding threads to SRD cells
                if (particle_idx[ i ] != -1u) {
                    if (particle_idx [ i ] < regular_particles) {
                        auto cidx = mpc_cells.getIndex(particle_position[i]);
                        particles[particle_idx[i]] = {0, 0, particle_position[i], particle_velocity[i], cidx};
                    }
                }
            }
        }

        generator[blockIdx.x * blockDim.x + threadIdx.x] = random;
    }

    void SRDStrategy::collideParticles() {
        srdCollision<<<ctx.cuda_config.sharing_blocks, 32, ctx.cuda_config.shared_bytes>>>(
                                    ctx.particles, ctx.mpc_cells, ctx.generator.data(), ctx.grid_shift,
                                    {ctx.parameters.volume_size[0], ctx.parameters.volume_size[1], ctx.parameters.volume_size[2]},
                                    {ctx.parameters.periodicity[0], ctx.parameters.periodicity[1], ctx.parameters.periodicity[2]},
                                    ctx.parameters.delta_t, ctx.parameters.drag, ctx.parameters.thermal_velocity, ctx.parameters.n,
                                    ctx.uniform_counter, ctx.uniform_list, ctx.cuda_config.shared_bytes);
        error_check("collision_step");
    }


    TrivialSRDStrategy::TrivialSRDStrategy(BackendContext& ctx)
        : CollisionStrategy(ctx), rotation_axes(ctx.mpc_cells.size())
    {}

    // Trivial SRD kernels — plain strided scatter/reduce, no shared memory
    namespace srd {

        /**
        *  @brief Scatter particles into cells: accumulate density and sum of velocities.
        *         Uses addReduceOnly — no CoM accumulation needed for pure SRD.
        */
        __global__ void scatter(DeviceVector<Particle>         particles,
                                DeviceVolumeContainer<MPCCell> mpc_cells)
        {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;

            for (auto end = particles.size(); idx < end; idx += stride) {
                auto particle = gpu_utilities::texture_load(particles.data() + idx);
                mpc_cells[particle.cell_idx].addReduceOnly(particle.velocity);
            }
        }

        /**
        *  @brief Average cell accumulators: produces mean_velocity per cell.
        *         Uses averageReduceOnly — no CoM needed for pure SRD.
        */
        __global__ void averageCells(DeviceVolumeContainer<MPCCell> mpc_cells)
        {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;

            for (; idx < mpc_cells.size(); idx += stride)
                mpc_cells[idx].averageReduceOnly();
        }

        /**
        *  @brief Draw one random unit rotation axis per cell and store it.
        *         One thread handles multiple cells via striding.
        */
        __global__ void generateAxes(DeviceVector<Xoshiro128Plus> generator,
                                     DeviceVector<mpcd::Vector>   rotation_axes)
        {
            int  idx    = blockIdx.x * blockDim.x + threadIdx.x,
                 stride = blockDim.x * gridDim.x;
            auto random = generator[idx];

            for (; idx < rotation_axes.size(); idx += stride)
                rotation_axes[idx] = random.genUnitVector();

            generator[blockIdx.x * blockDim.x + threadIdx.x] = random;
        }

        /**
        *  @brief Apply SRD rotation to each particle and unapply the grid shift.
        *         Per particle: v_final = v_mean + R(alpha, axis) * (v - v_mean)
        */
        __global__ void applyCollision(
                DeviceVector<mpcd::Vector>     rotation_axes,
                DeviceVector<Particle>         particles,
                DeviceVolumeContainer<MPCCell> mpc_cells,
                mpcd::Vector                   grid_shift,
                mpcd::Vector                   volume_size,
                mpcd::Float                    drag,
                mpcd::Float                    delta_t)
        {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;

            float const sin_alpha = sinf((M_PI * 120) / 180);
            float const cos_alpha = cosf((M_PI * 120) / 180);

            auto apply_periodic_boundaries = [&](mpcd::Vector r) {
                r.x = fmodf(r.x + 1.5f * volume_size.x, volume_size.x) - volume_size.x * 0.5f;
                r.y = fmodf(r.y + 1.5f * volume_size.y, volume_size.y) - volume_size.y * 0.5f;
                r.z = fmodf(r.z + 1.5f * volume_size.z, volume_size.z) - volume_size.z * 0.5f;
                return r;
            };

            for (auto end = particles.size(); idx < end; idx += stride) {
                auto particle = gpu_utilities::texture_load(particles.data() + idx);
                auto cell_idx = particle.cell_idx;
                auto axis     = rotation_axes[cell_idx];
                auto mean_vel = mpc_cells[cell_idx].mean_velocity;

                auto v_rel    = particle.velocity - mean_vel;

                // Rodrigues rotation of v_rel by alpha around axis
                auto v_para   = axis * v_rel.dotProduct(axis);
                auto v_perp   = v_rel - v_para;
                auto v_rot    = v_para + cos_alpha * v_perp + sin_alpha * v_perp.crossProduct(axis);

                particle.velocity  = mean_vel + v_rot;
                particle.velocity.x += drag * delta_t;
                particle.position  = apply_periodic_boundaries(particle.position - grid_shift);
                particle.cell_idx  = mpc_cells.getIndex(particle.position);

                particles[idx] = particle;
            }
        }

    } // namespace srd

    void TrivialSRDStrategy::collideParticles() {
        // cells are cleared by CudaBackendImpl::collisionStep() before dispatch
        srd::scatter<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.particles, ctx.mpc_cells);
        error_check("srd_trivial_scatter");

        srd::averageCells<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.mpc_cells);
        error_check("srd_trivial_average_cells");

        srd::generateAxes<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(ctx.generator, rotation_axes);
        error_check("srd_trivial_generate_axes");

        srd::applyCollision<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(
            rotation_axes, ctx.particles, ctx.mpc_cells, ctx.grid_shift,
            {ctx.parameters.volume_size[0], ctx.parameters.volume_size[1], ctx.parameters.volume_size[2]},
            ctx.parameters.drag, ctx.parameters.delta_t);
        error_check("srd_trivial_apply_collision");
    }

    void SortingSRDStrategy::sortParticles() {
        CollisionStrategy::sortParticles(); // bypass TrivialSRDStrategy's no-op, call counting sort
    }

} // namespace mpcd::cuda
