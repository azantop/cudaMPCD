#include <mpcd/api/simulation_parameters.hpp>

#include "extended_collision_strategy.hpp"
#include "gpu_error_check.hpp"
#include "common/mechanic.hpp"

namespace mpcd::cuda {

    namespace ext {

        /**
        *  @brief grouping into sub-groups by random plane:
        *          (i)   momentum transpher between groups
        *          (ii)  thrermalised in soubgroups
        *          (iii) angular momentum conservation
        */
        __global__ __launch_bounds__(32) void fusedCollision(DeviceVector<Particle> particles,
                                                                DeviceVolumeContainer<MPCCell> mpc_cells,
                                                                Xoshiro128Plus* generator,
                                                                mpcd::Vector grid_shift,
                                                                mpcd::Vector volume_size,
                                                                mpcd::IntVector periodicity,
                                                                mpcd::Float delta_t,
                                                                mpcd::Float drag,
                                                                mpcd::Float thermal_velocity,
                                                                uint32_t n_density,
                                                                DeviceVector<uint32_t> uniform_counter,
                                                                DeviceVector<uint32_t> uniform_list, uint32_t const shared_bytes)
        {
            extern __shared__ uint32_t shared_mem [];

            // asign shared memory for particle positions and velocities
            uint32_t const max_particles      = shared_bytes / (sizeof(uint32_t) + 2*sizeof(mpcd::Vector)); // this is the per particle memory
            uint32_t     * particle_idx       = shared_mem;                                                         // storing the indices of loaded particles
            mpcd::Vector * particle_position  = reinterpret_cast<mpcd::Vector*>(particle_idx + max_particles);  // 1st vector
            mpcd::Vector * particle_velocity  = particle_position + max_particles;                                  // 2nd vector

            auto           random              = generator[ blockIdx.x * blockDim.x + threadIdx.x ];
            Particle* ghost_particles     = nullptr; // override this to add ghost particles!
            uint32_t const cell_lookup_size    = uniform_list.size() / uniform_counter.size();
            auto     const regular_particles   = particles.size();
            auto           shift               = fabs(grid_shift.z);
            unsigned       sign                = grid_shift.z > 0;
            float const    scale               = 0.01f; // collision probability scale

            drag *= delta_t;

            for ( int cell_idx = blockIdx.x * blockDim.x + threadIdx.x,
                    stride   = blockDim.x * gridDim.x,
                    end      = mpc_cells.size();
                    __any_sync( 0xFFFFFFFF, cell_idx < end ); cell_idx += stride ) // iterate in complete warps, overhanging threads still need to join.
            {
                // ~~~ setup & load particles:

                int  n_particles  = (cell_idx < end) ? ::min( cell_lookup_size, uniform_counter[cell_idx] ) : 0; // read lookup size
                bool layer        = (mpc_cells.getZIdx(cell_idx) == (volume_size.z - (sign ? 1 : 2))); // wall layer?
                bool add_ghosts   = (not periodicity.z)
                                        and ((mpc_cells.getZIdx(cell_idx) == sign) or layer); // wall layer?
                int  added_ghosts = {};

                // pre assign ghost particles to include them into the shared memory partitioning:

                random.syncPhase();
                if (add_ghosts)
                    for (int i = 0; i < n_density; ++i)
                        if (((random.genUniformFloat() > shift) xor layer) xor sign)
                            ++added_ghosts;

                n_particles += added_ghosts;

                // dynamic partitioning of shared memory cache for individual threads/cooperative groups.
                // The warp computes a prefix sum of actual per-cell particle counts, uses __ballot_sync to determine how many
                // cells physically fit in the shared allocation, then partitions three arrays (particle_idx, particle_position, particle_velocity)
                // into that space dynamically. The active_cells count governs sub-warp thread grouping on the fly.

                int       prefix         = gpu_utilities::warp_prefix_sum(n_particles); // where does the threads storage start?
                // how many cells' particles fit into shared memory?
                int const active_cells   = ::min(8, __popc(__ballot_sync(-1u, prefix + n_particles < max_particles and cell_idx < mpc_cells.size())));
                int const group_size     = 32 / active_cells;
                int const sum            = __shfl_sync(-1u, prefix + n_particles, active_cells - 1); // total number of particles of the used number of cells.
                int       group_cell_idx = cell_idx;

                if (group_size > 1) // communicate group variables betweeen grouped threads.
                {
                    auto group_root = threadIdx.x / group_size;

                    n_particles    = __shfl_sync(0xFFFFFFFF, n_particles,  group_root);
                    layer          = __shfl_sync(0xFFFFFFFF, layer,        group_root);
                    added_ghosts   = __shfl_sync(0xFFFFFFFF, added_ghosts, group_root);
                    prefix         = __shfl_sync(0xFFFFFFFF, prefix,       group_root);
                    group_cell_idx = __shfl_sync(0xFFFFFFFF, cell_idx,     group_root);

                    add_ghosts = (added_ghosts != 0);
                }

                // decide if threads are left over and deactivate them:
                bool     const thread_active = prefix + n_particles < max_particles and group_cell_idx < mpc_cells.size();
                uint32_t const mask          = __ballot_sync(0xFFFFFFFF, thread_active); // mask of participating threads for following __shfl operations.

                // define initial coordinate offset to improve calculation of the moment of intertia tensor as the cell centre.
                auto offset = mpc_cells.getPosition( group_cell_idx );

                if (thread_active)
                {
                    for ( int i = threadIdx.x % group_size, end = n_particles - added_ghosts; i < end; i += group_size ) // prepare lookup table: which indices will be loaded?
                        particle_idx[ prefix + i ] = group_cell_idx + i * mpc_cells.size();

                    if (add_ghosts) // in wall layers: fill pre-assigned random "ghost" particles slots
                    {
                        auto z_scale = sign ? (layer ? 1 - shift : shift) : (layer ? shift : 1 - shift);

                        for ( int i = n_particles - added_ghosts + threadIdx.x % group_size; i < n_particles; i += group_size )
                        {
                            float z = z_scale * random.genUniformFloat();

                            particle_idx     [prefix + i] = -1u;
                            particle_velocity[prefix + i] = random.maxwellBoltzmann() * thermal_velocity;
                            particle_position[prefix + i] = mpcd::Vector(random.genUniformFloat() - 0.5f,
                                                                         random.genUniformFloat() - 0.5f,
                                                                         layer ? 0.5f - z : z - 0.5f)
                                                                + offset;
                        }
                    }
                }

                __syncwarp(); // dependencies in accesses to shared mem have to be synced -> memory fence...

                for ( int i = threadIdx.x; i < sum; i += 32 ) // read the lookup table of the MPCD cells in use.
                    if (particle_idx[i] != -1u)
                        particle_idx[i] = __ldg(uniform_list.data() + particle_idx[i]); // using texture load path

                __syncwarp();

                for (int i = threadIdx.x; i < sum; i += 32) // now transfer the particles into shared mem based on the lookup table
                {
                    if (particle_idx[i] != -1u)
                    {
                        // using texture load path __ldg()
                        auto addr = particle_idx[ i ] < particles.size() ? particles.data() : ghost_particles;
                        auto particle = gpu_utilities::texture_load(addr + particle_idx[i]);
                        particle_position[i] = particle.position;
                        particle_velocity[i] = particle.velocity;
                    }
                }
                __syncwarp();

                // ~~~ apply collision rule:

                if ( thread_active )
                {
                    random.syncPhase();

                    #if 1 // discrete axis set or continuous random vector

                        int constexpr steps = 4; // discretization. careful: avoid overweight of theta = 0 pole with step phi cases...

                        float theta, phi; // chose discretized random direction:
                        {
                            int select  = gpu_utilities::group_share(random.genUniformInt(0, steps * (steps - 1)), mask, group_size);
                            int phi_i   = select % steps + 1,
                                theta_i = select / steps + 1;

                            if ( (theta_i % 2) and (phi_i % 2) ) // edges
                                theta = theta_i == 1 ? 0.95531661812450927816f : float(M_PI) - 0.95531661812450927816f;
                            else
                                theta = theta_i * (float(M_PI) / steps);

                            phi = phi_i * (float(M_PI) / steps);
                        }
                        mpcd::Vector axis  = { __sinf(theta) * __cosf(phi), // transfer from spherical coordinates to cartesian.
                                               __sinf(theta) * __sinf(phi),
                                               __cosf(theta) };

                        if ( gpu_utilities::group_share(random.genUniformInt(0, 1), mask, group_size))
                            axis = -axis;
                    #else
                        mpcd::Vector axis = gpu_utilities::group_share(random.genUnitVector(), mask, group_size);
                    #endif

                    float z_centre = mpc_cells.getPosition(group_cell_idx).z;
                    mpcd::Vector centre_of_mass = {},
                                 mean_velocity  = {};

                    bool constexpr conserve_L = true;

                    // calculate cells' center of mass and mean velocity, iterate in thread groups:
                    for ( mpcd::Vector* position = particle_position + prefix + threadIdx.x % group_size,
                                    * velocity = particle_velocity + prefix + threadIdx.x % group_size,
                                    * end      = particle_position + prefix + n_particles;
                                position < end;
                                position += group_size, velocity += group_size )
                    {
                        centre_of_mass += (*position - offset);
                        mean_velocity  += *velocity;
                    }

                    // reduce in thread groups using the __shfl shuffle operations:
                    centre_of_mass = gpu_utilities::group_sum(centre_of_mass, mask, group_size) * (float(1) / n_particles) + offset;
                    mean_velocity  = gpu_utilities::group_sum(mean_velocity, mask, group_size) * (float(1) / n_particles);

                    // --------------- collision:

                    // devide particles into the groups defined by the random axis and center of mass.
                    float    projection_0 = {}, // projection of the mean velocities in group0 along the axis
                            projection_1 = {};
                    uint64_t group        = {}; // bitstring storing on which side particles lie.

                    for ( int i = threadIdx.x % group_size; i < n_particles; i += group_size )
                    {
                        particle_position[prefix + i] -= centre_of_mass; // tranfer into local comoving coordinate system
                        particle_velocity[prefix + i] -= mean_velocity;

                        uint64_t const side = static_cast< uint64_t >( particle_position[prefix + i].dotProduct( axis ) < 0 );
                        group |= (side << (i / group_size)); // store on which side partile i lies.

                        if ( side ) // we only need to consider one side, the other one can be derived from it.
                            projection_0 += particle_velocity[prefix + i].dotProduct( axis );
                    }

                    int size_0 = gpu_utilities::group_sum(__popc(group), mask, group_size); // size of group0
                    int size_1 = n_particles - size_0;

                    projection_0 = gpu_utilities::group_sum(projection_0, mask, group_size);
                    projection_1 = -projection_0 / size_1;
                    projection_0 =  projection_0 / size_0;

                    // calculate the collision probability based on the dynamics of the particles
                    #if 1 // saturate:
                        float probability = 1 - __expf(scale * (projection_1 - projection_0) * size_0 * size_1);
                    #else // cutoff:
                        float probability = scale * (projection_0 - projection_1) * size_0 * size_1;
                    #endif

                    mpcd::Vector   delta_L        = {}; // cells' change in angular momentum
                    bool           collide        = gpu_utilities::group_share(random.genUniformFloat(), mask, group_size) < probability;
                    unsigned const collision_mask = __ballot_sync(mask, collide); // which groups participate in calculating the collision?

                    if ( collide )
                    {
                        float transfer_0 = projection_1 * float(size_1) / size_0, // momentum trasferred to the other side of the plane.
                              transfer_1 = projection_0 * float(size_0) / size_1;

                        float mean_0 = {}, // we assign new random velocities in the groups, but have to remove their mean to assure momentum conservation.
                            mean_1 = {};

                        InertiaTensor< float > I = {};

                        random.syncPhase();
                        for (int i = threadIdx.x % group_size; i < n_particles; i += group_size)
                        {
                            bool const side     = (group >> (i / group_size)) & 0x1;
                            auto const v_random = random.gaussianf() * thermal_velocity;

                            ( side ? mean_0 : mean_1 )    += v_random;
                            delta_L                       += particle_position[prefix + i].crossProduct( particle_velocity[prefix + i] );
                            particle_velocity[prefix + i] += axis * ( v_random - particle_velocity[prefix + i].dotProduct( axis ) + (side ? transfer_0 : transfer_1));

                            auto squares  = particle_position[prefix + i].scaledWith( particle_position[prefix + i] );
                            I            += SymmetricMatrix< float > ( { squares.y + squares.z, squares.x + squares.z, squares.x + squares.y,
                                                                        -particle_position[prefix + i].x * particle_position[prefix + i].y,
                                                                        -particle_position[prefix + i].x * particle_position[prefix + i].z,
                                                                        -particle_position[prefix + i].y * particle_position[prefix + i].z } );
                        }

                        auto v_mean_0 = axis * (gpu_utilities::group_sum(mean_0, collision_mask, group_size) / size_0), // random velocities' mean
                            v_mean_1 = axis * (gpu_utilities::group_sum(mean_1, collision_mask, group_size) / size_1);

                        for ( int i = threadIdx.x % group_size; i < n_particles; i += group_size )
                        {
                            particle_velocity[prefix + i] -= ((group >> (i / group_size)) & 0x1) ? v_mean_0 : v_mean_1; // restore groups momentum conservation
                            delta_L -= particle_position[prefix + i].crossProduct(particle_velocity[prefix + i]); // sum up angular momentum change
                        }

                        I       =     gpu_utilities::group_sum(I,       collision_mask, group_size).inverse(1e-5f);
                        delta_L = I * gpu_utilities::group_sum(delta_L, collision_mask, group_size);
                    }

                    // --------------- end collision.

                    for ( mpcd::Vector* position = particle_position + prefix + threadIdx.x % group_size,
                                    * velocity = particle_velocity + prefix + threadIdx.x % group_size,
                                    * end      = particle_position + prefix + n_particles;
                                position < end;
                                position += group_size, velocity += group_size )
                    {
                        *velocity += mean_velocity; // restore momentum conservation
                        if ( conserve_L )
                            *velocity -= position -> crossProduct( delta_L ); // restore angular momentum conservation
                        *position += centre_of_mass; // convert coordinate back into global coordinate system
                    }
                }
                __syncwarp(); // ~~~ write particles back to ram:

                for (int i = threadIdx.x; i < sum; i += 32)
                {
                    if ( particle_idx[i] != -1u )
                    {
                        particle_velocity[i].x += drag;

                        auto apply_periodic_boundaries = [&] (auto r) // does not interfere with using walls...
                        {
                            r.x = fmodf(r.x + 1.5f * volume_size.x, volume_size.x) - volume_size.x * 0.5f;
                            r.y = fmodf(r.y + 1.5f * volume_size.y, volume_size.y) - volume_size.y * 0.5f;
                            r.z = fmodf(r.z + 1.5f * volume_size.z, volume_size.z) - volume_size.z * 0.5f;
                            return r;
                        };
                        particle_position[ i ] = apply_periodic_boundaries(particle_position[ i ] - grid_shift);

                        assert(particle_position[ i ].isFinite());
                        assert(particle_velocity[ i ].isFinite());

                        // ~~~ unified write back by switching pointer for usual / ghost particles:

                        auto addr = particle_idx[i] < particles.size() ? particles.data() : ghost_particles;
                        auto cidx = mpc_cells.getIndex(particle_position[i]);

                        *(addr + particle_idx[i]) = {uint16_t(0),
                                                    static_cast<int16_t>(particle_idx [ i ] < particles.size() ? 0u : 1u),
                                                    particle_position[i],
                                                    particle_velocity[i], cidx};
                    }
                }
                __syncwarp();

                if (threadIdx.x >= active_cells) // rewind skipped cells
                    cell_idx -= stride;

                cell_idx = __shfl_sync(0xFFFFFFFF, cell_idx, threadIdx.x + active_cells); // uniform interation, "shift" processed cells out of the warp iteration.
            }

            generator[blockIdx.x * blockDim.x + threadIdx.x] = random; // save new state of the generators
        }

        // ===============================================================================================
        // Trivial extended MPCD strategy — plain multi-pass kernels, no kernel fusion, no shared memory
        // ===============================================================================================

        // Pass 1: scatter velocity + position into cells (atomicAdd)
        __global__ void scatterParticles(DeviceVector<Particle> particles, DeviceVolumeContainer<MPCCell> mpc_cells)
        {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (auto end = particles.size(); idx < end; idx += stride) {
                auto particle = gpu_utilities::texture_load(particles.data() + idx);
                mpc_cells[particle.cell_idx].add(particle);
            }
        }

        // Pass 2: normalise mean_velocity and centre_of_mass by density
        __global__ void averageCells(DeviceVolumeContainer<MPCCell> mpc_cells)
        {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (; idx < mpc_cells.size(); idx += stride)
                if (mpc_cells[idx].density > 0)
                    mpc_cells[idx].average();
        }

        // Pass 3: draw one random splitting-plane axis per cell (discrete set, steps=4)
        __global__ void generateAxes(DeviceVector<Xoshiro128Plus> generator,
                                     DeviceVector<mpcd::Vector>   rotation_axes)
        {
            int  idx    = blockIdx.x * blockDim.x + threadIdx.x;
            int  stride = blockDim.x * gridDim.x;
            auto random = generator[idx];

            constexpr int steps = 4;

            for (int cell = idx; cell < (int)rotation_axes.size(); cell += stride) {
                int select  = random.genUniformInt(0, steps * (steps - 1));
                int phi_i   = select % steps + 1;
                int theta_i = select / steps + 1;

                float theta, phi;
                if ((theta_i % 2) && (phi_i % 2))
                    theta = theta_i == 1 ? 0.95531661812450927816f
                                        : float(M_PI) - 0.95531661812450927816f;
                else
                    theta = theta_i * (float(M_PI) / steps);

                phi = phi_i * (float(M_PI) / steps);

                mpcd::Vector axis = { __sinf(theta) * __cosf(phi),
                                      __sinf(theta) * __sinf(phi),
                                      __cosf(theta) };

                if (random.genUniformInt(0, 1))
                    axis = -axis;

                rotation_axes[cell] = axis;
            }

            generator[blockIdx.x * blockDim.x + threadIdx.x] = random;
        }

        /**
        *  @brief Extended collision — one thread per cell, sequential particle loops via the uniform_list.
        *
        *         Fully decomposed scatter/reduce (like trivial SRD) breaks down here for two reasons:
        *         (i)  the stochastic gate requires per-cell projection sums + group sizes that are
        *              not pre-stored — four extra scalars per cell that would bloat the struct.
        *         (ii) thermalisation with exact group momentum conservation is circular: the mean of
        *              the thermal draws is needed before they are applied.  Splitting this into two
        *              kernel passes would require either per-particle intermediate storage or saving
        *              the full RNG state per cell between launches.
        *
        *         The uniform_list is therefore the natural unit of work: each thread has direct
        *         O(n) access to its cell's particles and handles both loops locally.  Loop 2a
        *         accumulates the thermal mean; the RNG state is saved before loop 2a and replayed
        *         in loop 2b so the same gaussian draws are regenerated without any extra storage.
        *
        *         This is exactly why the fused extendedCollision was designed in the first place —
        *         it eliminates the two global-memory round-trips for particle velocities that this
        *         baseline pays, keeping everything in shared memory and warp registers.
        *
        *         Ghost particles (wall layers) are not supported in this trivial variant.
        */
        __global__ void collideAndAccumulate(
                DeviceVector<Particle>                          particles,
                DeviceVolumeContainer<MPCCell>                  mpc_cells,
                DeviceVector<Xoshiro128Plus>                    generator,
                DeviceVector<uint32_t>                          uniform_counter,
                DeviceVector<uint32_t>                          uniform_list,
                DeviceVector<mpcd::Vector>                      rotation_axes,
                DeviceVector<mpcd::InertiaTensor<mpcd::Float>>  inertia_tensors,
                DeviceVector<mpcd::Vector>                      angular_momentum,
                mpcd::Float                                     thermal_velocity)
        {
            int      idx    = blockIdx.x * blockDim.x + threadIdx.x;
            int      stride = blockDim.x * gridDim.x;
            auto     random = generator[idx];
            int      n_cells           = (int)mpc_cells.size();
            uint32_t cell_lookup_size  = uniform_list.size() / uniform_counter.size();
            float const scale          = 0.01f;

            for (int cell_idx = idx; cell_idx < n_cells; cell_idx += stride) {
                int n_particles = (int)::min(cell_lookup_size, uniform_counter[cell_idx]);
                if (n_particles < 2) continue;

                auto CoM      = mpc_cells[cell_idx].centre_of_mass;
                auto mean_vel = mpc_cells[cell_idx].mean_velocity;
                auto axis     = rotation_axes[cell_idx];

                // Loop 1: group projections and sizes (computed on the fly, not stored)
                float proj_0 = 0.f, proj_1 = 0.f;
                int   size_0 = 0,   size_1 = 0;

                for (int i = 0; i < n_particles; ++i) {
                    uint32_t pidx = uniform_list[cell_idx + i * n_cells];
                    auto p        = gpu_utilities::texture_load(particles.data() + pidx);
                    auto pos_cm   = p.position - CoM;
                    auto vel_cm   = p.velocity - mean_vel;
                    bool side     = pos_cm.dotProduct(axis) < 0;
                    float proj    = vel_cm.dotProduct(axis);
                    if (side) { proj_0 += proj; ++size_0; }
                    else      { proj_1 += proj; ++size_1; }
                }

                if (size_0 == 0 || size_1 == 0) continue;

                proj_0 /= size_0;
                proj_1 /= size_1;

                float probability = 1.f - __expf(scale * (proj_1 - proj_0) * size_0 * size_1);

                if (random.genUniformFloat() >= probability) continue;

                float transfer_0 = proj_1 * float(size_1) / float(size_0);
                float transfer_1 = proj_0 * float(size_0) / float(size_1);

                // Loop 2a: draw thermal velocities, accumulate mean corrections,
                //          inertia tensor, and pre-collision angular momentum
                auto rng_save = random;
                float mean_r0 = 0.f, mean_r1 = 0.f;
                mpcd::InertiaTensor<mpcd::Float> I = {};
                mpcd::Vector delta_L_before = {};

                for (int i = 0; i < n_particles; ++i) {
                    uint32_t pidx = uniform_list[cell_idx + i * n_cells];
                    auto p        = gpu_utilities::texture_load(particles.data() + pidx);
                    auto pos_cm   = p.position - CoM;
                    auto vel_cm   = p.velocity - mean_vel;
                    bool side     = pos_cm.dotProduct(axis) < 0;
                    float v_rand  = random.gaussianf() * thermal_velocity;

                    (side ? mean_r0 : mean_r1) += v_rand;
                    delta_L_before += pos_cm.crossProduct(vel_cm);

                    auto sq = pos_cm.scaledWith(pos_cm);
                    I += mpcd::SymmetricMatrix<mpcd::Float>(
                            sq.y + sq.z, sq.x + sq.z, sq.x + sq.y,
                            -pos_cm.x * pos_cm.y,
                            -pos_cm.x * pos_cm.z,
                            -pos_cm.y * pos_cm.z);
                }

                mean_r0 /= size_0;
                mean_r1 /= size_1;

                // Loop 2b: replay thermal draws, apply momentum conservation,
                //          write partial velocity, accumulate post-collision angular momentum
                random = rng_save;
                mpcd::Vector delta_L_after = {};

                for (int i = 0; i < n_particles; ++i) {
                    uint32_t pidx = uniform_list[cell_idx + i * n_cells];
                    auto p        = gpu_utilities::texture_load(particles.data() + pidx);
                    auto pos_cm   = p.position - CoM;
                    auto vel_cm   = p.velocity - mean_vel;
                    bool side     = pos_cm.dotProduct(axis) < 0;
                    float v_rand  = random.gaussianf() * thermal_velocity; // same sequence as loop 2a

                    float transfer = side ? transfer_0 : transfer_1;
                    float v_corr   = side ? mean_r0    : mean_r1;

                    auto vel_cm_final = vel_cm - axis * vel_cm.dotProduct(axis)
                                       + axis * (v_rand + transfer - v_corr);

                    delta_L_after += pos_cm.crossProduct(vel_cm_final);

                    // partial velocity: lab frame, angular momentum correction applied in pass 6
                    particles[pidx].velocity = mean_vel + vel_cm_final;
                }

                angular_momentum[cell_idx] = delta_L_before - delta_L_after;
                inertia_tensors [cell_idx] = I;
            }

            generator[blockIdx.x * blockDim.x + threadIdx.x] = random;
        }

        // Pass 5: invert inertia tensor and compute per-cell angular momentum correction
        __global__ void invertAndRotate(
                DeviceVector<mpcd::InertiaTensor<mpcd::Float>>  inertia_tensors,
                DeviceVector<mpcd::Vector>                      angular_momentum,
                DeviceVector<mpcd::Vector>                      rotated_angular_momentum)
        {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (; idx < inertia_tensors.size(); idx += stride)
                rotated_angular_momentum[idx] = inertia_tensors[idx].inverse(1e-5f) * angular_momentum[idx];
        }

        // Pass 6: apply angular momentum correction + drag + position update
        __global__ void applyAngularMomentum(
                DeviceVector<Particle>          particles,
                DeviceVolumeContainer<MPCCell>  mpc_cells,
                DeviceVector<mpcd::Vector>      rotated_angular_momentum,
                mpcd::Vector                    grid_shift,
                mpcd::Vector                    volume_size,
                mpcd::Float                     drag,
                mpcd::Float                     delta_t)
        {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            auto apply_periodic_boundaries = [&](mpcd::Vector r) {
                r.x = fmodf(r.x + 1.5f * volume_size.x, volume_size.x) - volume_size.x * 0.5f;
                r.y = fmodf(r.y + 1.5f * volume_size.y, volume_size.y) - volume_size.y * 0.5f;
                r.z = fmodf(r.z + 1.5f * volume_size.z, volume_size.z) - volume_size.z * 0.5f;
                return r;
            };

            for (auto end = particles.size(); idx < end; idx += stride) {
                auto particle   = particles[idx];
                auto cell_idx   = particle.cell_idx;
                auto CoM        = mpc_cells[cell_idx].centre_of_mass;
                auto correction = rotated_angular_momentum[cell_idx];
                auto pos_cm     = particle.position - CoM;

                particle.velocity  -= pos_cm.crossProduct(correction);
                particle.velocity.x += drag * delta_t;
                particle.position   = apply_periodic_boundaries(particle.position - grid_shift);
                particle.cell_idx   = mpc_cells.getIndex(particle.position);

                particles[idx] = particle;
            }
        }

    } // namespace ext


    // =============================================================================================
    // Trivial extended MPCD strategy — plain multi-pass kernels, no kernel fusion, no shared memory
    // =============================================================================================


    void ExtendedMPCStrategy::collideParticles() {
        ext::fusedCollision<<<ctx.cuda_config.sharing_blocks, 32, ctx.cuda_config.shared_bytes>>>(
                                    ctx.particles, ctx.mpc_cells, ctx.generator.data(), ctx.grid_shift,
                                    {ctx.parameters.volume_size[0], ctx.parameters.volume_size[1], ctx.parameters.volume_size[2]},
                                    {ctx.parameters.periodicity[0], ctx.parameters.periodicity[1], ctx.parameters.periodicity[2]},
                                    ctx.parameters.delta_t, ctx.parameters.drag, ctx.parameters.thermal_velocity, ctx.parameters.n,
                                    ctx.uniform_counter, ctx.uniform_list, ctx.cuda_config.shared_bytes);
        error_check("collision_step");
    }

    void SortingExtendedMPCStrategy::sortParticles() {
        CollisionStrategy::sortParticles(); // bypass TrivialExtendedMPCStrategy's no-op, call counting sort
    }

    TrivialExtendedMPCStrategy::TrivialExtendedMPCStrategy(BackendContext& ctx)
        : CollisionStrategy(ctx)
        , inertia_tensors(ctx.mpc_cells.size())
        , angular_momentum(ctx.mpc_cells.size())
        , rotated_angular_momentum(ctx.mpc_cells.size())
        , rotation_axes(ctx.mpc_cells.size())
    {}

    void TrivialExtendedMPCStrategy::collideParticles() {
        // cells are cleared by CudaBackendImpl::collisionStep() before dispatch
        ext::scatterParticles<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(
            ctx.particles, ctx.mpc_cells);
        error_check("ext_trivial_scatter_full");

        ext::averageCells<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(
            ctx.mpc_cells);
        error_check("ext_trivial_average_full");

        ext::generateAxes<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(
            ctx.generator, rotation_axes);
        error_check("ext_trivial_generate_axes");

        // zero per-cell intermediates before accumulation
        inertia_tensors.set(0);
        angular_momentum.set(0);
        rotated_angular_momentum.set(0);

        ext::collideAndAccumulate<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(
            ctx.particles, ctx.mpc_cells, ctx.generator,
            ctx.uniform_counter, ctx.uniform_list,
            rotation_axes, inertia_tensors, angular_momentum,
            ctx.parameters.thermal_velocity);
        error_check("ext_trivial_collide_accumulate");

        ext::invertAndRotate<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(
            inertia_tensors, angular_momentum, rotated_angular_momentum);
        error_check("ext_trivial_invert_rotate");

        ext::applyAngularMomentum<<<ctx.cuda_config.block_count, ctx.cuda_config.block_size>>>(
            ctx.particles, ctx.mpc_cells, rotated_angular_momentum, ctx.grid_shift,
            {ctx.parameters.volume_size[0], ctx.parameters.volume_size[1], ctx.parameters.volume_size[2]},
            ctx.parameters.drag, ctx.parameters.delta_t);
        error_check("ext_trivial_apply_angular_momentum");
    }

} // namespace mpcd::cuda
