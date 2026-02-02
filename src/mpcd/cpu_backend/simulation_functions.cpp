#include <vector>

#include "cpu_backend/simulation_functions.hpp"

namespace mpcd::cpu {
    void distribute_particles(std::vector<Particle>& particles, VolumeContainer<MPCCell>& mpc_cells, SimulationParameters const& params, Xoshiro128Plus& random) {
        mpcd::Vector scale = {params.volume_size[0], params.volume_size[1], params.volume_size[2]};
        mpcd::IntVector perio = {params.periodicity[0], params.periodicity[1], params.periodicity[2]};
        scale = scale - 2 * (1 - perio);
        auto channel_radius2 = (params.volume_size[0] - 2) * (params.volume_size[0] - 2) * 0.25f;

        // Initialize particles
        for (size_t i = 0; i < particles.size(); ++i) {
            Particle particle = {};
            bool          replace;

            do {
                replace  = false;
                particle.position = {random.genUniformFloat(), random.genUniformFloat(), random.genUniformFloat()}; // uniform on the unit cube.
                particle.position = (particle.position - mpcd::Float(0.5)).scaledWith(scale);  // rescale to the simulation volume

                if (params.experiment == ExperimentType::channel)
                    replace = replace or ((particle.position.z * particle.position.z + particle.position.y * particle.position.y) > channel_radius2);

            } while (replace);

            particle.velocity = random.maxwell_boltzmann() * params.thermal_velocity;
            particle.cell_idx = mpc_cells.get_index(particle.position);

            particles[i] = particle;
        }
    }

    /*
     * MPC translation step function
     */
    void translate_particles(std::vector<Particle>& particles, VolumeContainer<MPCCell>& mpc_cells,
                                        Xoshiro128Plus& random, mpcd::Vector grid_shift,
                                        mpcd::Vector volume_size, mpcd::IntVector periodicity, mpcd::Float delta_t,
                                        ExperimentType experiment,
                                        std::vector<uint32_t>& uniform_counter, std::vector<uint32_t>& uniform_list) {
        uint32_t cell_lookup_size = uniform_list.size() / uniform_counter.size();
        float channel_radius2 = ( volume_size.z - 2 ) * ( volume_size.z - 2 ) * 0.25f;

        auto apply_periodic_boundaries = [&] ( auto r )
        {
            r.x = fmodf(r.x + 1.5f * volume_size.x, volume_size.x) - volume_size.x * 0.5f;
            r.y = fmodf(r.y + 1.5f * volume_size.y, volume_size.y) - volume_size.y * 0.5f;
            r.z = fmodf(r.z + 1.5f * volume_size.z, volume_size.z) - volume_size.z * 0.5f; // does not interfere with using walls...
            return r;
        };

        for (int idx = 0; idx < particles.size(); ++idx) {
            auto particle = particles[idx];

            if (experiment != ExperimentType::channel) {
                if (not periodicity.z) { // if walls are present, calculate collisions
                    auto z_wall = (0.5f * volume_size.z - 1); // distance of the walls, remove one layer for ghost particles
                    auto next_z = particle.position.z + particle.velocity.z * delta_t;

                    if (fabsf(next_z) > z_wall) { // this is more safe than just calcualating the time
                        auto time_left = (next_z > 0 ? z_wall - particle.position.z : -z_wall - particle.position.z)
                                        / particle.velocity.z;

                        particle.position += particle.velocity * time_left;
                        particle.velocity = -particle.velocity; // bounce back roule, creates no-slip boundary condition
                        particle.position += particle.velocity * (delta_t - time_left);
                    } else {
                        particle.position += particle.velocity * delta_t;
                    }
                } else {
                    particle.position += particle.velocity * delta_t;
                }
            }
            else { // channel:
                particle.position += particle.velocity * delta_t; // advance particle

                if ((particle.position.z * particle.position.z + particle.position.y * particle.position.y) > channel_radius2) {
                    particle.position -= particle.velocity * delta_t; // apply correction if it left the channel
                    particle.velocity = -particle.velocity;
                }
            }

            particle.position = apply_periodic_boundaries(particle.position + grid_shift);
            particle.cell_idx = mpc_cells.get_index(particle.position);

            // make an entry in the index lookup for the cell in which the particle lies
            int slot = uniform_counter[particle.cell_idx];
            uniform_counter[particle.cell_idx] += 1;
            if ( slot < cell_lookup_size )
                uniform_list[particle.cell_idx + slot * mpc_cells.size()] = idx;
            else
                particle.position = apply_periodic_boundaries(particle.position - grid_shift);
            // if the lookup is full, the particle gets no shift because after the collision step particle are not shifted

            assert(particle.position.isFinite()); // check for error in floating point math
            assert(particle.velocity.isFinite());

            particles[idx] = particle;
        }
    }

    /*
     * SRD Collision step
     */
    void srd_collision(std::vector<Particle>& particles, VolumeContainer<MPCCell>& mpc_cells,
                                        Xoshiro128Plus& random, mpcd::Vector grid_shift,
                                        mpcd::Vector volume_size, mpcd::IntVector periodicity, mpcd::Float delta_t,
                                        mpcd::Float drag, mpcd::Float thermal_velocity, uint32_t n_density,
                                        std::vector<uint32_t>& uniform_counter, std::vector<uint32_t>& uniform_list) {
        uint32_t const cell_lookup_size   = uniform_list.size() / uniform_counter.size();
        auto     const regular_particles  = particles.size();

        auto const     shift     = fabs(grid_shift.z); // create wall's ghost particles on the fly
        unsigned const sign      = grid_shift.z > 0;
        float const    sin_alpha = sinf((M_PI * 120) / 180), // these are used to represent the SRD rotation matrix
                       cos_alpha = cosf((M_PI * 120) / 180);

        drag *= delta_t; // pressure gradient that accelerates the flow

        auto apply_periodic_boundaries = [&] (auto r) // does not interfere with using walls...
        {
            r.x = fmodf( r.x + 1.5f * volume_size.x, volume_size.x ) - volume_size.x * 0.5f;
            r.y = fmodf( r.y + 1.5f * volume_size.y, volume_size.y ) - volume_size.y * 0.5f;
            r.z = fmodf( r.z + 1.5f * volume_size.z, volume_size.z ) - volume_size.z * 0.5f;
            return r;
        };

        Particle buffer[cell_lookup_size];
        int particle_idx[cell_lookup_size];

        for (uint32_t cell_idx = 0; cell_idx < mpc_cells.size(); ++cell_idx) {
            MPCCell cell = {};
            uint32_t n_particles = (cell_idx < mpc_cells.size()) ? std::min(cell_lookup_size, uniform_counter[cell_idx]) : 0; // load the table size

            bool const    layer        = (mpc_cells.get_z_idx(cell_idx) == (volume_size.z - (sign ? 1 : 2)));
            bool const    add_ghosts   = (not periodicity.z) and ((mpc_cells.get_z_idx(cell_idx) == sign) or layer); // wall layer?
            uint32_t      added_ghosts = {};

            if (add_ghosts) // create wall's ghost particles on the fly; prepare number of ghosts
                for (int i = 0; i < n_density; ++i)
                    if ((random.genUniformFloat() > shift) xor layer xor sign)
                        ++added_ghosts;

            n_particles += added_ghosts;

            for (uint32_t i = 0; i < std::min(n_particles - added_ghosts, cell_lookup_size); ++i) {
                particle_idx[i] = uniform_list[cell_idx + i * mpc_cells.size()];
                buffer[i] = particles[particle_idx[i]];
            }

            if (add_ghosts) { // create wall's ghost particles on the fly
                auto pos = mpc_cells.get_position(cell_idx);

                for (int i = n_particles - added_ghosts; i < n_particles; ++i) {
                    float z;

                    do {
                        z = random.genUniformFloat();
                    } while (((z < shift) xor layer) xor sign);

                    buffer[i].cidx = -1;
                    buffer[i].position = Vector(random.genUniformFloat() - 0.5f, random.genUniformFloat() - 0.5f, z - 0.5f) + pos;
                    buffer[i].velocity = random.maxwell_boltzmann() * thermal_velocity;
                }
            }

            if (n_particles > 1) {
                for (uint32_t i = 0; i < n_particles; ++i)
                    cell.unlocked_add(buffer[i]);

                cell.average();
                auto axis = random.genUnitVector();

                for (uint32_t i = 0; i < n_particles; ++i) { // rotation step:
                    auto v      = buffer[i].velocity - cell.mean_velocity;
                    auto v_para = axis * (v.dotProduct(axis));
                    auto v_perp = v - v_para;

                    buffer[i].velocity = v_para + cos_alpha * v_perp + sin_alpha * v_perp.crossProduct(axis);
                }

                for (uint32_t i = 0; i < n_particles; ++i) { // finilize step:
                    buffer[i].velocity += cell.get_correction(buffer[i].position);
                    buffer[i].position = apply_periodic_boundaries(buffer[i].position - grid_shift);
                    buffer[i].velocity.x += drag;
                }
            }

            for (uint32_t i = 0; i < std::min(n_particles - added_ghosts, cell_lookup_size); ++i) {
                if (buffer[i].cidx != -1u) {
                    if (buffer[i].cidx < regular_particles) {
                        buffer[i].cell_idx = mpc_cells.get_index(buffer[i].position);
                        buffer[i].flags = 0;
                        buffer[i].cidx = 0;
                        particles[particle_idx[i]] = buffer[i];
                    }
                }
            }
        }
    }
} // namespace mpcd::cpu
