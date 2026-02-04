#include <mpcd/api/simulation_parameters.hpp>

#include "backend/backend.hpp"
#include "cpu_backend/cpu_backend.hpp"
#include "cpu_backend/simulation_functions.hpp"

namespace mpcd::cpu {

    CPUBackend::CPUBackend(SimulationParameters const& params) : Backend(params),
                                                                 particles(params.N),
                                                                 mpc_cells({params.volume_size[0], params.volume_size[1], params.volume_size[2]}),
                                                                 cell_states(static_cast<size_t>(params.volume_size[0] * params.volume_size[1] * params.volume_size[2])),
                                                                 kahan_c(cell_states.size()),
                                                                 uniform_list(mpc_cells.size() * 4 * params.n),
                                                                 uniform_counter(mpc_cells.size()),
                                                                 step_counter(0),
                                                                 resort_rate(1000),
                                                                 sampling_state(SAMPLING_COMPLETED),
                                                                 sample_counter(0)
    {
        distribute_particles(particles, mpc_cells, params, random);
    }

    void CPUBackend::translationStep() {
        grid_shift = {random.genUniformFloat() - Float(0.5),
                      random.genUniformFloat() - Float(0.5),
                      random.genUniformFloat() - Float(0.5)};

        uniform_counter.assign(uniform_counter.size(), 0);
        translate_particles(particles, mpc_cells, random, grid_shift,
                           {parameters.volume_size[0], parameters.volume_size[1], parameters.volume_size[2]},
                           {parameters.periodicity[0], parameters.periodicity[1], parameters.periodicity[2]},
                           parameters.delta_t, parameters.experiment, uniform_counter, uniform_list);
    }

    void CPUBackend::collisionStep() {
        mpc_cells.set({});

        switch (parameters.algorithm) {
            case MPCDAlgorithm::srd:
                srd_collision(particles, mpc_cells, random, grid_shift,
                              {parameters.volume_size[0], parameters.volume_size[1], parameters.volume_size[2]},
                              {parameters.periodicity[0], parameters.periodicity[1], parameters.periodicity[2]},
                              parameters.delta_t, parameters.drag, parameters.thermal_velocity,
                              parameters.n, uniform_counter, uniform_list);
                break;
            case MPCDAlgorithm::extended:
                extended_collision(particles, mpc_cells, random, grid_shift,
                                       {parameters.volume_size[0], parameters.volume_size[1], parameters.volume_size[2]},
                                       {parameters.periodicity[0], parameters.periodicity[1], parameters.periodicity[2]},
                                       parameters.delta_t, parameters.drag, parameters.thermal_velocity,
                                       parameters.n, uniform_counter, uniform_list);
                break;
            default:
                throw std::runtime_error("Unsupported MPCD algorithm");
        }
    }

    void CPUBackend::writeSample() {
        // TODO
    }
    void CPUBackend::writeBackupFile() {
        // TODO
    }

    void CPUBackend::step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) {
            translationStep();
            collisionStep();
        }
    }

    void CPUBackend::stepAndAccumulateSample(int n_steps) {
        for (int i = 0; i < n_steps; ++i) {
            step(1);

            mpc_cells.set({}); // Clear cells
            for (auto& particle : particles) {
                mpc_cells[particle.position].density += 1;
                mpc_cells[particle.position].mean_velocity += particle.velocity;
            }

            for (auto& cell : mpc_cells)
                cell.average_reduce_only();

            if (sampling_state == SAMPLING_COMPLETED) {
                sampling_state = SAMPLING_IN_PROGRESS;
                sample_counter = 0;
                kahan_c.assign(kahan_c.size(), {});
                for (int j = 0; j < mpc_cells.size(); ++j) {
                    cell_states[j].density = mpc_cells[j].density;
                    cell_states[j].mean_velocity = mpc_cells[j].mean_velocity;
                }
            } else {
                sample_counter++;
                for (int j = 0; j < mpc_cells.size(); ++j) {
                    {
                        auto y = mpc_cells[j].density - kahan_c[j].density;
                        auto t = cell_states[j].density + y;
                        kahan_c[j].density = (t - cell_states[j].density) - y;
                        cell_states[j].density = t;
                    }
                    auto y = mpc_cells[j].mean_velocity - kahan_c[j].mean_velocity;
                    auto t = cell_states[j].mean_velocity + y;
                    kahan_c[j].mean_velocity = (t - cell_states[j].mean_velocity) - y;
                    cell_states[j].mean_velocity = t;
                }
            }
        }
    }

    void CPUBackend::getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity) {
        mean_density.resize(cell_states.size());
        mean_velocity.resize(cell_states.size() * 3);
        auto inverse = 1.0 / sample_counter;
        for (size_t i = 0; i < cell_states.size(); ++i) {
            mean_density[i] = cell_states[i].density * inverse;
            mean_velocity[i * 3 + 0] = cell_states[i].mean_velocity.x * inverse;
            mean_velocity[i * 3 + 1] = cell_states[i].mean_velocity.y * inverse;
            mean_velocity[i * 3 + 2] = cell_states[i].mean_velocity.z * inverse;
        }
        sampling_state = SAMPLING_COMPLETED;
    }

    size_t CPUBackend::getNParticles() {
        return parameters.N;
    }

    void CPUBackend::getParticlePositions(std::vector<float>& positions) {
        positions.resize(particles.size() * 3);
        for (size_t i = 0; i < particles.size(); ++i) {
            positions[3*i + 0] = particles[i].position.x;
            positions[3*i + 1] = particles[i].position.y;
            positions[3*i + 2] = particles[i].position.z;
        }
    }

    void CPUBackend::getParticleVelocities(std::vector<float>& velocities) {
        velocities.resize(particles.size() * 3);
        for (size_t i = 0; i < particles.size(); ++i) {
            velocities[3*i + 0] = particles[i].velocity.x;
            velocities[3*i + 1] = particles[i].velocity.y;
            velocities[3*i + 2] = particles[i].velocity.z;
        }
    }

    void CPUBackend::setParticlePositions(std::vector<float> const& positions) {
        // TODO
    }

    void CPUBackend::setParticleVelocities(std::vector<float> const& velocities) {
        // TODO
    }

    void CPUBackend::setParticles(std::vector<float> const& positions, std::vector<float> const& velocities) {
        // TODO
    }
} // namespace mpcd::cuda
