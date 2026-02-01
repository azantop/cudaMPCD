#include <mpcd/api/simulation_parameters.hpp>

#include "backend/backend.hpp"
#include "cpu_backend/cpu_backend.hpp"

namespace mpcd::cpu {

    void CPUBackend::translationStep() {
        // TODO
    }
    void CPUBackend::collisionStep() {
        // TODO
    }

    CPUBackend::CPUBackend(SimulationParameters const& params) : Backend(params),
                                                                 particles(params.N)
    {

        std::cout << "Logging output:\n Volume size: " << params.volume_size[0] << "x" << params.volume_size[1] << "x" << params.volume_size[2] << std::endl;
        std::cout << "n particles per cell: " << params.n << std::endl;

        const int64_t vol_x = static_cast<int64_t>(params.volume_size[0]);
        const int64_t vol_y = static_cast<int64_t>(params.volume_size[1]);
        const int64_t vol_z = static_cast<int64_t>(params.volume_size[2]);
        const int64_t n = static_cast<int64_t>(params.n);

        const int64_t total_cells = vol_x * vol_y * vol_z;
        const int64_t total_particles = total_cells * n;

        std::cout << "Calculated total cells: " << total_cells << std::endl;
        std::cout << "Calculated total particles: " << total_particles << std::endl;
        std::cout << "About to allocate particles vector..." << std::endl;


        mpcd::Vector scale = {params.volume_size[0], params.volume_size[1], params.volume_size[2]};
        mpcd::IntVector perio = {params.periodicity[0], params.periodicity[1], params.periodicity[2]};
        scale = scale + 2 * (1 - perio);

        // Initialize particles
        for (size_t i = 0; i < particles.size(); ++i) {
            particles[i].position = {random.uniform_float(), random.uniform_float(), random.uniform_float()}; // uniform on the unit cube.
            particles[i].position = (particles[i].position - mpcd::Float(0.5)).scaledWith(scale);  // rescale to the simulation volume
            particles[i].velocity = random.maxwell_boltzmann() * params.thermal_velocity;
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
        // TODO
    }

    void CPUBackend::getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity) {
        // TODO
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
