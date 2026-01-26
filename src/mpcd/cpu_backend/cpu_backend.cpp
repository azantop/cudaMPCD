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

    CPUBackend::CPUBackend(SimulationParameters const& params) : Backend(params) {}

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
        // TODO
    }

    void CPUBackend::getParticleVelocities(std::vector<float>& velocities) {
        // TODO
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
