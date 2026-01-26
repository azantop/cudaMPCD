#include <mpcd/api/simulation_handle.hpp>
#include <mpcd/common/simulation_parameters.hpp>

#include "backend/backend.hpp"
#include "common/backend_factory.hpp"

namespace mpcd::api {

    SimulationHandle::SimulationHandle(SimulationParameters const& params, std::string backend_type) : backend_(create_backend(params, backend_type)) {}

    SimulationHandle::~SimulationHandle() = default;

    SimulationHandle::SimulationHandle(SimulationHandle&&) noexcept = default;
    SimulationHandle& SimulationHandle::operator=(SimulationHandle&&) noexcept = default;

    void SimulationHandle::step(int n_steps) {
        backend_->step(n_steps);
    }

    void SimulationHandle::stepAndAccumulateSample(int n_steps) {
        backend_->stepAndAccumulateSample(n_steps);
    }

    void SimulationHandle::getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity) {
        backend_->getSampleMean(mean_density, mean_velocity);
    }

    void SimulationHandle::getParticlePositions(std::vector<float>& positions) const {
        backend_->getParticlePositions(positions);
    }

    void SimulationHandle::getParticleVelocities(std::vector<float>& velocities) const {
        backend_->getParticleVelocities(velocities);
    }

    void SimulationHandle::setParticlePositions(std::vector<float> const& positions) {
        backend_->setParticlePositions(positions);
    }

    void SimulationHandle::setParticleVelocities(std::vector<float> const& velocities) {
        backend_->setParticleVelocities(velocities);
    }

    void SimulationHandle::setParticles(std::vector<float> const& positions, std::vector<float> const& velocities) {
        backend_->setParticles(positions, velocities);
    }
} // namespace mpcd::api
