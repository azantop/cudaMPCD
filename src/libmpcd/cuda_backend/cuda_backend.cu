#include "cuda_backend.hpp"
#include "cuda_backend_impl.hpp"

namespace mpcd::cuda {

    // Constructor/Destructor
    CudaBackend::CudaBackend(SimulationParameters const& params)
        : Backend(params), pImpl(std::make_unique<CudaBackendImpl>(params)) {}

    CudaBackend::~CudaBackend() = default;

    // Backend interface delegation
    void CudaBackend::step(int n_steps) {
        pImpl->step(n_steps);
    }

    void CudaBackend::stepAndAccumulateSample(int n_steps) {
        pImpl->stepAndAccumulateSample(n_steps);
    }

    void CudaBackend::getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity) {
        pImpl->getSampleMean(mean_density, mean_velocity);
    }

    size_t CudaBackend::getNParticles() {
        return pImpl->getNParticles();
    }

    void CudaBackend::getParticlePositions(std::vector<float>& positions) {
        pImpl->getParticlePositions(positions);
    }

    void CudaBackend::getParticleVelocities(std::vector<float>& velocities) {
        pImpl->getParticleVelocities(velocities);
    }

    void CudaBackend::setParticlePositions(std::vector<float> const& positions) {
        pImpl->setParticlePositions(positions);
    }

    void CudaBackend::setParticleVelocities(std::vector<float> const& velocities) {
        pImpl->setParticleVelocities(velocities);
    }

    void CudaBackend::setParticles(std::vector<float> const& positions, std::vector<float> const& velocities) {
        pImpl->setParticles(positions, velocities);
    }

    // Data I/O methods
    void CudaBackend::writeSample() {
        pImpl->writeSample();
    }

    void CudaBackend::writeBackupFile() {
        pImpl->writeBackupFile();
    }

} // namespace mpcd::cuda
