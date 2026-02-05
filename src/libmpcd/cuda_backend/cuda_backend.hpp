#pragma once

#include <memory>
#include <mpcd/api/simulation_parameters.hpp>

#include "common/backend.hpp"

namespace mpcd::cuda {

    class CudaBackendImpl;

    class CudaBackend : public Backend
    {
        void translationStep();  // SRD streaming step
        void collisionStep();    // SRD collision step

        std::unique_ptr<CudaBackendImpl> pImpl;

        public:

        // routines:
        CudaBackend(SimulationParameters const&);  // initialization
        ~CudaBackend();  // cleanup

        // data io:
        void writeSample();
        void writeBackupFile();

        // backend overrides:
        void step(int n_steps) override;
        void stepAndAccumulateSample(int n_steps) override;
        void getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity) override;
        size_t getNParticles() override;
        void getParticlePositions(std::vector<float>& positions) override;
        void getParticleVelocities(std::vector<float>& velocities) override;
        void setParticlePositions(std::vector<float> const& positions) override;
        void setParticleVelocities(std::vector<float> const& velocities) override;
        void setParticles(std::vector<float> const& positions, std::vector<float> const& velocities) override;
    };
} // namespace mpcd::cuda
