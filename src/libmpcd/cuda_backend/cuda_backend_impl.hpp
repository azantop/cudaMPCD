#pragma once

#include <memory>
#include <vector>

#include "backend_context.hpp"
#include "collision_strategy.hpp"

namespace mpcd::cuda {

    class CudaBackendImpl
    {
        BackendContext                     ctx;
        std::unique_ptr<CollisionStrategy> strategy;

        enum SamplingState {
            SAMPLING_IN_PROGRESS,
            SAMPLING_COMPLETED
        };
        SamplingState sampling_state;
        size_t        sample_counter;

        void translationStep();
        void collisionStep();
        void sortParticles();

    public:
        CudaBackendImpl(SimulationParameters const&);
        ~CudaBackendImpl();

        void writeSample();
        void writeBackupFile();

        void step(int n_steps);
        void stepAndAccumulateSample(int n_steps);
        void getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity);
        size_t getNParticles();
        void getParticlePositions(std::vector<float>& positions);
        void getParticleVelocities(std::vector<float>& velocities);
        void setParticlePositions(std::vector<float> const& positions);
        void setParticleVelocities(std::vector<float> const& velocities);
        void setParticles(std::vector<float> const& positions, std::vector<float> const& velocities);
    };

} // namespace mpcd::cuda
