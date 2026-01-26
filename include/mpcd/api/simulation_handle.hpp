#pragma once

#include <cstddef>
#include <string>
#include <memory>
#include <vector>

#include <mpcd/common/simulation_parameters.hpp>

namespace mpcd {
    class Backend;

    namespace api {
        class SimulationHandle {
        public:
            SimulationHandle(SimulationParameters const&, std::string backend_type);
            ~SimulationHandle();

            SimulationHandle(SimulationHandle&&) noexcept;
            SimulationHandle& operator=(SimulationHandle&&) noexcept;

            SimulationHandle(const SimulationHandle&) = delete;
            SimulationHandle& operator=(const SimulationHandle&) = delete;

            void step(int n_steps);

            void stepAndAccumulateSample(int n_steps);
            void getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity);

            void getParticlePositions(std::vector<float>& positions) const;
            void getParticleVelocities(std::vector<float>& velocities) const;
            void setParticlePositions(std::vector<float> const& positions);
            void setParticleVelocities(std::vector<float> const& velocities);
            void setParticles(std::vector<float> const& positions, std::vector<float> const& velocities);

        private:
            std::unique_ptr<Backend> backend_;    // opaque ownership
        };
    } // namespace api
} // namespace mpcd
