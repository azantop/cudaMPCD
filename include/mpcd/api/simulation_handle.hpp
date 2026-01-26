#pragma once

#include <cstddef>
#include <string>
#include <memory>
#include <vector>

#include <mpcd/api/simulation_parameters.hpp>

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

            SimulationParameters const& getParameters() const;

            void step(int n_steps); // Perform n_steps of the simulation

            // Perform n_steps of the simulation and accumulate sample data
            void stepAndAccumulateSample(int n_steps);
            // Retrieve the mean density and velocity from the accumulated samples:
            void getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity);

            // Retrieve the MPC particle positions and velocities from the simulation:
            void getParticlePositions(std::vector<float>& positions) const;
            void getParticleVelocities(std::vector<float>& velocities) const;
            // Set the MPC particle positions and velocities in the simulation:
            void setParticlePositions(std::vector<float> const& positions);
            void setParticleVelocities(std::vector<float> const& velocities);
            void setParticles(std::vector<float> const& positions, std::vector<float> const& velocities);

        private:
            // Bridge pattern; pointer to the backend implementation
            std::unique_ptr<Backend> backend_;
        };
    } // namespace api
} // namespace mpcd
