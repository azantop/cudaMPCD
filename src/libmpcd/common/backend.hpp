#pragma once
#include <vector>

#include <mpcd/api/simulation_parameters.hpp>

namespace mpcd {

    /*
     * @brief Abstract backend interface for MPCD simulations.
     */
    struct Backend {
        virtual void step(int n_steps) = 0;

        virtual void stepAndAccumulateSample(int n_steps) = 0;
        virtual void getSampleMean(std::vector<float>& mean_density, std::vector<float>& mean_velocity) = 0;

        virtual size_t getNParticles() = 0;
        virtual void getParticlePositions(std::vector<float>& positions) = 0;
        virtual void getParticleVelocities(std::vector<float>& velocities) = 0;
        virtual void setParticlePositions(std::vector<float> const& positions) = 0;
        virtual void setParticleVelocities(std::vector<float> const& velocities) = 0;
        virtual void setParticles(std::vector<float> const& positions, std::vector<float> const& velocities) = 0;

        explicit Backend(SimulationParameters const& p) : parameters(p) {}
        virtual ~Backend() = default;

        SimulationParameters const& getParameters() const { return parameters; }

        protected:
        SimulationParameters     parameters;
    };
}
