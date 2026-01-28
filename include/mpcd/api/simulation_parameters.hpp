#pragma once

#include <array>
#include <string>

namespace mpcd {
    enum class ExperimentType {
        standart, // Simulate in standart 3D periodic domain
        channel, // Simulate in 2D periodic channel domain
    };

    enum class MPCDAlgorithm {
        srd, // Simulate using SRD algorithm
        extended, // Simulate using extended algorithm for incompressible flow
    };

    struct SimulationParameters
    {
        ExperimentType experiment;
        MPCDAlgorithm  algorithm;

        std::array<int, 3>   periodicity;
        std::array<float, 3> volume_size;

        unsigned int N, // number of particles
                     n, // number of particles per cell
                     device_id = 0;

        float       delta_t, // SRD time step
                    temperature, // fluid temperature
                    thermal_velocity,
                    drag, // pressure drag force applied for poiseuille flow
                    thermal_sigma;

        // parameters used in c++ simulator routine:
        unsigned int equilibration_steps,
                     steps,
                     sample_every,
                     average_samples;
        bool         sample_fluid;
        bool         write_backup,
                     read_backup;

        std::string  output_directory,
                     load_directory;

        SimulationParameters() = default;
        SimulationParameters(std::string const& load_dir, bool read_only=false);
    };
} // namespace mpcd
