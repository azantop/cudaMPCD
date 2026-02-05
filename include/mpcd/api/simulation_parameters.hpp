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
        ExperimentType experiment = ExperimentType::standart;
        MPCDAlgorithm  algorithm = MPCDAlgorithm::extended;

        std::array<int, 3>   periodicity = {1, 1, 1};
        std::array<float, 3> volume_size = {10.0f, 10.0f, 10.0f};

        unsigned int N, // number of particles
                     n = 10, // number of particles per cell
                     device_id = 0;

        float       delta_t = 0.01f, // SRD time step
                    temperature = 1.0f, // fluid temperature
                    thermal_velocity = 1.0f,
                    drag = 0.0f; // pressure drag force applied for poiseuille flow

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
