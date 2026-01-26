#pragma once

#include <array>
#include <string>

namespace mpcd {
    enum class ExperimentType {
        standart,
        channel,
    };

    enum class MPCDAlgorithm {
        srd,
        extended,
    };

    struct SimulationParameters
    {
        ExperimentType experiment;
        MPCDAlgorithm  algorithm;

        std::array<int, 3>   periodicity,
                            domain_periodicity,
                            direction_is_split;
        std::array<float, 3> volume_size;

        unsigned int N, // number of particles
                    n, // number of particles per cell
                    device_id = 0;

        float        delta_t, // SRD time step
                    temperature,
                    thermal_velocity,
                    drag,
                    thermal_sigma;

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
