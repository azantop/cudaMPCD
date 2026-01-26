#pragma once

#include <cstddef>
#include <mpcd/api/simulation_parameters.hpp>

namespace mpcd {
    enum class ProbingType {
        snapshots_only,
        start_accumulating,
        accumulate,
        finish_accumulation
    };

    inline bool do_sampling(size_t const& time_step, SimulationParameters const& parameters) {
        return !(time_step % parameters.sample_every);
    }

    inline ProbingType what_to_do(int const& time_step, SimulationParameters const& parameters) {
        if ( parameters.average_samples == 1 or time_step == -1 )
            return ProbingType::snapshots_only;

        size_t at = ((time_step / parameters.sample_every) % parameters.average_samples);

        if (at == 1)
            return ProbingType::start_accumulating;
        else if (at == 0)
            return ProbingType::finish_accumulation;
        else
            return ProbingType::accumulate;
    }
} // namespace mpcd::cuda
