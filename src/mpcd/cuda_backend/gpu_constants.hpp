#pragma once
#include <mpcd/common/simulation_parameters.hpp>

#include "device_volume_container"
#include "mpc_cell.hpp"
#include "particle.hpp"
#include "gpu_random.hpp"

// To avoid filling the kernel's stack with function arguments, large structs are placed in the GPU's constant memorgy.
// This is especially true for the parameter_set

namespace gpu_const
{
    extern __constant__ SimulationParameters                     parameters;
    extern __constant__ DeviceVolumeContainer<MPCCell>  mpc_cells;
    extern __constant__ UnifiedVector<Particle>              particles;
    extern __constant__ DeviceVector<Xoshiro128Plus>             generator;
    extern __constant__ DeviceVector<uint32_t>                   uniform_list,
                                                                 uniform_counter;
}
