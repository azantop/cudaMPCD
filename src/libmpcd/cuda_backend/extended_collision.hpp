#pragma once

#include "common/particle.hpp"
#include "common/vector_3d.hpp"
#include "common/mpc_cell.hpp"
#include "common/random.hpp"

#include "gpu_arrays.hpp"
#include "device_volume_container.hpp"

namespace mpcd::cuda {
    __global__ void extendedCollision(DeviceVector<Particle> particles,
                                    DeviceVolumeContainer<MPCCell> mpc_cells,
                                    Xoshiro128Plus* generator,
                                    mpcd::Vector grid_shift,
                                    mpcd::Vector volume_size,
                                    mpcd::IntVector periodicity,
                                    mpcd::Float delta_t,
                                    mpcd::Float drag,
                                    mpcd::Float thermal_velocity,
                                    uint32_t n_density,
                                    DeviceVector<uint32_t> uniform_counter,
                                    DeviceVector<uint32_t> uniform_list, uint32_t const shared_bytes);
} // namespace mpcd::cuda
