#pragma once

#include "common/particle.hpp"
#include "common/vector_3d.hpp"

#include "gpu_arrays.hpp"
#include "device_volume_container.hpp"
#include "mpc_cell.hpp"
#include "gpu_random.hpp"

namespace mpcd::cuda {
    __global__ void extendedCollision(DeviceVector<Particle> particles,
                                    DeviceVolumeContainer<MPCCell> mpc_cells,
                                    Xoshiro128Plus* generator,
                                    math::Vector grid_shift,
                                    math::Vector volume_size,
                                    math::IntVector periodicity,
                                    math::Float delta_t,
                                    math::Float drag,
                                    math::Float thermal_velocity,
                                    uint32_t n_density,
                                    DeviceVector<uint32_t> uniform_counter,
                                    DeviceVector<uint32_t> uniform_list, uint32_t const shared_bytes);
} // namespace mpcd::cuda
