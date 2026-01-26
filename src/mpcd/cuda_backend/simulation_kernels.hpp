#pragma once
#include <mpcd/common/simulation_parameters.hpp>

#include "common/vector_3d.hpp"
#include "common/particle.hpp"
#include "gpu_arrays.hpp"
#include "device_volume_container.hpp"
#include "mpc_cell.hpp"
#include "gpu_random.hpp"

namespace mpcd::cuda {
    namespace initialize
    {
        //__global__ void srd_cells(SimulationParameters parameters, DeviceVolumeContainer<MPCCell> mpc_cells);

        __global__ void seedRandomNumberGenerators(DeviceVector<Xoshiro128Plus> generator, DeviceVector<uint64_t> seed);

        __global__ void distributeParticles( DeviceVector<Particle> particles, DeviceVolumeContainer<MPCCell> mpc_cells,
                                            DeviceVector<Xoshiro128Plus> generator, math::Vector grid_shift,
                                            math::Vector volume_size, math::IntVector periodicity, math::Float thermal_velocity,
                                            ExperimentType experiment_type,
                                            uint32_t start);
    }  //  namespace initialize

    __global__ void translateParticles( DeviceVector<Particle> particles, DeviceVolumeContainer<MPCCell> mpc_cells,
                                        DeviceVector<Xoshiro128Plus> generator, math::Vector grid_shift,
                                        math::Vector volume_size, math::IntVector periodicity, math::Float delta_t,
                                        ExperimentType experiment_type,
                                        DeviceVector<uint32_t> uniform_counter, DeviceVector<uint32_t> uniform_list);

    __global__ void srdCollision( DeviceVector<Particle> particles, DeviceVolumeContainer<MPCCell> mpc_cells,
                                Xoshiro128Plus* generator,
                                math::Vector grid_shift, math::Vector volume_size, math::IntVector periodicity,
                                math::Float delta_t, math::Float drag, math::Float thermal_velocity,
                                uint32_t n_density, DeviceVector<uint32_t> uniform_counter,
                                DeviceVector<uint32_t> uniform_list, uint32_t const shared_bytes);

    namespace sampling
    {
        __global__ void addParticles(SimulationParameters parameters, DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<Particle> particles );
        __global__ void averageCells(DeviceVolumeContainer<MPCCell> mpc_cells);
        __global__ void snapshot(DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<FluidState> cell_states);
        __global__ void accumulate(DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<FluidState> cell_states);
        __global__ void finish(size_t n_samples, DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<FluidState> cell_states);
    }
} // namespace mpcd::cuda
