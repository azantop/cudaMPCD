#pragma once
#include <mpcd/api/simulation_parameters.hpp>

#include "common/vector_3d.hpp"
#include "common/particle.hpp"
#include "common/mpc_cell.hpp"
#include "common/random.hpp"
#include "gpu_arrays.hpp"
#include "device_volume_container.hpp"

namespace mpcd::cuda {
    namespace initialize
    {
        __global__ void seedRandomNumberGenerators(DeviceVector<Xoshiro128Plus> generator, DeviceVector<uint64_t> seed);

        // Randomly distribute particles in the simulation domain
        __global__ void distributeParticles( DeviceVector<Particle> particles, DeviceVolumeContainer<MPCCell> mpc_cells,
                                            DeviceVector<Xoshiro128Plus> generator, mpcd::Vector grid_shift,
                                            mpcd::Vector volume_size, mpcd::IntVector periodicity, mpcd::Float thermal_velocity,
                                            ExperimentType experiment_type,
                                            uint32_t start);
    }  //  namespace initialize

    // MPC streaming step kernel
    __global__ void translateParticles( DeviceVector<Particle> particles, DeviceVolumeContainer<MPCCell> mpc_cells,
                                        DeviceVector<Xoshiro128Plus> generator, mpcd::Vector grid_shift,
                                        mpcd::Vector volume_size, mpcd::IntVector periodicity, mpcd::Float delta_t,
                                        ExperimentType experiment_type,
                                        DeviceVector<uint32_t> uniform_counter, DeviceVector<uint32_t> uniform_list);

    // Standart SRD collision kernel
    __global__ void srdCollision( DeviceVector<Particle> particles, DeviceVolumeContainer<MPCCell> mpc_cells,
                                Xoshiro128Plus* generator,
                                mpcd::Vector grid_shift, mpcd::Vector volume_size, mpcd::IntVector periodicity,
                                mpcd::Float delta_t, mpcd::Float drag, mpcd::Float thermal_velocity,
                                uint32_t n_density, DeviceVector<uint32_t> uniform_counter,
                                DeviceVector<uint32_t> uniform_list, uint32_t const shared_bytes);

    namespace sampling
    {
        // Reduce particles to cells calculating density and velocity
        __global__ void addParticles(DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<Particle> particles );
        __global__ void averageCells(DeviceVolumeContainer<MPCCell> mpc_cells);

        // Averaging functionality:
        __global__ void snapshot(DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<FluidState> cell_states);
        __global__ void accumulate(DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<FluidState> cell_states, DeviceVector<FluidState> kahan_c);
        __global__ void finish(size_t n_samples, DeviceVolumeContainer<MPCCell> mpc_cells, DeviceVector<FluidState> cell_states);
    }
} // namespace mpcd::cuda
