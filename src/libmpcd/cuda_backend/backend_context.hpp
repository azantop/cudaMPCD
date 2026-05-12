#pragma once

#include <mpcd/api/simulation_parameters.hpp>

#include "common/vector_3d.hpp"
#include "common/particle.hpp"
#include "common/mpc_cell.hpp"
#include "common/random.hpp"
#include "gpu_arrays.hpp"
#include "device_volume_container.hpp"

namespace mpcd::cuda {

struct BackendContext {
    SimulationParameters             parameters;
    UnifiedVector<Particle>          particles;        // SRD fluid particles
    DeviceVector<Particle>           particles_sorted; // use for gpu sorting later

    mpcd::Vector                     grid_shift;       // SRD grid shift

    DeviceVolumeContainer<MPCCell>   mpc_cells;        // SRD cell storage
    UnifiedVector<FluidState>        cell_states;      // for averaging over the fluid state
    UnifiedVector<FluidState>        kahan_c;          // cache for kahan summation

    // The indices for fluid particles are stored in a lookup table for the collision step.
    // This optimizes the data throughput, because particles can be stored in shared memory
    // and only need to be loaded once:
    DeviceVector<uint32_t>           uniform_list;     // the index lookup
    DeviceVector<uint32_t>           uniform_counter;  // next free table entry, used with atomicAdd

    DeviceVector<uint32_t>           cell_prefix;      // exclusive prefix sum of per-cell counts
    DeviceVector<uint32_t>           sort_tmp;         // scratch for multi-level prefix sum
    bool                             use_tmp_sort_buffer; // true when VRAM allows a scatter-sort buffer

    DeviceVector<Xoshiro128Plus>     generator;        // random number generators for the gpu
    Xorshift1024Star                 random;           // random number generator for the cpu

    // To further optimize memory loading, the particle array is sorted according to the SRD cell-index.
    // This enables array striding, ie. coalesced memory loading:
    size_t                           step_counter;
    size_t                           resort_rate;

    struct {
        size_t block_size;
        size_t block_count;
        size_t multiprocessors;
        size_t shared_bytes;
        size_t sharing_blocks;
        size_t internal_step_counter;
        size_t resort_rate;
    } cuda_config;

    explicit BackendContext(SimulationParameters const&);
    ~BackendContext();
};

} // namespace mpcd::cuda
