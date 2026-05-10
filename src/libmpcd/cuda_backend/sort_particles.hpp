#pragma once
#include <cstdint>
#include "common/particle.hpp"
#include "gpu_arrays.hpp"

namespace mpcd::cuda {

    // Count particles per cell into uniform_counter; store per-cell rank in particle.cidx.
    void binParticles(DeviceVector<Particle> particles, DeviceVector<uint32_t> uniform_counter,
                      size_t block_count, size_t block_size);

    // Multi-level exclusive prefix sum: uniform_counter[i] (counts) → cell_prefix[i] (starts).
    // sort_tmp is scratch space; size >= ceil(N/256) + ceil(N/256^2) + … (N elements is always safe).
    void computePrefixSum(DeviceVector<uint32_t> uniform_counter, DeviceVector<uint32_t> cell_prefix,
                          DeviceVector<uint32_t> sort_tmp);

    // Scatter particles to sorted positions in particles_sorted.
    void moveParticles(DeviceVector<Particle> particles, DeviceVector<Particle> particles_sorted,
                       DeviceVector<uint32_t> cell_prefix, size_t block_count, size_t block_size);

    // Rewrite particle.cell_idx to its final sorted position (prerequisite for sortInPlace).
    void writeTargetIndices(DeviceVector<Particle> particles, DeviceVector<uint32_t> cell_prefix,
                            size_t block_count, size_t block_size);

    // Cyclic-permutation in-place sort; requires writeTargetIndices to have been called first.
    void sortInPlace(DeviceVector<Particle> particles, size_t block_count, size_t block_size);

} // namespace mpcd::cuda
