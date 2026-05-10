#include <cstdint>
#include <vector>
#include "sort_particles.hpp"
#include "gpu_utilities.hpp"

namespace mpcd::cuda {

    namespace {

        __global__ void kernelBinParticles(Particle* particles, uint32_t* uniform_counter, size_t n_particles) {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;
            for (; idx < n_particles; idx += stride)
                particles[idx].cidx = (int16_t)atomicAdd(
                    uniform_counter + gpu_utilities::texture_load(&particles[idx].cell_idx), 1u);
        }

        // Returns the inclusive prefix sum for this thread's lane within its warp.
        __device__ __forceinline__ uint32_t warpInclusiveSum(uint32_t value) {
            for (int offset = 1; offset < 32; offset <<= 1) {
                uint32_t other = __shfl_up_sync(0xffffffffu, value, offset);
                if ((threadIdx.x & 31) >= offset)
                    value += other;
            }
            return value;
        }

        // Block-wise exclusive scan. Writes one block total to block_sums[blockIdx.x].
        // Launch with shared mem = blockDim.x * sizeof(uint32_t).
        __global__ void kernelExclusiveScan(uint32_t* __restrict__ in, uint32_t* __restrict__ block_sums, uint32_t N) {
            extern __shared__ uint32_t shared_mem[];
            uint32_t idx     = threadIdx.x + blockIdx.x * blockDim.x;
            int      lane_id = threadIdx.x & 31;
            int      warp_id = threadIdx.x >> 5;
            int      n_warps = blockDim.x >> 5;

            uint32_t value  = idx < N ? in[idx] : 0u;
            uint32_t prefix = warpInclusiveSum(value);

            if (lane_id == 31)
                shared_mem[warp_id] = prefix;
            __syncthreads();

            if (warp_id == 0) {
                uint32_t warp_value  = lane_id < n_warps ? shared_mem[lane_id] : 0u;
                uint32_t warp_prefix = warpInclusiveSum(warp_value);

                if (lane_id < n_warps)
                    shared_mem[lane_id] = warp_prefix - warp_value;  // exclusive warp offset

                if (lane_id == n_warps - 1)
                    block_sums[blockIdx.x] = warp_prefix;  // block total (inclusive at last warp)
            }
            __syncthreads();

            if (idx < N)
                in[idx] = (prefix - value) + shared_mem[warp_id];
        }

        __global__ void kernelAddBlockSums(uint32_t* __restrict__ in, uint32_t* __restrict__ block_prefix, uint32_t N) {
            uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < N)
                in[idx] += block_prefix[blockIdx.x];
        }

        // Multi-level exclusive scan. d_tmp must be large enough for all intermediate block sums
        // (ceil(N/256) + ceil(N/256^2) + ... entries; allocating N is always sufficient).
        void runExclusiveScan(uint32_t* d_input, uint32_t* d_result, uint32_t* d_tmp, uint32_t N) {
            cudaMemcpy(d_result, d_input, N * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

            constexpr int block_size = 256;
            std::vector<std::pair<uint32_t*, uint32_t>> levels;
            levels.push_back({d_result, N});

            int blocks;
            do {
                blocks = ((int)levels.back().second + block_size - 1) / block_size;
                kernelExclusiveScan<<<blocks, block_size, block_size * sizeof(uint32_t)>>>(
                    levels.back().first, d_tmp, levels.back().second);
                levels.push_back({d_tmp, (uint32_t)blocks});
                d_tmp += blocks;
            } while (blocks != 1);

            // levels.back() holds the inclusive block total of the terminal scan (grand sum).
            // It must not be propagated — skip the last entry.
            for (int i = (int)levels.size() - 3; i >= 0; --i)
                kernelAddBlockSums<<<levels[i + 1].second, block_size>>>(
                    levels[i].first, levels[i + 1].first, levels[i].second);
        }

        __global__ void kernelMoveParticles(Particle* particles, Particle* particles_sorted,
                                            uint32_t* cell_prefix, size_t n_particles) {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;
            for (; idx < n_particles; idx += stride) {
                auto p = particles[idx];
                particles_sorted[cell_prefix[p.cell_idx] + (uint32_t)p.cidx] = p;
            }
        }

        __global__ void kernelWriteTargetIndices(Particle* particles, uint32_t* cell_prefix, size_t n_particles) {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;
            for (; idx < n_particles; idx += stride)
                particles[idx].cell_idx = cell_prefix[particles[idx].cell_idx] + (uint32_t)particles[idx].cidx;
        }

        __global__ void kernelSortInPlace(Particle* particles, size_t n_particles) {
            size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;
            for (; idx < n_particles; idx += stride) {
                auto running_idx = idx;
                auto particle    = particles[running_idx];
                auto sorting_idx = (size_t)particle.cell_idx;

                while (sorting_idx != running_idx) {
                    auto next_idx          = (size_t)atomicExch(&particles[sorting_idx].cell_idx, (uint32_t)sorting_idx);
                    auto buffer            = particles[sorting_idx];
                    particles[sorting_idx] = particle;
                    particle               = buffer;
                    particle.cell_idx      = (uint32_t)next_idx;
                    running_idx            = sorting_idx;
                    sorting_idx            = next_idx;
                }
            }
        }

    } // anonymous namespace

    void binParticles(DeviceVector<Particle> particles, DeviceVector<uint32_t> uniform_counter,
                      size_t block_count, size_t block_size) {
        kernelBinParticles<<<block_count, block_size>>>(particles.data(), uniform_counter.data(), particles.size());
    }

    void computePrefixSum(DeviceVector<uint32_t> uniform_counter, DeviceVector<uint32_t> cell_prefix,
                          DeviceVector<uint32_t> sort_tmp) {
        runExclusiveScan(uniform_counter.data(), cell_prefix.data(), sort_tmp.data(),
                         (uint32_t)uniform_counter.size());
    }

    void moveParticles(DeviceVector<Particle> particles, DeviceVector<Particle> particles_sorted,
                       DeviceVector<uint32_t> cell_prefix, size_t block_count, size_t block_size) {
        kernelMoveParticles<<<block_count, block_size>>>(particles.data(), particles_sorted.data(),
                                                         cell_prefix.data(), particles.size());
    }

    void writeTargetIndices(DeviceVector<Particle> particles, DeviceVector<uint32_t> cell_prefix,
                            size_t block_count, size_t block_size) {
        kernelWriteTargetIndices<<<block_count, block_size>>>(particles.data(), cell_prefix.data(), particles.size());
    }

    void sortInPlace(DeviceVector<Particle> particles, size_t block_count, size_t block_size) {
        kernelSortInPlace<<<block_count, block_size>>>(particles.data(), particles.size());
    }

} // namespace mpcd::cuda
