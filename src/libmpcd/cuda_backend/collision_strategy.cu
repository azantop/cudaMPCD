#include <algorithm>
#include <memory>

#include "collision_strategy.hpp"
#include "srd_collision_strategy.hpp"
#include "extended_collision_strategy.hpp"
#include "sort_particles.hpp"

namespace mpcd::cuda {

    // CollisionStrategy constructor is inline in the header (ctx(ctx) initializer)

    void CollisionStrategy::sortParticles() {
        ctx.uniform_counter.set(0);
        binParticles(ctx.particles, ctx.uniform_counter, ctx.cuda_config.block_count, ctx.cuda_config.block_size);
        error_check("sort_binning");

        computePrefixSum(ctx.uniform_counter, ctx.cell_prefix, ctx.sort_tmp);
        error_check("sort_prefix_sum");

        if (ctx.use_tmp_sort_buffer) {
            moveParticles(ctx.particles, ctx.particles_sorted, ctx.cell_prefix, ctx.cuda_config.block_count, ctx.cuda_config.block_size);
            error_check("sort_move");
            std::swap(ctx.particles.device_store, ctx.particles_sorted.store);
        } else {
            writeTargetIndices(ctx.particles, ctx.cell_prefix, ctx.cuda_config.block_count, ctx.cuda_config.block_size);
            error_check("sort_write_indices");
            sortInPlace(ctx.particles, ctx.cuda_config.block_count, ctx.cuda_config.block_size);
            error_check("sort_in_place");
        }
    }

    std::unique_ptr<CollisionStrategy> makeCollisionStrategy(BackendContext& ctx) {
        switch (ctx.parameters.algorithm) {
            case MPCDAlgorithm::srd:
                switch (ctx.parameters.collision_kernel) {
                    case CollisionKernel::trivial:   return std::make_unique<TrivialSRDStrategy>(ctx);
                    case CollisionKernel::sorting:   return std::make_unique<SortingSRDStrategy>(ctx);
                    case CollisionKernel::optimized: return std::make_unique<SRDStrategy>(ctx);
                    default:                         return std::make_unique<SRDStrategy>(ctx);
                }
            case MPCDAlgorithm::extended:
                return std::make_unique<ExtendedMPCStrategy>(ctx);
            default:
                return std::make_unique<SRDStrategy>(ctx);
        }
    }

} // namespace mpcd::cuda
