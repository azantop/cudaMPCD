#pragma once

#include "common/particle.hpp"
#include "common/vector_3d.hpp"
#include "common/mpc_cell.hpp"
#include "common/random.hpp"
#include "collision_strategy.hpp"
#include "gpu_arrays.hpp"
#include "device_volume_container.hpp"

namespace mpcd::cuda {

    /**
     * @brief Baseline SRD collision strategy using plain scatter/reduce kernels.
     *
     * Memory strategy:
     *   Particles are accessed in their current unsorted order. Each particle
     *   atomically contributes to its cell's mean velocity (addReduceOnly).
     *   Random rotation axes are stored per-cell in a separate DeviceVector
     *   allocated at construction — this is the only extra device allocation
     *   beyond BackendContext.
     *
     * Kernel sequence (4 launches, no fusion):
     *   1. scatter          — particles → cells, atomic addReduceOnly
     *   2. averageCells     — cells → mean_velocity via averageReduceOnly
     *   3. generateAxes     — cells → rotation_axes, one RNG draw per cell
     *   4. applyCollision   — particles → rotated velocities + grid shift unapply
     *
     * Kernel fusion: none. Each pass is a separate global memory round-trip.
     * This function is intended for performance comparison.
     */
    struct TrivialSRDStrategy : public CollisionStrategy {
        DeviceVector<mpcd::Vector> rotation_axes; // per-cell random rotation axis

        explicit TrivialSRDStrategy(BackendContext& ctx);
        virtual void sortParticles()    override {} // no sorting applied
        void collideParticles() override;
    };

    /**
     * @brief TrivialSRDStrategy with a particle sort prepended to the collision step.
     *
     * Memory strategy:
     *   Before collision, particles are regularly sorted by cell index
     *   using the GPU counting sort from CollisionStrategy::sortParticles().
     *   This improves coalesced memory reads and writes by aliging particle memomry order
     *   and cell memory order.
     *
     * Extra calls vs TrivialSRDStrategy:
     *   sortParticles() explicitly calls CollisionStrategy::sortParticles(), bypassing
     *   TrivialSRDStrategy's no-op override. This adds 2–3 kernel launches
     *   (binParticles, computePrefixSum, moveParticles or sortInPlace).
     *
     * Kernel fusion: none beyond what TrivialSRDStrategy provides.
     * This function is intended for performance comparison.
     */
    struct SortingSRDStrategy : public TrivialSRDStrategy {
        using TrivialSRDStrategy::TrivialSRDStrategy;
        void sortParticles() override; // re-enables counting sort, bypassing TrivialSRDStrategy's no-op
        // collideParticles() inherited from TrivialSRDStrategy
    };

    /**
     * @brief Optimized SRD strategy using warp-level shared memory and kernel fusion.
     *
     * Memory strategy:
     *   Particles are loaded into per-warp shared memory in cell-contiguous
     *   batches using a uniform_list lookup table populated during the preceding
     *   translation step. Each warp processes as many complete SRD cells as fit
     *   into the shared memory budget (cuda_config.shared_bytes), eliminating
     *   repeated global memory reads for particles within the same cell.
     *   Ghost particles for wall boundary cells are generated on-the-fly in
     *   shared memory — no extra device allocation.
     *
     * Kernel fusion:
     *   The scatter (mean velocity accumulation), rotation axis draw, Rodrigues
     *   rotation, and position unshift are all fused into a single kernel launch
     *   (srdCollision). No separate averageCells or generateAxes launch.
     *   Warp shuffles (__shfl_sync, __ballot_sync) replace atomic operations
     *   for intra-cell reductions, avoiding serialisation on global atomics.
     *
     *
     * Suitable for production runs on Volta/Turing/Ampere hardware where
     * shared memory per SM is sufficient to hold several cells' worth of
     * particles per warp.
     */
    struct SRDStrategy : public CollisionStrategy {
        using CollisionStrategy::CollisionStrategy;
        void collideParticles() override;
    };

} // namespace mpcd::cuda
