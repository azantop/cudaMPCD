#pragma once

#include "common/particle.hpp"
#include "common/vector_3d.hpp"
#include "common/mpc_cell.hpp"
#include "common/random.hpp"

#include "collision_strategy.hpp"
#include "gpu_arrays.hpp"
#include "device_volume_container.hpp"

namespace mpcd::cuda {

    struct TrivialSRDStrategy : public CollisionStrategy {
        using CollisionStrategy::CollisionStrategy;
        void sortParticles()    override {} // no sorting applied
        void collideParticles() override;
    };

    struct SortingSRDStrategy : public CollisionStrategy {
        using CollisionStrategy::CollisionStrategy;
        void collideParticles() override;
    };

    struct SRDStrategy : public CollisionStrategy {
        using CollisionStrategy::CollisionStrategy;
        void collideParticles() override;
    };

} // namespace mpcd::cuda
