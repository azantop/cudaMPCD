#pragma once

#include "common/particle.hpp"
#include "common/vector_3d.hpp"
#include "common/mpc_cell.hpp"
#include "common/random.hpp"

#include "collision_strategy.hpp"
#include "gpu_arrays.hpp"
#include "device_volume_container.hpp"

namespace mpcd::cuda {

    struct ExtendedMPCStrategy : public CollisionStrategy {
        using CollisionStrategy::CollisionStrategy;
        void collideParticles() override;
    };

    struct TrivialExtendedMPCStrategy : public CollisionStrategy {
         // Extra per-cell storage needed for SRD collision:
        DeviceVector<mpcd::InertiaTensor<mpcd::Float>> inertia_tensors;
        DeviceVector<mpcd::Vector>                     angular_momentum;
        DeviceVector<mpcd::Vector>                     rotated_angular_momentum;
        DeviceVector<mpcd::Vector>                     rotation_axes;

        explicit TrivialExtendedMPCStrategy(BackendContext& ctx);
        void sortParticles()    override {} // no sorting applied
        void collideParticles() override {}
    };

} // namespace mpcd::cuda
