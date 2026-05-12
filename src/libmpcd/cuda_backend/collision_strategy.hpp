#pragma once

#include <memory>
#include "backend_context.hpp"

namespace mpcd::cuda {

struct CollisionStrategy {
    BackendContext& ctx;

    explicit CollisionStrategy(BackendContext& ctx) : ctx(ctx) {}

    virtual void sortParticles();   // default: GPU counting sort
    virtual void collideParticles() = 0;
    virtual ~CollisionStrategy()    = default;
};

std::unique_ptr<CollisionStrategy> makeCollisionStrategy(BackendContext&);

} // namespace mpcd::cuda
