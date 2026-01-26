#pragma once
#include <memory>
#include <string>

#include <mpcd/common/simulation_parameters.hpp>

#include "backend/backend.hpp"
#include "cuda_backend/cuda_backend.hpp"

namespace mpcd {

    std::unique_ptr<Backend> create_backend(SimulationParameters const& params, std::string const& backend_type) {
        if (backend_type == "cuda") {
            return std::make_unique<cuda::CudaBackend>(params);
        } else {
            throw std::runtime_error("Unsupported backend type");
        }
    }
} // namespace mpcd
