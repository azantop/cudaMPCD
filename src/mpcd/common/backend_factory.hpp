#pragma once
#include <memory>
#include <string>
#include <iostream>

#include <mpcd/api/simulation_parameters.hpp>

#include "backend/backend.hpp"
#include "cuda_backend/cuda_backend.hpp"
#include "cpu_backend/cpu_backend.hpp"

namespace mpcd {

    std::unique_ptr<Backend> create_backend(SimulationParameters const& params, std::string const& backend_type) {
        if (backend_type == "cuda") {
            std::cout << "Creating CUDA backend" << std::endl;
            return std::make_unique<cuda::CudaBackend>(params);
        } else if (backend_type == "cpu") {
            std::cout << "Creating CPU backend (Preliminary Dummy Backend)" << std::endl;
            return std::make_unique<cpu::CPUBackend>(params);
        } else {
            throw std::runtime_error("Unsupported backend type");
        }
    }
} // namespace mpcd
