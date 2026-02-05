#pragma once
#include <memory>
#include <string>
#include <iostream>

#include <mpcd/api/simulation_parameters.hpp>

#include "backend.hpp"
#include "cpu_backend/cpu_backend.hpp"
#ifdef USE_CUDA
    #include "cuda_backend/cuda_backend.hpp"
#endif

namespace mpcd {
    std::unique_ptr<Backend> create_backend(SimulationParameters const& params, std::string const& backend_type) {
        if (backend_type == "cpu") {
            std::cout << "Creating CPU backend (WARNING: CPU backend is not fully supported)" << std::endl;
            return std::make_unique<cpu::CPUBackend>(params);
        }
        #ifdef USE_CUDA
        else if (backend_type == "cuda") {
            std::cout << "Creating CUDA backend" << std::endl;
            return std::make_unique<cuda::CudaBackend>(params);
        }
        #endif
        else {
            throw std::runtime_error("Unsupported backend type");
        }
    }
} // namespace mpcd
