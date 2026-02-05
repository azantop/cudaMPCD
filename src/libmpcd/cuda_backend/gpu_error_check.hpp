#pragma once
#include <cstdio>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

namespace mpcd::cuda {
    inline __host__ __device__ void error_check( cudaError_t error, const char* loc ="unknown location" ){
        if ( error != 0 ) {
            #ifndef __CUDA_ARCH__
                //printf( "error: %s, (%s)", cudaGetErrorString( error ), loc );
                cudaDeviceSynchronize();
                std::string error_message = std::string(cudaGetErrorString(error))
                                                + " at " + std::string(loc);
                throw std::runtime_error(error_message);
            #else
                printf( "error: %s", loc );
            #endif
        }
    }

    inline __host__ __device__ void error_check( const char* loc ="unknown location" ) {
        #ifndef __CUDA_ARCH__
            #ifndef NDEBUG
                cudaDeviceSynchronize();
            #endif

            cudaError_t error = cudaGetLastError();
            if ( error != 0 ) {
                    //printf( "error: %s, (%s)\n", cudaGetErrorString( error ), loc );
                    cudaDeviceSynchronize();
                    std::string error_message = std::string(cudaGetErrorString(error))
                                                    + " at " + std::string(loc);
                    throw std::runtime_error(error_message);
            }
        #endif
    }

    inline bool hasCudaDevice() {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);

        if (err != cudaSuccess) {
            return false;
        }
        return count > 0;
    }
} // namespace mpcd::cuda
