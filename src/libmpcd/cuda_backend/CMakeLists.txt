#if(DEFINED ENV{CUDA})
#endif()
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

add_library(cuda_backend SHARED
    cuda_backend.cu
    cuda_backend_impl.cu
    gpu_arrays.cu
    simulation_kernels.cu
    extended_collision.cu
    cuda_allocator.cu
)

set_target_properties(cuda_backend PROPERTIES
    CUDA_ARCHITECTURES "61;72"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(cuda_backend
    PRIVATE ${CMAKE_SOURCE_DIR}/src/libmpcd
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cuda_backend PRIVATE CUDA::cudart)

if(HDF5_FOUND)
    add_library(hdf5adapter STATIC
        ../adapters/hdf5/h5cpp.cpp
    )
    target_link_libraries(cuda_backend PRIVATE hdf5adapter HDF5::HDF5)
endif()
