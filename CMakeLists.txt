cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)

project(async_vector_bindings LANGUAGES CXX CUDA)

# Find required packages
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(include PUBLIC ${HDF5_INCLUDE_DIRS})
# Find pybind11
find_package(pybind11 QUIET)

# Set CUDA environment if defined
#if(DEFINED ENV{CUDA})
    include_directories(PUBLIC $ENV{CUDA}/include)
#endif()

# Compile CUDA files separately
add_library(simulator SHARED
    src/mpcd/api/simulator_handle.cu
    src/mpcd/common/simulation_parameters.cu
    src/mpcd/cuda_backend/cuda_backend.cu
    src/mpcd/cuda_backend/gpu_arrays.cu
    src/mpcd/cuda_backend/simulation_kernels.cu
    src/mpcd/cuda_backend/extended_collision.cu
    src/mpcd/cuda_backend/cuda_allocator.cu
    src/mpcd/adapters/hdf5/h5cpp.cpp
)
target_include_directories(simulator
    PRIVATE ${CMAKE_SOURCE_DIR}/src/mpcd
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(simulator
    PRIVATE HDF5::HDF5
)

add_library(cuda_objs SHARED
    src/mpcd/cuda_backend/gpu_arrays.cu
)
target_include_directories(cuda_objs
    PRIVATE ${CMAKE_SOURCE_DIR}/src/mpcd
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

add_executable(mpcd_main
    src/mpcd/main.cu
)
target_include_directories(mpcd_main
    PRIVATE ${CMAKE_SOURCE_DIR}/src/mpcd
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(mpcd_main
    PRIVATE simulator
)

pybind11_add_module(simulator_bindings
    bindings/bindings.cpp
    $<TARGET_OBJECTS:simulator>
)

# Set CUDA properties
set_property(TARGET simulator_bindings PROPERTY CUDA_ARCHITECTURES 61 72)
set_target_properties(simulator_bindings PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET simulator_bindings PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Set C++ standard
target_compile_features(simulator_bindings PUBLIC cxx_std_17)

# Link libraries
target_link_libraries(simulator_bindings PUBLIC ${HDF5_LIBRARIES})

# Compiler-specific options
target_compile_definitions(simulator_bindings PRIVATE VERSION_INFO="1.0.0")

# Set output directory
set_target_properties(simulator_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
install(TARGETS simulator_bindings LIBRARY DESTINATION python/mpcd)
