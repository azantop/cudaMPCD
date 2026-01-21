cmake_minimum_required (VERSION 3.8)
cmake_policy(SET CMP0074 NEW)

project(cudaMPCD LANGUAGES CXX CUDA)

find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(include PUBLIC ${HDF5_INCLUDE_DIRS})

if(DEFINED ENV{CUDA})
    include_directories(PUBLIC $ENV{CUDA}/include)
endif()

# Find pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # If pybind11 is not found, try to fetch it
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Common sources for both executable and Python module
set(COMMON_SOURCES
    src/h5cpp.cpp
    src/cuda_allocator.cpp
    src/simulation_context.cu
    src/gpu_functions.cu
    src/extended_collision.cu
    src/gpu_constants.cu
)

# Original executable
add_executable(main
    src/main.cpp
    ${COMMON_SOURCES}
)

set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 61 72)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET main PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_features(main PUBLIC cxx_std_17)
target_link_libraries(main PUBLIC ${HDF5_LIBRARIES})

# Python module
pybind11_add_module(cudampcd_py
    src/python_bindings.cpp
    ${COMMON_SOURCES}
)

set_property(TARGET cudampcd_py PROPERTY CUDA_ARCHITECTURES 61 72)
set_target_properties(cudampcd_py PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cudampcd_py PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_features(cudampcd_py PUBLIC cxx_std_17)
target_link_libraries(cudampcd_py PUBLIC ${HDF5_LIBRARIES})

# Compiler-specific options
target_compile_definitions(cudampcd_py PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
