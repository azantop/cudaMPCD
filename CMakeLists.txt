cmake_minimum_required(VERSION 3.22)
cmake_policy(SET CMP0074 NEW)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX CUDA
)

# Enable testing
include(CTest)
enable_testing()

# Add tests subdirectory
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Find required packages
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(include PUBLIC ${HDF5_INCLUDE_DIRS})

# Find pybind11
find_package(pybind11 REQUIRED)

# Set CUDA environment if defined
#if(DEFINED ENV{CUDA})
    include_directories(PUBLIC $ENV{CUDA}/include)
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
#endif()

# Compile CUDA files separately
add_library(simulator_backend SHARED
    src/mpcd/api/simulator_handle.cu
    src/mpcd/api/simulation_parameters.cu
    src/mpcd/cuda_backend/cuda_backend.cu
    src/mpcd/cuda_backend/gpu_arrays.cu
    src/mpcd/cuda_backend/simulation_kernels.cu
    src/mpcd/cuda_backend/extended_collision.cu
    src/mpcd/cuda_backend/cuda_allocator.cu
    src/mpcd/cpu_backend/cpu_backend.cpp
    src/mpcd/cpu_backend/simulation_functions.cpp
    src/mpcd/adapters/hdf5/h5cpp.cpp
)
target_include_directories(simulator_backend
    PRIVATE ${CMAKE_SOURCE_DIR}/src/mpcd
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(simulator_backend
    PRIVATE HDF5::HDF5
)

# Plain executable target
add_executable(mpcd_main
    src/mpcd/main.cu
)
target_include_directories(mpcd_main
    PRIVATE ${CMAKE_SOURCE_DIR}/src/mpcd
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(mpcd_main
    PRIVATE simulator_backend
)

# Python module target
pybind11_add_module(_pympcd
    bindings/bindings.cpp
    $<TARGET_OBJECTS:simulator_backend>
)

# Set CUDA properties
set_target_properties(_pympcd PROPERTIES CUDA_ARCHITECTURES "61;72")
set_target_properties(_pympcd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(_pympcd PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Set C++ standard
target_compile_features(_pympcd PUBLIC cxx_std_17)

# Link libraries
target_link_libraries(_pympcd PUBLIC ${HDF5_LIBRARIES})

# Compiler-specific options
target_compile_definitions(_pympcd PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Set output directory
set_target_properties(_pympcd PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# This is passing in the version as a define just as an example

# The install directory is the output (wheel) directory
install(TARGETS simulator_backend DESTINATION pympcd)
install(TARGETS _pympcd DESTINATION pympcd)

# Install Python files
install(DIRECTORY src/mpcd/
    DESTINATION pympcd
    FILES_MATCHING PATTERN "*.py"
)
