cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)

project(async_vector_bindings LANGUAGES CXX CUDA)

# Find required packages
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(include PUBLIC ${HDF5_INCLUDE_DIRS})
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# Find pybind11
find_package(pybind11 QUIET)

# Set CUDA environment if defined
#if(DEFINED ENV{CUDA})
    include_directories(PUBLIC $ENV{CUDA}/include)
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
    src/mpcd/adapters/hdf5/h5cpp.cpp
)
target_include_directories(simulator_backend
    PRIVATE ${CMAKE_SOURCE_DIR}/src/mpcd
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(simulator_backend
    PRIVATE HDF5::HDF5
)
install(TARGETS simulator_backend
    LIBRARY DESTINATION "${PY_SITE_PACKAGES}/pympcd"
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
target_compile_definitions(_pympcd PRIVATE VERSION_INFO="1.0.0")

# Set output directory
set_target_properties(_pympcd PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)

# Compute site-packages path for the current Python
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PY_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python site-packages: ${PY_SITE_PACKAGES}")

# Install target
install(TARGETS _pympcd
    LIBRARY DESTINATION "${PY_SITE_PACKAGES}/pympcd"
)
# Install Python files
install(DIRECTORY src/mpcd/
    DESTINATION "${PY_SITE_PACKAGES}/pympcd"
    FILES_MATCHING PATTERN "*.py"
)
