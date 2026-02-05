cmake_minimum_required(VERSION 3.22)
cmake_policy(SET CMP0074 NEW)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

include(CTest)
find_package(CUDAToolkit)
find_package(HDF5 COMPONENTS CXX)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA=1)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found - enabling CUDA backend")
else()
    message(STATUS "CUDA not found - disabling CUDA backend")
endif()

if(HDF5_FOUND)
    add_compile_definitions(USE_HDF5=1)
    include_directories(include PUBLIC ${HDF5_INCLUDE_DIRS})
    message(STATUS "HDF5 found - enabling HDF5 support")
else()
    message(STATUS "HDF5 not found - disabling HDF5 support")
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_subdirectory(src)
