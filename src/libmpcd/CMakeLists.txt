find_package(HDF5 COMPONENTS CXX)
if(HDF5_FOUND)
    add_compile_definitions(USE_HDF5=1)
    include_directories(include PUBLIC ${HDF5_INCLUDE_DIRS})
    message(STATUS "HDF5 found - enabling HDF5 support")
else()
    message(STATUS "HDF5 not found - disabling HDF5 support")
endif()

add_subdirectory(cpu_backend)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA=1)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found - enabling CUDA backend")
    add_subdirectory(cuda_backend)
else()
    message(STATUS "CUDA not found - disabling CUDA backend")
endif()

add_library(mpcd_backend SHARED
    api/simulator_handle.cpp
)
target_include_directories(mpcd_backend
    PRIVATE ${CMAKE_SOURCE_DIR}/src/libmpcd
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(mpcd_backend PUBLIC cpu_backend)
if(CUDAToolkit_FOUND)
    target_link_libraries(mpcd_backend PUBLIC cuda_backend)
endif()

set_target_properties(mpcd_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON
)
