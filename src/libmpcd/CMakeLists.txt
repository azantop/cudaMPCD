add_subdirectory(cpu_backend)
if(CUDAToolkit_FOUND)
    add_subdirectory(cuda_backend)
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
