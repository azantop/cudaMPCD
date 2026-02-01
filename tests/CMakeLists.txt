# GoogleTest for C++/CUDA tests
find_package(GTest REQUIRED)
include(GoogleTest)

# Helper function for CPP tests
function(add_cpp_test test_name)
    add_executable(${test_name} ${ARGN})
    target_include_directories(${test_name}
        PRIVATE ${CMAKE_SOURCE_DIR}/src/mpcd
        PUBLIC ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(${test_name}
        PRIVATE
        simulator_backend
        GTest::gtest_main
    )
    target_compile_features(${test_name} PRIVATE cxx_std_17)
    gtest_discover_tests(${test_name})
endfunction()

# Helper function for CUDA tests
function(add_cuda_test test_name)
    add_executable(${test_name} ${ARGN})
    target_include_directories(${test_name}
        PRIVATE ${CMAKE_SOURCE_DIR}/src/mpcd
        PUBLIC ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(${test_name}
        PRIVATE
        simulator_backend
        GTest::gtest_main
    )
    target_compile_features(${test_name} PRIVATE cxx_std_17)
    set_property(TARGET ${test_name} PROPERTY CUDA_ARCHITECTURES 70 75 80)
    gtest_discover_tests(${test_name})
endfunction()

# Unit tests for vector operations
add_cpp_test(test_vector_3d
    unit/common/test_vector_3d.cpp
)

# Unit tests for GPU arrays
add_cuda_test(test_gpu_arrays
    unit/cuda/test_gpu_arrays.cu
)

# Unit tests for GPU simulation kernels
add_cuda_test(test_simulation_kernels
    unit/cuda/test_simulation_kernels.cu
)
