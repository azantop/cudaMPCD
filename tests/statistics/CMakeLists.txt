find_package(pybind11 REQUIRED)

pybind11_add_module(_test_statistics bindings.cpp)

target_link_libraries(_test_statistics PRIVATE mpcd_backend)

# Include paths to main source
target_include_directories(_test_statistics
    PRIVATE ${CMAKE_SOURCE_DIR}/src/libmpcd
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

# Install in tests directory
set_target_properties(_test_statistics PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib_tests
)

find_program(PYTHON_EXECUTABLE python3)
if(PYTHON_EXECUTABLE)
    add_test(NAME TestStatistics
                COMMAND ${PYTHON_EXECUTABLE} test_statistics.py
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    set_tests_properties(TestStatistics PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib_tests"
    )
endif()
