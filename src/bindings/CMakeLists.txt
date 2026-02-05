find_package(pybind11)

if(pybind11_FOUND)
    message(STATUS "pybind11 ${pybind11_VERSION} found - enabling python bindings")

    pybind11_add_module(_pympcd
        bindings.cpp
    )
    target_link_libraries(_pympcd PRIVATE
        mpcd_backend
    )

    target_compile_features(_pympcd PUBLIC cxx_std_17)
    target_compile_definitions(_pympcd PRIVATE VERSION_INFO=${PROJECT_VERSION})

    target_include_directories(_pympcd
        PUBLIC ${CMAKE_SOURCE_DIR}/include
    )

    set_target_properties(_pympcd PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH ON
    )

    # The install directory is the output (wheel) directory
    if(CUDAToolkit_FOUND)
        install(TARGETS
            _pympcd mpcd_backend cuda_backend
            DESTINATION pympcd
        )
    else()
        install(TARGETS
            _pympcd mpcd_backend
            DESTINATION pympcd
        )
    endif()
    # Install Python files
    install(DIRECTORY src/pympcd/
        DESTINATION pympcd
        FILES_MATCHING PATTERN "*.py"
    )
endif()
