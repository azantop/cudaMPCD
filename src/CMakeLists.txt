# Backend and standalone executable:
add_subdirectory(libmpcd)

# Python bindings:
add_subdirectory(bindings)

# Plain executable target
add_executable(mpcd_main
    main.cpp
    libmpcd/api/simulation_parameters.cpp
)
target_include_directories(mpcd_main
    PRIVATE ${CMAKE_SOURCE_DIR}/src/libmpcd
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(mpcd_main
    PRIVATE cuda_backend
)
