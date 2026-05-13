# Compile as shared library
add_library(cpu_backend STATIC
    cpu_backend.cpp
    simulation_functions.cpp
)

set_target_properties(cpu_backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cpu_backend PRIVATE
  $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CXX>:-O0 -g>>
)

target_include_directories(cpu_backend
    PRIVATE ${CMAKE_SOURCE_DIR}/src/libmpcd
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)
