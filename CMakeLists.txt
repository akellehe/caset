cmake_minimum_required(VERSION 3.22)
project(caset LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python & pybind11 provided by scikit-build-core env
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)  # from pip pybind11

# Sources
set(CASET_SOURCES
        caset/src/bindings.cpp
        caset/src/Vertex.cpp
        caset/src/Edge.cpp
        )

# Build an extension module named "caset" (importable as `import caset`)
pybind11_add_module(caset MODULE ${CASET_SOURCES})

target_include_directories(caset PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/caset/src)
target_link_libraries(caset PRIVATE pybind11::module Python::Python)

# On Windows, ensure correct suffix.
if (WIN32)
    set_target_properties(caset PROPERTIES SUFFIX ".pyd")
endif()

# Optional: fastmath/visibility
if (MSVC)
    target_compile_options(caset PRIVATE /EHsc /O2)
else()
    target_compile_options(caset PRIVATE -O3 -fvisibility=hidden)
endif()
