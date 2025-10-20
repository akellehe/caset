cmake_minimum_required(VERSION 3.22)
project(caset LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the modern FindPython + pybind11 config
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(CASET_SOURCES
        caset/src/bindings.cpp
        caset/src/Vertex.cpp
        caset/src/Edge.cpp
        )

pybind11_add_module(caset MODULE ${CASET_SOURCES})

target_include_directories(caset PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/caset/src)

# This is all you need:
target_link_libraries(caset PRIVATE pybind11::module)
# DO NOT link Python::Python here.
# If you insist on a Python target, use Python::Module (provided by Development.Module), e.g.:
# target_link_libraries(caset PRIVATE Python::Module)

if (WIN32)
    set_target_properties(caset PROPERTIES SUFFIX ".pyd")
endif()

if (MSVC)
    target_compile_options(caset PRIVATE /EHsc /O2)
else()
    target_compile_options(caset PRIVATE -O3 -fvisibility=hidden)
endif()

