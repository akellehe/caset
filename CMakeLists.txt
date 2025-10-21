cmake_minimum_required(VERSION 3.22)
project(caset LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use the modern FindPython + pybind11 config
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "Python executable:    ${Python_EXECUTABLE}")
message(STATUS "Python version:       ${Python_VERSION}")
message(STATUS "Python include dirs:  ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries:     ${Python_LIBRARIES}")  # might be empty for Module-only
message(STATUS "CMAKE_BINARY_DIR:     ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

find_package(pybind11 CONFIG REQUIRED)

set(CASET_SOURCES
        src/bindings.cpp
        src/Vertex.cpp
        src/Edge.cpp
        )

pybind11_add_module(caset MODULE ${CASET_SOURCES})
target_include_directories(caset PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(caset PRIVATE pybind11::module)

install(TARGETS caset
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
        ARCHIVE DESTINATION .
        BUNDLE DESTINATION .
)

add_custom_command(
        TARGET caset POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Built module: $<TARGET_FILE:caset>"
        VERBATIM)