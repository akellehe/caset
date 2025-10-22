cmake_minimum_required(VERSION 3.22)
project(caset LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "Python executable:        ${Python_EXECUTABLE}")
message(STATUS "Python version:           ${Python_VERSION}")
message(STATUS "Python include dirs:      ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries:         ${Python_LIBRARIES}")  # might be empty for Module-only
message(STATUS "Python site packages:     ${Python_SITEARCH}")
message(STATUS "CMAKE_BINARY_DIR:         ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

find_package(pybind11 CONFIG REQUIRED)

set(Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/libtorch/share/cmake/Torch)
list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}")

find_package(Torch REQUIRED)

#find_package(CUDAToolkit REQUIRED)


set(CASET_SOURCES
        src/bindings.cpp
        src/Vertex.cpp
        src/Edge.cpp
        src/Simplex.cpp
        )

pybind11_add_module(caset MODULE ${CASET_SOURCES})
target_include_directories(caset PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include ${Python_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(caset PRIVATE pybind11::module ${TORCH_LIBRARIES})

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
