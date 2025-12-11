cmake_minimum_required(VERSION 3.18)
project(caset LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- <Find Dependencies> --- #
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(TORCH_CXX11_ABI "1" CACHE STRING "_GLIBCXX_USE_CXX11_ABI value for building against torch (default 1)")
execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
        "import torch, os; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (NOT TORCH_CMAKE_PREFIX_PATH)
    message(FATAL_ERROR "Could not get torch.utils.cmake_prefix_path from ${Python_EXECUTABLE}")
endif ()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
message(STATUS "Torch CMake prefix: ${TORCH_CMAKE_PREFIX_PATH}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries:    ${TORCH_LIBRARIES}")
message(STATUS "_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
# --- </Find Dependencies> --- #

# ---- <Define Sources> ----
file(GLOB_RECURSE CASET_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
)
set(SOURCES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DSOURCES_ROOT="${SOURCES_ROOT}")
# ---- </Define Sources> ----

# --- <Define the caset Target> --- #

# --- <Build the Python module> --- #
pybind11_add_module(caset MODULE ${CASET_SOURCES})
target_compile_definitions(caset PRIVATE SOURCES_ROOT="${SOURCES_ROOT}")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(caset PRIVATE -O0 -g -ftime-report)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(caset PRIVATE -O0 -g -ftime-trace)
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(caset PRIVATE -O3)
endif ()

target_include_directories(caset
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
)

# Match PyTorch’s C++11 ABI (default 1; override if needed)
target_compile_definitions(caset PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
# RPATH so libc10/libtorch are found at runtime
set_target_properties(caset PROPERTIES
        BUILD_RPATH "${TORCH_LIB_DIR}"
        INSTALL_RPATH "${TORCH_LIB_DIR};$ORIGIN;$ORIGIN/.."
        BUILD_WITH_INSTALL_RPATH OFF
)
target_link_directories(caset PRIVATE ${TORCH_LIB_DIR}) # site-packages ONLY!
target_link_libraries(caset PRIVATE ${TORCH_LIBRARIES})

# Extra belt-and-suspenders rpath
if (UNIX AND NOT APPLE)
    target_link_options(caset PRIVATE "-Wl,-rpath,${TORCH_LIB_DIR}")
endif ()

set_target_properties(caset PROPERTIES
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF
)
# ---- </Build the Python module> ----

# --- <Install the Python Module> --- #
install(TARGETS caset
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
        ARCHIVE DESTINATION .
        BUNDLE DESTINATION .
)
# --- </Install the Python Module> --- #

# --- <Build the C++ Library> --- #
file(GLOB_RECURSE
        LIBCASET_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
)

message(STATUS "libcaset sources: ${LIBCASET_SOURCES}")

add_library(
        libcaset
        STATIC
        ${LIBCASET_SOURCES}
)

target_compile_definitions(libcaset PRIVATE SOURCES_ROOT="${SOURCES_ROOT}")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(libcaset PRIVATE -O0 -g -ftime-report)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(libcaset PRIVATE -O0 -g -ftime-trace)
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(libcaset PRIVATE -O3)
endif ()


target_include_directories(libcaset
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
)

# Match PyTorch’s C++11 ABI (default 1; override if needed)
target_compile_definitions(libcaset PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
# RPATH so libc10/libtorch are found at runtime
set_target_properties(libcaset PROPERTIES
        BUILD_RPATH "${TORCH_LIB_DIR}"
        INSTALL_RPATH "${TORCH_LIB_DIR};$ORIGIN;$ORIGIN/.."
        BUILD_WITH_INSTALL_RPATH OFF
)
target_link_directories(libcaset PRIVATE ${TORCH_LIB_DIR}) # site-packages ONLY!
target_link_libraries(libcaset PRIVATE ${TORCH_LIBRARIES})

# Extra belt-and-suspenders rpath
if (UNIX AND NOT APPLE)
    target_link_options(libcaset PRIVATE "-Wl,-rpath,${TORCH_LIB_DIR}")
endif ()

set_target_properties(libcaset PROPERTIES
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF
)

# --- </Build the C++ Library> --- #


# --- <Enable Gtest> --- #
enable_testing()
include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Prevent GoogleTest from forcing / changing weird flags on Apple
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_test(NAME unit_tests COMMAND unit_tests)
add_executable(unit_tests
        tests/test_simplex.cpp
)

target_link_libraries(unit_tests PRIVATE gtest gtest_main libcaset)
# --- </Enable Gtest> --- #

# --- <Generate Python stubs for caset (.pyi)> ---
set(STUB_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/stubs")

add_custom_command(
        TARGET caset POST_BUILD
        BYPRODUCTS "${STUB_OUT_DIR}/caset.pyi"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${STUB_OUT_DIR}"
        #Make the just-built module importable:
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=$<TARGET_FILE_DIR:caset>
        ${Python_EXECUTABLE} -m pybind11_stubgen caset -o "${STUB_OUT_DIR}"
        COMMENT "Generating pybind11 stubs -> ${STUB_OUT_DIR}/caset.pyi"
)

install(FILES "${STUB_OUT_DIR}/caset.pyi" DESTINATION .)  # Install the stub next to the extension in the package

# --- </Generate Python stubs for caset (.pyi)> ---
