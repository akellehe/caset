cmake_minimum_required(VERSION 3.18)
project(caset LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Python & pybind11 ----
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)


# ---- <Locate Python's torch package WITHOUT importing it> ----
# Option to override ABI if needed
set(TORCH_CXX11_ABI "1" CACHE STRING "_GLIBCXX_USE_CXX11_ABI value for building against torch (default 1)")

execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
        "import torch, os; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_CMAKE_PREFIX_PATH)
    message(FATAL_ERROR "Could not get torch.utils.cmake_prefix_path from ${Python_EXECUTABLE}")
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)

message(STATUS "Torch CMake prefix: ${TORCH_CMAKE_PREFIX_PATH}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries:    ${TORCH_LIBRARIES}")
message(STATUS "_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
# ---- </Locate Python's torch package WITHOUT importing it> ----

# ---- <Define Sources> ----
file(GLOB_RECURSE CASET_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
)
# ---- </Define Sources> ----

# ---- <Build the Python module> ----
pybind11_add_module(caset MODULE ${CASET_SOURCES})

set(SOURCES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(caset PRIVATE SOURCES_ROOT="${SOURCES_ROOT}")
add_definitions(-DSOURCES_ROOT="${SOURCES_ROOT}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(caset PRIVATE CASET_DEBUG=1)
    add_definitions(-DSOURCES_ROOT="${SOURCES_ROOT}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(caset PRIVATE -O3)
endif()

target_include_directories(caset
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
)

# Link ONLY against torch from site-packages
target_link_directories(caset PRIVATE ${TORCH_LIB_DIR})
target_link_libraries(caset PRIVATE ${TORCH_LIBRARIES})

# Match PyTorch’s C++11 ABI (default 1; override if needed)
target_compile_definitions(caset PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})

# RPATH so libc10/libtorch are found at runtime
set_target_properties(caset PROPERTIES
        BUILD_RPATH "${TORCH_LIB_DIR}"
        INSTALL_RPATH "${TORCH_LIB_DIR};$ORIGIN;$ORIGIN/.."
        BUILD_WITH_INSTALL_RPATH OFF
)

# Extra belt-and-suspenders rpath
#if(UNIX AND NOT APPLE)
#    target_link_options(caset PRIVATE "-Wl,-rpath,${TORCH_LIB_DIR}")
#endif()

set_target_properties(caset PROPERTIES
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF
)
# ---- </Build the Python module> ----

# Install next to your package
install(TARGETS caset
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
        ARCHIVE DESTINATION .
        BUNDLE  DESTINATION .
)

# --- Generate Python stubs for caset (.pyi) ---
#set(STUB_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/stubs")

#add_custom_command(
#        TARGET caset POST_BUILD
#        BYPRODUCTS "${STUB_OUT_DIR}/caset.pyi"
#        COMMAND ${CMAKE_COMMAND} -E make_directory "${STUB_OUT_DIR}"
#         Make the just-built module importable:
#        COMMAND ${CMAKE_COMMAND} -E env
#        PYTHONPATH=$<TARGET_FILE_DIR:caset>
#        ${Python_EXECUTABLE} -m pybind11_stubgen caset -o "${STUB_OUT_DIR}"
#        COMMENT "Generating pybind11 stubs -> ${STUB_OUT_DIR}/caset.pyi"
#)

# Install the stub next to the extension in the package
#install(FILES "${STUB_OUT_DIR}/caset.pyi" DESTINATION .)
