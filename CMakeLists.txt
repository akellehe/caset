cmake_minimum_required(VERSION 3.18)
project(caset LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Python & pybind11 ----
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Option to override ABI if needed
set(TORCH_CXX11_ABI "1" CACHE STRING "_GLIBCXX_USE_CXX11_ABI value for building against torch (default 1)")

# ---- Locate Python's torch package WITHOUT importing it ----
# 1) Get torch package directory
execute_process(
        COMMAND ${Python_EXECUTABLE} -c
        "import importlib.util, pathlib, sys; s=importlib.util.find_spec('torch'); \
d=pathlib.Path(s.origin).parent if s and s.origin else None; \
print(str(d/'include') if d else ''); print(str(d/'lib') if d else '')"
        OUTPUT_VARIABLE _torch_out
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" ";" _TORCH_PARTS "${_torch_out}")

if(NOT _TORCH_PARTS OR _TORCH_PARTS STREQUAL "")
    message(FATAL_ERROR "Python '${Python_EXECUTABLE}' cannot find the 'torch' package. Activate the venv that has torch installed, or install torch there.")
endif()

list(LENGTH _TORCH_PARTS _TORCH_LEN)
if(_TORCH_LEN LESS 2)
    message(FATAL_ERROR "Failed to locate torch include/lib via importlib. Got: '${_torch_out}'")
endif()

list(GET _TORCH_PARTS 0 TORCH_INCLUDE_DIR)
list(GET _TORCH_PARTS 1 TORCH_LIB_DIR)

# On a mac it's libc10.dylib; on linux it's libc10.so
if(NOT EXISTS "${TORCH_LIB_DIR}/libc10.dylib" AND NOT EXISTS "${TORCH_LIB_DIR}/libc10.so")
    message(FATAL_ERROR "Torch lib dir '${TORCH_LIB_DIR}' does not contain libc10.dylib or libc10.so. You are probably mixing environments or have a broken install.")
endif()

message(STATUS "Torch include: ${TORCH_INCLUDE_DIR}")
message(STATUS "Torch lib:     ${TORCH_LIB_DIR}")
message(STATUS "_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} (override with -DTORCH_CXX11_ABI=0|1)")

# ---- Sources ----
set(CASET_SOURCES
        src/bindings.cpp
        src/Edge.cpp
        src/EdgeList.cpp
        src/Metric.cpp
        src/Signature.cpp
        src/Simplex.cpp
        src/Spacetime.cpp
        src/Vertex.cpp
        src/VertexList.cpp
)

# ---- Build module ----
pybind11_add_module(caset MODULE ${CASET_SOURCES})

target_include_directories(caset
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIR}/torch/csrc/api/include
)

# Link ONLY against torch from site-packages
target_link_directories(caset PRIVATE ${TORCH_LIB_DIR})
target_link_libraries(caset PRIVATE torch torch_cpu c10)

# Match PyTorch’s C++11 ABI (default 1; override if needed)
target_compile_definitions(caset PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})

# RPATH so libc10/libtorch are found at runtime
set_target_properties(caset PROPERTIES
        BUILD_RPATH "${TORCH_LIB_DIR}"
        INSTALL_RPATH "${TORCH_LIB_DIR};$ORIGIN;$ORIGIN/.."
        BUILD_WITH_INSTALL_RPATH OFF
)

# Extra belt-and-suspenders rpath
if(UNIX AND NOT APPLE)
    target_link_options(caset PRIVATE "-Wl,-rpath,${TORCH_LIB_DIR}")
endif()

# Install next to your package
install(TARGETS caset
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
        ARCHIVE DESTINATION .
        BUNDLE  DESTINATION .
)
