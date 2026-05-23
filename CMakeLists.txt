cmake_minimum_required(VERSION 3.18)
option(TESSERA_CUDA "Build with CUDA GPU acceleration for the Regge solver" ON)
if (DEFINED ENV{TESSERA_CUDA} AND NOT "$ENV{TESSERA_CUDA}" STREQUAL "")
    set(TESSERA_CUDA $ENV{TESSERA_CUDA})
endif()

if (TESSERA_CUDA)
    # Help CMake find nvcc at the standard /usr/local/cuda path
    if (NOT CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        project(tessera LANGUAGES CXX CUDA)
    else()
        message(WARNING "CUDA requested but nvcc not found — falling back to CPU.")
        set(TESSERA_CUDA OFF)
        project(tessera LANGUAGES CXX)
    endif()
else()
    project(tessera LANGUAGES CXX)
endif()

# ---- Compiler cache (ccache / sccache) ----
# Content-addressed object cache: makes clean rebuilds and per-wheel-tag
# builds near-instant — in particular ITensor's ~35 translation units,
# whose objects hash identically across Python versions and build dirs.
# Auto-enabled when found; set TESSERA_CCACHE=0 to opt out.
if (NOT "$ENV{TESSERA_CCACHE}" STREQUAL "0")
    find_program(CCACHE_PROGRAM NAMES ccache sccache)
    if (CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        message(STATUS "Compiler cache enabled: ${CCACHE_PROGRAM}")
    else()
        message(STATUS "Compiler cache not found — install ccache for fast rebuilds.")
    endif()
endif()

option(TESSERA_ASAN "Build tessera with AddressSanitizer/UBSan" $ENV{TESSERA_ASAN})
if(NOT TESSERA_ASAN)
    set(TESSERA_ASAN OFF)
endif()
option(TESSERA_VERBOSE "Build tessera with C++ level logging" $ENV{TESSERA_VERBOSE})
if(NOT TESSERA_VERBOSE)
    set(TESSERA_VERBOSE OFF)
endif()
option(TESSERA_ASSERTIONS "Build with assertions all over the place to encourage early failures." $ENV{TESSERA_ASSERTIONS})
if (NOT TESSERA_ASSERTIONS)
    set(TESSERA_ASSERTIONS OFF)
endif()
option(TESSERA_PROFILE "Build with -g -O1 to keep symbol names for stack traces" $ENV{TESSERA_PROFILE})
if (NOT TESSERA_PROFILE)
    set(TESSERA_PROFILE OFF)
endif()
# ---- Quantum subsystem (Schwinger / MPS / DMRG): auto-detected ----
# Mirrors TESSERA_CUDA above: built automatically when its prerequisites
# are present, with an env var to override. Prerequisites are (1) the
# ITensor submodule checked out under third_party/itensor and (2) a
# BLAS/LAPACK backend. TESSERA_QUANTUM=1 forces it on — fetching the
# ITensor submodule automatically when it is missing; a missing BLAS/LAPACK
# is still a hard error. TESSERA_QUANTUM=0 forces it off.
#
# Kept as a plain variable rather than option(): it is recomputed on every
# configure, so checking out the submodule and re-running the build picks
# the subsystem up without needing a fresh build directory.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/itensor/itensor/itensor.h")
    set(_quantum_itensor_present TRUE)
else()
    set(_quantum_itensor_present FALSE)
endif()
if (APPLE)
    set(_quantum_lapack_present TRUE)   # Accelerate framework ships with macOS
else()
    find_package(LAPACK QUIET)
    set(_quantum_lapack_present ${LAPACK_FOUND})
endif()

if (DEFINED ENV{TESSERA_QUANTUM} AND NOT "$ENV{TESSERA_QUANTUM}" STREQUAL "")
    set(TESSERA_QUANTUM $ENV{TESSERA_QUANTUM})        # explicit override
elseif (_quantum_itensor_present AND _quantum_lapack_present)
    set(TESSERA_QUANTUM ON)                           # auto-detect
else()
    set(TESSERA_QUANTUM OFF)
endif()

if (TESSERA_QUANTUM)
    set(TESSERA_QUANTUM ON)
    # The ITensor submodule is the one prerequisite we can satisfy for the
    # user: when quantum is explicitly requested (TESSERA_QUANTUM=1) but the
    # submodule is missing, fetch it with git instead of erroring out. The
    # auto-detect path never reaches here without the submodule — only an
    # explicit request does — so a bare `pip install -e .` stays opt-in.
    if (NOT _quantum_itensor_present)
        find_package(Git QUIET)
        if (NOT Git_FOUND OR NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
            message(FATAL_ERROR
                "TESSERA_QUANTUM=1 but third_party/itensor is not checked out, "
                "and it cannot be fetched automatically (git not found, or "
                "this is not a git checkout).\n"
                "  Install tessera from a git clone, or fetch it manually:\n"
                "    git submodule update --init --recursive")
        endif()
        message(STATUS "TESSERA_QUANTUM=1: fetching the ITensor submodule "
                       "(one-time network download)...")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
                    -- third_party/itensor third_party/itensor_tdvp
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RESULT_VARIABLE _quantum_submodule_result)
        if (NOT _quantum_submodule_result EQUAL 0)
            message(FATAL_ERROR
                "Failed to fetch the ITensor submodule (git exited "
                "${_quantum_submodule_result}).\n"
                "  Check your network, then re-run the build — or fetch it "
                "manually:\n"
                "    git submodule update --init --recursive")
        endif()
        if (NOT EXISTS
                "${CMAKE_CURRENT_SOURCE_DIR}/third_party/itensor/itensor/itensor.h")
            message(FATAL_ERROR
                "ITensor submodule fetch reported success but "
                "third_party/itensor is still empty.")
        endif()
        set(_quantum_itensor_present TRUE)
        message(STATUS "ITensor submodule ready.")
    endif()
    if (NOT _quantum_lapack_present)
        message(FATAL_ERROR
            "TESSERA_QUANTUM is on but no BLAS/LAPACK backend was found.\n"
            "  Install one (e.g. apt-get install liblapack-dev libblas-dev),\n"
            "  or set TESSERA_QUANTUM=0 to skip the quantum subsystem.")
    endif()
else()
    set(TESSERA_QUANTUM OFF)
    if (NOT _quantum_itensor_present)
        message(STATUS "Quantum subsystem OFF — third_party/itensor not checked "
                       "out (run `git submodule update --init --recursive`).")
    elseif (NOT _quantum_lapack_present)
        message(STATUS "Quantum subsystem OFF — no BLAS/LAPACK backend found.")
    else()
        message(STATUS "Quantum subsystem OFF — disabled via TESSERA_QUANTUM=0.")
    endif()
endif()
# ---- Read version from pyproject.toml ----
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_CONTENTS)
string(REGEX MATCH "version = \"([0-9]+\\.[0-9]+\\.[0-9]+[^\"]*)" _ "${PYPROJECT_CONTENTS}")
set(TESSERA_VERSION "${CMAKE_MATCH_1}")
message(STATUS "tessera version: ${TESSERA_VERSION}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Python & pybind11 ----
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Locate pybind11 >= 3.0. tessera's bindings use the v3 API, so an older
# v2 must never be selected — and the distro's ``pybind11-dev`` installs a
# config into ``/usr/lib/cmake``, a default find_package() search path.
#
# The ``3.0`` floor is enforced *by* find_package: a too-old config is
# skipped, and if nothing satisfies it the build fails loudly instead of
# silently linking the wrong ABI. CMake's CONFIG-mode search order then
# covers both supported build paths without any manual pinning:
#   * ``pip install`` — scikit-build-core puts the pinned pybind11 (from
#     build-system.requires) on CMAKE_PREFIX_PATH; searched ahead of HINTS.
#   * direct ``cmake`` — the active interpreter's own pybind11, supplied as
#     the HINTS fallback below. Empty (and harmless) if none is installed.
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_hint
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
find_package(pybind11 3.0 CONFIG REQUIRED HINTS "${_pybind11_hint}")
message(STATUS "pybind11 ${pybind11_VERSION} (${pybind11_DIR})")
find_package(ZLIB REQUIRED)

include(CheckCXXSourceCompiles)

check_cxx_source_compiles("
  #include <syncstream>
  int main() {
      std::osyncstream out(std::cout);
      out << \"test\" << std::endl;
      return 0;
  }
" HAVE_STD_OSYNCSTREAM)

if (HAVE_STD_OSYNCSTREAM)
    add_compile_definitions(TESSERA_HAS_OSYNCSTREAM=1)
else()
    add_compile_definitions(TESSERA_HAS_OSYNCSTREAM=0)
endif()



# ---- <Define source partitions> ------------------------------------------
# Two libraries:
#   tessera_core  — pybind-free C++ implementation: mesh primitives,
#                 spacetime, observables, Poset infrastructure, etc.
#                 Linked into _tessera and (when enabled) tessera_quantum.
#   tessera_quantum (when TESSERA_QUANTUM=ON) — Schwinger / DMRG / TDVP code,
#                 also pybind-free, links tessera_core PUBLIC for shared
#                 mesh / Poset types.
# Plus a thin pybind11 module:
#   _tessera       — bindings.cpp + quantum/bindings.cpp; links the two libs.
file(GLOB_RECURSE TESSERA_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

# Quantum lives in its own static lib; exclude its sources from tessera_core.
set(TESSERA_CORE_SOURCES ${TESSERA_SOURCES})
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX "/(src|include)/quantum/")
# InteractionSimulation lives in src/simulations/ for architectural
# alignment with CDT (#44), but depends on Eigen + ITensor and so belongs
# in tessera_quantum, not tessera_core.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX
    "/(src|include)/simulations/InteractionSimulation\\.(cpp|h)$")
# bindings.cpp is the pybind11 entry point — goes into _tessera, not tessera_core.
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX "/src/bindings\\.cpp$")
# Per-subsystem Bindings.cpp files also belong to _tessera, not tessera_core
# (they include pybind11 / Python.h).
list(FILTER TESSERA_CORE_SOURCES EXCLUDE REGEX "/src/[^/]+/Bindings\\.cpp$")

# Sources for the pybind11 module. quantum bindings only when enabled.
set(TESSERA_PYBIND_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mesh/Bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spacetime/Bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/observables/Bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/simulations/Bindings.cpp")
if(TESSERA_QUANTUM)
    list(APPEND TESSERA_PYBIND_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/quantum/Bindings.cpp")
endif()
# ---- </Define source partitions> -----------------------------------------

# ---- <Build CUDA kernels as static library> ----
if (TESSERA_CUDA)
    file(GLOB_RECURSE TESSERA_CUDA_SOURCES
            CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    )
    add_library(tessera_cuda STATIC ${TESSERA_CUDA_SOURCES})
    set_target_properties(tessera_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(tessera_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(tessera_cuda PRIVATE CUDA::cudart)
endif()

# ---- <tessera_core static library> ----------------------------------------
# Holds all the C++ implementation that has no pybind dependency. Both
# the Python module (_tessera) and the optional tessera_quantum library link
# this so we don't compile the same .cpp's twice.
add_library(tessera_core STATIC ${TESSERA_CORE_SOURCES})
target_include_directories(tessera_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(tessera_core PUBLIC ZLIB::ZLIB)

# Optional OpenMP for the spectral-graph heat-kernel parallel loop and
# similar embarrassingly-parallel hot paths. Bit-identical to the
# serial code at OMP_NUM_THREADS=1 (the default in our 10-CPU-budget
# scan policy); only kicks in for solo / exploratory runs that set
# OMP_NUM_THREADS > 1 explicitly. Required so `#pragma omp parallel`
# compiles and links cleanly; if OpenMP is absent the pragmas are
# silently no-ops (we don't gate compilation on it).
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tessera_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found (${OpenMP_CXX_VERSION}); spectral-graph "
                   "heat-kernel will respect OMP_NUM_THREADS.")
else()
    message(STATUS "OpenMP not found; #pragma omp directives in "
                   "tessera_core will compile as no-ops.")
endif()
# tessera_core is pybind-free: the binding code (which uses pybind11) lives
# only in src/bindings.cpp and src/quantum/Bindings.cpp, both compiled
# directly into the _tessera Python module. Earlier versions of tessera had
# stray `#include <pybind11/pybind11.h>` in mesh/spacetime headers that
# served no purpose; those have been removed so tessera_core no longer
# needs Python at link time. Test executables linking tessera_quantum →
# tessera_core get clean C++ symbols only.
set_target_properties(tessera_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

# Compile definitions that must reach consumers (so #ifdef SOURCES_ROOT,
# TESSERA_VERSION, TESSERA_VERBOSE, etc. work in the Python bindings TU and
# in test executables).
target_compile_definitions(tessera_core PUBLIC
    SOURCES_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    TESSERA_VERSION="${TESSERA_VERSION}")

if (TESSERA_CUDA)
    message(STATUS "CUDA acceleration turned ON.")
    target_compile_definitions(tessera_core PUBLIC TESSERA_CUDA=1)
    target_link_libraries(tessera_core PUBLIC tessera_cuda CUDA::cudart)
else()
    message(STATUS "CUDA acceleration turned OFF.")
endif()

if(TESSERA_ASAN)
    message(STATUS "ASAN/UBSAN support ON")
    target_compile_options(tessera_core PUBLIC
            -O1 -g3
            -fno-omit-frame-pointer
            -fsanitize=address,undefined
            -fno-sanitize-recover=address,undefined
            -fno-optimize-sibling-calls)
    target_link_options(tessera_core PUBLIC
            -fsanitize=address,undefined)
    target_compile_definitions(tessera_core PUBLIC ASAN_ENABLED=1)
else()
    message(STATUS "ASAN/UBSAN support OFF")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR TESSERA_PROFILE)
    message(STATUS "DEBUGGING (-O1, -g3) Support ON.")
    target_compile_options(tessera_core PRIVATE
        -O1 -g3 -fno-omit-frame-pointer -fno-optimize-sibling-calls)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "DEBUGGING (-O1, -g3) Support OFF, will use -O3.")
    target_compile_options(tessera_core PRIVATE -O3)
endif()

if (TESSERA_VERBOSE)
    message(STATUS "Logging turned ON.")
    target_compile_definitions(tessera_core PUBLIC TESSERA_VERBOSE=1)
else()
    message(STATUS "Logging turned OFF.")
endif()

if (TESSERA_ASSERTIONS)
    message(STATUS "Assertions turned ON.")
    target_compile_definitions(tessera_core PUBLIC TESSERA_ASSERTIONS=1)
else()
    message(STATUS "Assertions turned OFF.")
endif()
# ---- </tessera_core static library> --------------------------------------

# ---- <Build the Python module> ----
pybind11_add_module(_tessera MODULE ${TESSERA_PYBIND_SOURCES})
target_link_libraries(_tessera PRIVATE tessera_core)

set_target_properties(_tessera PROPERTIES
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF)
# ---- </Build the Python module> ----

# Install the C extension inside the tessera/ package directory
install(TARGETS _tessera
        LIBRARY DESTINATION tessera
        RUNTIME DESTINATION tessera
        ARCHIVE DESTINATION tessera
        BUNDLE  DESTINATION tessera
)

# ---- <Quantum subsystem (ITensor + Schwinger MPO)> ----
if(TESSERA_QUANTUM)
    message(STATUS "Quantum subsystem turned ON (ITensor + DMRG).")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildITensor.cmake)
    tessera_build_itensor()

    file(GLOB_RECURSE TESSERA_QUANTUM_SOURCES
            CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/quantum/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/quantum/*.hpp")
    # InteractionSimulation lives in src/simulations/ for architectural
    # alignment with CDT (#44) but its Eigen/ITensor dependency makes it
    # part of tessera_quantum, not tessera_core.
    list(APPEND TESSERA_QUANTUM_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/simulations/InteractionSimulation.cpp")
    # bindings.cpp uses pybind11 + Python.h and goes into _tessera, not the
    # pybind-free static lib.
    list(FILTER TESSERA_QUANTUM_SOURCES EXCLUDE REGEX "/src/quantum/Bindings\\.cpp$")
    add_library(tessera_quantum STATIC ${TESSERA_QUANTUM_SOURCES})
    target_include_directories(tessera_quantum PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    # PUBLIC link to tessera_core: tessera::Poset / Vertex / EdgeList live in
    # tessera_core, and quantum's Poset-using headers expose them in their
    # public API. PUBLIC propagates symbols to test executables too.
    target_link_libraries(tessera_quantum PUBLIC itensor Eigen3::Eigen tessera_core)
    set_target_properties(tessera_quantum PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON)

    enable_testing()
    file(GLOB TESSERA_QUANTUM_TESTS
            CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/quantum/*.cpp")
    foreach(_test_src ${TESSERA_QUANTUM_TESTS})
        get_filename_component(_test_name "${_test_src}" NAME_WE)
        add_executable(${_test_name} "${_test_src}")
        # tessera_quantum PUBLIC-links tessera_core, so tests get its symbols
        # transitively without an explicit tessera_core link here.
        target_link_libraries(${_test_name} PRIVATE tessera_quantum)
        set_target_properties(${_test_name} PROPERTIES
                CXX_STANDARD 20
                CXX_STANDARD_REQUIRED ON)
        add_test(NAME ${_test_name} COMMAND ${_test_name})
    endforeach()

    # Wire the quantum bindings TU (added to TESSERA_PYBIND_SOURCES above)
    # into _tessera by linking against the implementation library and
    # flipping the TESSERA_QUANTUM compile flag so src/bindings.cpp
    # registers the `quantum` submodule.
    target_link_libraries(_tessera PRIVATE tessera_quantum)
    target_compile_definitions(_tessera PRIVATE TESSERA_QUANTUM=1)
endif()  # the auto-detect block above already reports when quantum is OFF
# ---- </Quantum subsystem> ----

if(NOT TESSERA_ASAN)
# --- Generate Python stubs for tessera (.pyi) ---
set(STUB_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/stubs")

add_custom_command(
        TARGET _tessera POST_BUILD
        BYPRODUCTS "${STUB_OUT_DIR}/_tessera.pyi"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${STUB_OUT_DIR}"
        # Make the just-built module importable:
        COMMAND ${Python_EXECUTABLE} -c
            "import pybind11_stubgen" && ${CMAKE_COMMAND} -E env
            PYTHONPATH=$<TARGET_FILE_DIR:_tessera>
            ${Python_EXECUTABLE} -m pybind11_stubgen _tessera -o "${STUB_OUT_DIR}"
                --ignore-invalid-expressions="tessera::.*"
            || ${CMAKE_COMMAND} -E echo "Warning: pybind11_stubgen not found, skipping stub generation"
        COMMENT "Generating pybind11 stubs -> ${STUB_OUT_DIR}/_tessera.pyi"
)

# Install the stub next to the extension in the package
install(FILES "${STUB_OUT_DIR}/_tessera.pyi" DESTINATION tessera OPTIONAL)
endif()
