# Build ITensor v3 (third_party/itensor) as a static library that the rest of
# tessera can link against. Mirrors what third_party/itensor/itensor/Makefile
# does, but in CMake.
#
# ITensor's upstream build:
#   1. configure step writes itensor/config.h with #define PLATFORM_<name>
#      (selecting which LAPACK calling convention to use)
#   2. compiles ~35 .cc files into a static lib/libitensor.a
#   3. expects callers to add -DPLATFORM_<name> on the command line
#
# Our CMake equivalent:
#   1. configure_file expands cmake/itensor_config.h.in into a generated
#      itensor/config.h under ${CMAKE_BINARY_DIR}/itensor_generated/
#   2. add_library(itensor STATIC ...) compiles the same .cc list
#   3. target_compile_definitions adds the matching PLATFORM_* macro
#
# Why the generated config.h goes into a separate include path rather than
# directly into third_party/itensor: keeps the submodule checkout clean
# (no untracked file inside it) and lets multiple build configurations
# coexist by writing to their own build dirs.

function(tessera_build_itensor)
    set(ITENSOR_ROOT "${CMAKE_SOURCE_DIR}/third_party/itensor")
    if(NOT EXISTS "${ITENSOR_ROOT}/itensor/itensor.h")
        message(FATAL_ERROR
            "third_party/itensor not populated. Run: "
            "git submodule update --init --recursive")
    endif()

    # Cache variable so users can override with -DITENSOR_PLATFORM=openblas
    # at configure time. Default "lapack" matches the most common Linux
    # setup (libblas + liblapack via the system package manager).
    set(ITENSOR_PLATFORM "lapack" CACHE STRING
        "ITensor BLAS/LAPACK backend: lapack | openblas | mkl | acml | macos")
    set_property(CACHE ITENSOR_PLATFORM PROPERTY STRINGS
        lapack openblas mkl acml macos)

    set(ITENSOR_GEN_DIR "${CMAKE_BINARY_DIR}/itensor_generated")
    file(MAKE_DIRECTORY "${ITENSOR_GEN_DIR}/itensor")
    configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/itensor_config.h.in"
        "${ITENSOR_GEN_DIR}/itensor/config.h"
        @ONLY)

    # ─── ITensor source list ────────────────────────────────────────────
    # Mirrors third_party/itensor/itensor/Makefile's SOURCES list with HDF5
    # serialization disabled (we don't need it; saves an HDF5 dependency).
    # If you bump the ITensor submodule pin and the upstream Makefile adds
    # / removes a .cc, update this list to match.
    set(_itensor_sources
        util/args.cc
        util/input.cc
        util/cputime.cc
        tensor/lapack_wrap.cc
        tensor/vec.cc
        tensor/mat.cc
        tensor/gemm.cc
        tensor/algs.cc
        tensor/contract.cc
        itdata/dense.cc
        itdata/combiner.cc
        itdata/diag.cc
        itdata/qdense.cc
        itdata/qcombiner.cc
        itdata/qdiag.cc
        itdata/scalar.cc
        qn.cc
        tagset.cc
        index.cc
        indexset.cc
        itensor.cc
        spectrum.cc
        decomp.cc
        hermitian.cc
        svd.cc
        global.cc
        mps/mps.cc
        mps/mpsalgs.cc
        mps/mpo.cc
        mps/mpoalgs.cc
        mps/autompo.cc)
    list(TRANSFORM _itensor_sources PREPEND "${ITENSOR_ROOT}/itensor/")

    add_library(itensor STATIC ${_itensor_sources})
    # SYSTEM include hides ITensor's headers from -Wall on consumer code.
    set(ITENSOR_TDVP_ROOT "${CMAKE_SOURCE_DIR}/third_party/itensor_tdvp")
    target_include_directories(itensor SYSTEM PUBLIC
        "${ITENSOR_ROOT}"        # for "itensor/all.h", "itensor/mps/dmrg.h", etc.
        "${ITENSOR_GEN_DIR}"     # for the generated "itensor/config.h"
        "${ITENSOR_TDVP_ROOT}")  # for "tdvp.h" — header-only TDVP add-on
                                 # at https://github.com/ITensor/TDVP, since
                                 # ITensor v3 core has no built-in TDVP.
    # ITensor v3 requires C++17; we propagate this PUBLIC so consumers
    # building against `itensor` get the floor automatically. (tessera itself
    # uses C++20; the higher std subsumes the requirement.)
    target_compile_features(itensor PUBLIC cxx_std_17)
    set_target_properties(itensor PROPERTIES
        POSITION_INDEPENDENT_CODE ON  # static lib will link into _tessera.so
        CXX_EXTENSIONS OFF)
    target_compile_definitions(itensor PUBLIC
        "PLATFORM_${ITENSOR_PLATFORM}"
        "__ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES=0")

    # ITensor's own source has minor compiler-warning noise (deprecated
    # `register`, unused warnings inside its template machinery). PRIVATE
    # silences them only while compiling ITensor — consumer warnings on
    # tessera code are unaffected.
    target_compile_options(itensor PRIVATE
        -Wno-deprecated-declarations
        -Wno-unused-but-set-variable
        -Wno-unused-variable
        -Wno-unused-parameter
        -Wno-sign-compare)

    # ─── BLAS / LAPACK linkage ──────────────────────────────────────────
    # ITensor calls into Fortran-style LAPACK routines (dgemm_, zheev_,
    # etc.). Different backends use different name-mangling and calling
    # conventions, hence the per-platform branches.
    if(ITENSOR_PLATFORM STREQUAL "lapack")
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
        target_link_libraries(itensor PUBLIC LAPACK::LAPACK BLAS::BLAS pthread)
    elseif(ITENSOR_PLATFORM STREQUAL "openblas")
        find_package(BLAS REQUIRED)
        target_link_libraries(itensor PUBLIC BLAS::BLAS pthread)
    elseif(ITENSOR_PLATFORM STREQUAL "mkl")
        find_package(MKL REQUIRED)
        target_link_libraries(itensor PUBLIC MKL::MKL pthread)
    elseif(ITENSOR_PLATFORM STREQUAL "macos")
        target_link_libraries(itensor PUBLIC "-framework Accelerate")
    else()
        message(FATAL_ERROR "ITENSOR_PLATFORM=${ITENSOR_PLATFORM} not yet wired")
    endif()
endfunction()
