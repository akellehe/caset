cmake_minimum_required(VERSION 3.18)
option(CASET_CUDA "Build with CUDA GPU acceleration for the Regge solver" ON)
if (DEFINED ENV{CASET_CUDA} AND NOT "$ENV{CASET_CUDA}" STREQUAL "")
    set(CASET_CUDA $ENV{CASET_CUDA})
endif()

if (CASET_CUDA)
    # Help CMake find nvcc at the standard /usr/local/cuda path
    if (NOT CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        project(caset LANGUAGES CXX CUDA)
    else()
        message(WARNING "CUDA requested but nvcc not found — falling back to CPU.")
        set(CASET_CUDA OFF)
        project(caset LANGUAGES CXX)
    endif()
else()
    project(caset LANGUAGES CXX)
endif()

option(CASET_ASAN "Build caset with AddressSanitizer/UBSan" $ENV{CASET_ASAN})
if(NOT CASET_ASAN)
    set(CASET_ASAN OFF)
endif()
option(CASET_VERBOSE "Build caset with C++ level logging" $ENV{CASET_VERBOSE})
if(NOT CASET_VERBOSE)
    set(CASET_VERBOSE OFF)
endif()
option(CASET_ASSERTIONS "Build with assertions all over the place to encourage early failures." $ENV{CASET_ASSERTIONS})
if (NOT CASET_ASSERTIONS)
    set(CASET_ASSERTIONS OFF)
endif()
option(CASET_PROFILE "Build with -g -O1 to keep symbol names for stack traces" $ENV{CASET_PROFILE})
if (NOT CASET_PROFILE)
    set(CASET_PROFILE OFF)
endif()

# ---- Read version from pyproject.toml ----
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_CONTENTS)
string(REGEX MATCH "version = \"([0-9]+\\.[0-9]+\\.[0-9]+[^\"]*)" _ "${PYPROJECT_CONTENTS}")
set(CASET_VERSION "${CMAKE_MATCH_1}")
message(STATUS "caset version: ${CASET_VERSION}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Python & pybind11 ----
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
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
    add_compile_definitions(CASET_HAS_OSYNCSTREAM=1)
else()
    add_compile_definitions(CASET_HAS_OSYNCSTREAM=0)
endif()



# ---- <Define Sources> ----
file(GLOB_RECURSE CASET_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
)
# ---- </Define Sources> ----

# ---- <Build CUDA kernels as static library> ----
if (CASET_CUDA)
    file(GLOB_RECURSE CASET_CUDA_SOURCES
            CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    )
    add_library(caset_cuda STATIC ${CASET_CUDA_SOURCES})
    set_target_properties(caset_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(caset_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(caset_cuda PRIVATE CUDA::cudart)
endif()

# ---- <Build the Python module> ----
pybind11_add_module(_caset MODULE ${CASET_SOURCES})

if (CASET_CUDA)
    message(STATUS "CUDA acceleration turned ON.")
    target_compile_definitions(_caset PRIVATE CASET_CUDA=1)
    target_link_libraries(_caset PRIVATE caset_cuda CUDA::cudart)
else()
    message(STATUS "CUDA acceleration turned OFF.")
endif()

if(CASET_ASAN)
    message(STATUS "ASAN/UBSAN support ON")
    target_compile_options(_caset PRIVATE
            -O1 -g3
            -fno-omit-frame-pointer
            -fsanitize=address,undefined
            -fno-sanitize-recover=address,undefined
            -fno-optimize-sibling-calls
    )
    target_link_options(_caset PRIVATE
            -fsanitize=address,undefined
    )
    # Optional: make ASan reports symbolize nicely
    target_compile_definitions(_caset PRIVATE ASAN_ENABLED=1)
else()
    message(STATUS "ASAN/UBSAN support OFF")
endif()

set(SOURCES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(_caset PRIVATE
    SOURCES_ROOT="${SOURCES_ROOT}"
    CASET_VERSION="${CASET_VERSION}"
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CASET_PROFILE)
    message(STATUS "DEBUGGING (-O1, -g3) Support ON.")
    target_compile_options(_caset PRIVATE -O1 -g3 -fno-omit-frame-pointer -fno-optimize-sibling-calls)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "DEBUGGING (-O1, -g3) Support OFF, will use -O3.")
    target_compile_options(_caset PRIVATE -O3)
endif()

if (CASET_VERBOSE)
    message(STATUS "Logging turned ON.")
    target_compile_definitions(_caset PRIVATE CASET_VERBOSE=1)
else()
    message(STATUS "Logging turned OFF.")
endif()

if (CASET_ASSERTIONS)
    message(STATUS "Assertions turned ON.")
    target_compile_definitions(_caset PRIVATE CASET_ASSERTIONS=1)
else()
    message(STATUS "Assertions turned OFF.")
endif()


target_include_directories(_caset
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(_caset PRIVATE ZLIB::ZLIB)

set_target_properties(_caset PROPERTIES
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF
)
# ---- </Build the Python module> ----

# Install the C extension inside the caset/ package directory
install(TARGETS _caset
        LIBRARY DESTINATION caset
        RUNTIME DESTINATION caset
        ARCHIVE DESTINATION caset
        BUNDLE  DESTINATION caset
)

if(NOT CASET_ASAN)
# --- Generate Python stubs for caset (.pyi) ---
set(STUB_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/stubs")

add_custom_command(
        TARGET _caset POST_BUILD
        BYPRODUCTS "${STUB_OUT_DIR}/_caset.pyi"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${STUB_OUT_DIR}"
        # Make the just-built module importable:
        COMMAND ${Python_EXECUTABLE} -c
            "import pybind11_stubgen" && ${CMAKE_COMMAND} -E env
            PYTHONPATH=$<TARGET_FILE_DIR:_caset>
            ${Python_EXECUTABLE} -m pybind11_stubgen _caset -o "${STUB_OUT_DIR}"
                --ignore-invalid-expressions="caset::.*"
            || ${CMAKE_COMMAND} -E echo "Warning: pybind11_stubgen not found, skipping stub generation"
        COMMENT "Generating pybind11 stubs -> ${STUB_OUT_DIR}/_caset.pyi"
)

# Install the stub next to the extension in the package
install(FILES "${STUB_OUT_DIR}/_caset.pyi" DESTINATION caset OPTIONAL)
endif()