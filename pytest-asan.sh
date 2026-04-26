#!/bin/bash

export TESSERA_ASAN=ON
export TESSERA_ASSERTIONS=ON
export TESSERA_VERBOSE=ON
export CC=gcc
export CXX=g++
export ASAN="$(gcc -print-file-name=libasan.so)"
export LIBSTDCXX="$(g++ -print-file-name=libstdc++.so)"
export LD_PRELOAD="$ASAN:$LIBSTDCXX"
export ASAN_OPTIONS="abort_on_error=1:detect_leaks=0:fast_unwind_on_malloc=0:strict_string_checks=1"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"
export LD_PRELOAD="$ASAN":"$LIBSTDCXX"
export LOG_LEVEL=10

echo "Preloading libraries: $LD_PRELOAD"
echo $1

python -m pytest -q $1 -s
 
