#!/bin/bash

export ASAN="$(gcc -print-file-name=libasan.so)"
export LIBSTDCXX="$(g++ -print-file-name=libstdc++.so)"
export LD_PRELOAD="$ASAN:$LIBSTDCXX" \
export ASAN_OPTIONS="abort_on_error=1:detect_leaks=0:fast_unwind_on_malloc=0"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" 
export LD_PRELOAD="$ASAN":"$LIBSTDCXX"

echo $1

gdb --args python -m pytest -q $1 -s
 
