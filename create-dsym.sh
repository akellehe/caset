#!/bin/zsh

# This file creates a dSYM file that is used by lldb on MacOS to connect symbols to line numbers.

SO=$(python -c "import caset; print(caset.__file__)")
dsymutil "$SO" -o "$SO.dSYM"
