#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <source.c|source.cpp> <output_binary>"
  exit 2
fi

SRC="$1"
OUT="$2"

# Homebrew prefixes
LLVM_PREFIX=$(brew --prefix llvm 2>/dev/null || true)
OMP_PREFIX=$(brew --prefix libomp 2>/dev/null || true)

if [ -z "$LLVM_PREFIX" ] || [ -z "$OMP_PREFIX" ]; then
  echo "Please install dependencies first: brew install llvm libomp" >&2
  exit 1
fi

# Choose C or C++ compiler based on extension
CC="$LLVM_PREFIX/bin/clang"
case "$SRC" in
  *.cpp|*.cxx|*.cc|*.CPP|*.CXX|*.CC) CC="$LLVM_PREFIX/bin/clang++" ;;
esac

CFLAGS="-O3 -fopenmp -I$OMP_PREFIX/include"
LDFLAGS="-L$OMP_PREFIX/lib -Wl,-rpath,$OMP_PREFIX/lib"

echo "$CC $CFLAGS $SRC $LDFLAGS -o $OUT"
exec "$CC" $CFLAGS "$SRC" $LDFLAGS -o "$OUT"

