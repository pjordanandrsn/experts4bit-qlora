#!/usr/bin/env bash
# bench/sc1/llamacpp/build_harness.sh -- build nll_teacher_forced against an ALREADY-BUILT llama.cpp tree.
#
#   build_harness.sh <llama.cpp src dir> [<build dir> = <src>/build] [<output binary> = <build>/bin/nll_teacher_forced]
#
# Links the shared libllama + libggml + libggml-base from <build>/bin (llama.cpp's default BUILD_SHARED_LIBS=ON layout,
# CMAKE_LIBRARY_OUTPUT_DIRECTORY = build/bin) with an rpath to that directory, so the harness runs the SAME CUDA/CPU
# backends the pinned tree's llama-server runs. Only headers from the tree are used (include/, ggml/include/, the vendored
# nlohmann/json); the pinned commit is baked in as SC1_LLAMA_COMMIT / SC1_LLAMA_TAG from `git -C <src>`.
# Refuses a static-only build (no libllama.{so,dylib}) and a source tree that is not a git checkout.
set -euo pipefail

SRC=${1:?usage: build_harness.sh <llama.cpp src> [<build dir>] [<out binary>]}
BUILD=${2:-$SRC/build}
OUT=${3:-$BUILD/bin/nll_teacher_forced}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CXX=${CXX:-c++}

[ -f "$SRC/include/llama.h" ] || { echo "build_harness: $SRC has no include/llama.h" >&2; exit 2; }
[ -f "$SRC/vendor/nlohmann/json.hpp" ] || { echo "build_harness: $SRC has no vendor/nlohmann/json.hpp" >&2; exit 2; }
COMMIT=$(git -C "$SRC" rev-parse HEAD) || { echo "build_harness: $SRC is not a git checkout (the pin must be verifiable)" >&2; exit 2; }
TAG=$(git -C "$SRC" describe --tags --exact-match 2>/dev/null || echo unknown)

LIBDIR="$BUILD/bin"
if [ ! -e "$LIBDIR/libllama.so" ] && [ ! -e "$LIBDIR/libllama.dylib" ]; then
    echo "build_harness: no libllama.so/.dylib in $LIBDIR -- build the tree first with BUILD_SHARED_LIBS=ON (the default):" >&2
    echo "  cmake -S $SRC -B $BUILD ... && cmake --build $BUILD --target llama" >&2
    exit 2
fi

mkdir -p "$(dirname "$OUT")"
set -x
"$CXX" -std=c++17 -O2 -Wall -Wextra \
    -I"$SRC/include" -I"$SRC/ggml/include" -I"$SRC/vendor" \
    -DSC1_LLAMA_COMMIT="\"$COMMIT\"" -DSC1_LLAMA_TAG="\"$TAG\"" \
    "$HERE/nll_teacher_forced.cpp" -o "$OUT" \
    -L"$LIBDIR" -lllama -lggml -lggml-base -Wl,-rpath,"$LIBDIR" -pthread
set +x

{
    echo "harness: $OUT"
    echo "source: $HERE/nll_teacher_forced.cpp"
    echo "llama.cpp: $SRC"
    echo "llama_commit: $COMMIT"
    echo "llama_tag: $TAG"
    echo "libdir: $LIBDIR"
    echo "cxx: $("$CXX" --version 2>/dev/null | head -1)"
    echo "built_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT.buildinfo"
echo "build_harness: built $OUT against llama.cpp $COMMIT ($TAG)"
