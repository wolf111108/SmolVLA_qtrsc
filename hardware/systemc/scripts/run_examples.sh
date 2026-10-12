#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${SYSTEMC_BUILD_DIR:-${example_dir}/../build/systemc}"
cmake_bin="${CMAKE:-cmake}"
systemc_prefix="${SYSTEMC_PREFIX:-${HOME}/.local/opt/systemc-3.0.2}"

if ! command -v "$cmake_bin" >/dev/null 2>&1; then
    echo "CMake not found: $cmake_bin. See hardware/systemc/README.md." >&2
    exit 127
fi
"$cmake_bin" -S "$example_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    "-DCMAKE_PREFIX_PATH=$systemc_prefix"
"$cmake_bin" --build "$build_dir" --parallel "${BUILD_JOBS:-2}"
CTEST_OUTPUT_ON_FAILURE=1 "$cmake_bin" --build "$build_dir" --target test
echo "Examples passed. Waveform: $build_dir/counter.vcd"
