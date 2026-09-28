#!/usr/bin/env bash
# Run the self-checking DMA command protocol example; no Python/GPU required.
set -euo pipefail

hardware_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
iverilog_bin="${IVERILOG:-iverilog}"
vvp_bin="${VVP:-vvp}"

for simulator_bin in "$iverilog_bin" "$vvp_bin"; do
    if ! command -v "$simulator_bin" >/dev/null 2>&1; then
        printf 'Missing simulator executable: %s\nInstall Icarus Verilog (iverilog + vvp), then rerun this script.\n' "$simulator_bin" >&2
        exit 127
    fi
done

build_dir="$hardware_dir/build/dma_cmd"
mkdir -p -- "$build_dir"
cd -- "$build_dir"

"$iverilog_bin" -g2012 -Wall -s tb_dma_cmd -o sim.vvp \
    "$hardware_dir/models/dma_cmd_model.sv" \
    "$hardware_dir/tb/tb_dma_cmd.sv" 2>&1 | tee compile.log
"$vvp_bin" sim.vvp 2>&1 | tee run.log

if ! grep -Fxq 'PASS: commands=7 responses=6 reset_aborted=1' run.log; then
    printf 'Simulation ended without the expected PASS marker.\n' >&2
    exit 1
fi
printf 'Simulation outputs: %s\n' "$build_dir"
