#!/usr/bin/env bash
#
# Build and run the self-checking tiny_gemm testbench.
#
# No Python, GPU or model checkpoint is needed. An HDL simulator must be
# reachable: either Icarus Verilog (iverilog + vvp) or Cadence Xcelium (xrun).
# Where only Xcelium is installed the script loads its environment module
# automatically when xrun is not already on PATH.
#
#   (no arguments)     run both scenarios; the default regression
#   SCENARIO=positive  run only the passing scenario
#   SCENARIO=fault     run only the checker self-test, which must FAIL
#   SIM=iverilog|xrun  force a simulator instead of auto-detecting
#   XCE_MODULE=...     override the Xcelium module name
#   DUMP=1             also write a VCD waveform
#
# Artifacts land in hardware/build/tiny_gemm/ and are not tracked by git.
#
set -uo pipefail

hardware_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
rtl_file="$hardware_dir/rtl/tiny_gemm.sv"
mem_file="$hardware_dir/tb/tiny_gemm_mem.sv"
tb_file="$hardware_dir/tb/tb_tiny_gemm.sv"
build_dir="$hardware_dir/build/tiny_gemm"
xce_module="${XCE_MODULE:-cadence/xcelium/2203_01}"
pass_marker='PASS: tiny_gemm all tests passed'
fault_marker='ERROR: TB_FAULT_INJECT'

for f in "$rtl_file" "$mem_file" "$tb_file"; do
    if [ ! -f "$f" ]; then
        printf 'Missing source file: %s\n' "$f" >&2
        exit 2
    fi
done

rm -rf -- "$build_dir"
mkdir -p -- "$build_dir"
cd -- "$build_dir" || exit 2

# --------------------------------------------------------------- simulator
if ! command -v module >/dev/null 2>&1 && [ -f /usr/share/Modules/init/bash ]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
fi

pick_simulator() {
    local sim="${SIM:-auto}"
    if [ "$sim" != "auto" ]; then
        case "$sim" in
            iverilog|xrun) printf '%s' "$sim"; return ;;
            *) printf 'unsupported SIM value: %s\n' "$sim" >&2; printf 'none'; return ;;
        esac
    fi
    if command -v iverilog >/dev/null 2>&1 && command -v vvp >/dev/null 2>&1; then
        printf 'iverilog'
        return
    fi
    if ! command -v xrun >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
        module load "$xce_module" >/dev/null 2>&1 || true
    fi
    if command -v xrun >/dev/null 2>&1; then
        printf 'xrun'
        return
    fi
    printf 'none'
}

sim="$(pick_simulator)"
if [ "$sim" = "none" ]; then
    printf 'No HDL simulator available.\n' >&2
    printf 'Install Icarus Verilog (iverilog + vvp), or load Xcelium:\n' >&2
    printf '  module load %s\n' "$xce_module" >&2
    exit 127
fi
printf 'Simulator: %s\n' "$sim"

# ------------------------------------------------------------------- run
# $1 = scenario name, $2 = "fault" to enable the checker self-test.
# Defines are accumulated as plain strings on purpose: an empty bash array
# expanded under `set -u` is an error on shells older than 4.4.
run_scenario() {
    local name="$1" inject="$2"
    local defines="" rc=0

    printf '\n=== tiny_gemm scenario: %s ===\n' "$name"

    if [ "$inject" = "fault" ]; then
        if [ "$sim" = "iverilog" ]; then defines="$defines -DTB_FAULT_INJECT"
        else                                defines="$defines +define+TB_FAULT_INJECT"
        fi
    fi
    if [ "${DUMP:-0}" = "1" ]; then
        if [ "$sim" = "iverilog" ]; then defines="$defines -DTB_DUMP"
        else                                defines="$defines +define+TB_DUMP"
        fi
    fi

    if [ "$sim" = "iverilog" ]; then
        # Word splitting of $defines is intended here.
        # shellcheck disable=SC2086
        iverilog -g2012 -Wall -s tb_tiny_gemm -o sim.vvp $defines \
            "$rtl_file" "$mem_file" "$tb_file" > "compile_${name}.log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ]; then
            printf 'Compile failed (status %d), see compile_%s.log\n' "$rc" "$name" >&2
            return "$rc"
        fi
        vvp sim.vvp 2>&1 | tee "run_${name}.log"
        rc="${PIPESTATUS[0]}"
    else
        # shellcheck disable=SC2086
        xrun -64bit -sv -access +rwc -timescale 1ns/1ps \
             -xmlibdirname "$build_dir/xcelium.d" \
             -top tb_tiny_gemm $defines \
             "$rtl_file" "$mem_file" "$tb_file" 2>&1 | tee "run_${name}.log"
        rc="${PIPESTATUS[0]}"
    fi

    return "$rc"
}

verify_positive() {
    if ! grep -Fq "$pass_marker" "run_positive.log"; then
        printf 'FAIL: "%s" not found in run_positive.log\n' "$pass_marker" >&2
        return 1
    fi
    if grep -Fq "$fault_marker" "run_positive.log"; then
        printf 'FAIL: run_positive.log contains the fault-injection marker\n' >&2
        return 1
    fi
    printf 'positive scenario PASS\n'
    return 0
}

verify_fault() {
    if grep -Fq "$pass_marker" "run_fault.log"; then
        printf 'FAIL: the injected corruption was not detected\n' >&2
        return 1
    fi
    if ! grep -Fq "$fault_marker" "run_fault.log"; then
        printf 'FAIL: "%s" not found in run_fault.log\n' "$fault_marker" >&2
        return 1
    fi
    printf 'fault scenario correctly rejected the corrupted result\n'
    return 0
}

scenario="${SCENARIO:-all}"
overall=0

if [ "$scenario" = "all" ] || [ "$scenario" = "positive" ]; then
    run_scenario positive normal
    sim_rc=$?
    printf 'Note: simulator exit status for the positive scenario was %d\n' "$sim_rc"
    verify_positive || overall=1
fi

if [ "$scenario" = "all" ] || [ "$scenario" = "fault" ]; then
    run_scenario fault fault
    sim_rc=$?
    printf 'Note: the fault scenario is expected to fail; simulator exit status was %d\n' "$sim_rc"
    verify_fault || overall=1
fi

printf '\nSimulation outputs: %s\n' "$build_dir"
if [ "$overall" -ne 0 ]; then
    printf 'tiny_gemm regression FAILED\n' >&2
    exit 1
fi
printf 'tiny_gemm regression PASSED\n'
