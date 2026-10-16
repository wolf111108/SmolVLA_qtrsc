#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_DIR="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
cd "$REPOSITORY_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_OUTPUT="${RUN_OUTPUT:-outputs/2026-10-16_phaseL_gemm-single-inference-timeline/int8_w4}"
if [[ -e "$RUN_OUTPUT" ]]; then
  echo "Output already exists; choose a fresh RUN_OUTPUT: $RUN_OUTPUT" >&2
  exit 1
fi
mkdir -p "$RUN_OUTPUT"
"$PYTHON_BIN" "$EXPERIMENT_DIR/scripts/run_one.py" --output-dir "$RUN_OUTPUT" "$@" 2>&1 | tee "$RUN_OUTPUT/run.log"
"$PYTHON_BIN" "$EXPERIMENT_DIR/scripts/export_timeline.py" --run-dir "$RUN_OUTPUT/hardware" 2>&1 | tee "$RUN_OUTPUT/export.log"
"$PYTHON_BIN" "$EXPERIMENT_DIR/scripts/plot_timeline.py" --timeline-dir "$RUN_OUTPUT/timeline" 2>&1 | tee "$RUN_OUTPUT/plot.log"
