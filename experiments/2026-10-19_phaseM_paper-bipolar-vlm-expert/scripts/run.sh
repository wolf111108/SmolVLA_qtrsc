#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
EXPERIMENT="$ROOT/experiments/2026-10-19_phaseM_paper-bipolar-vlm-expert"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_OUTPUT="${RUN_OUTPUT:-$ROOT/outputs/2026-10-19_phaseM_paper-bipolar-vlm-expert/int8_w4}"
if [[ -e "$RUN_OUTPUT" ]]; then
  echo "Choose a fresh RUN_OUTPUT: $RUN_OUTPUT" >&2
  exit 1
fi
CAPTURE_LOG="$(mktemp)"
trap 'if [[ -d "$RUN_OUTPUT" ]]; then cp "$CAPTURE_LOG" "$RUN_OUTPUT/run.log"; fi; rm -f "$CAPTURE_LOG"' EXIT
"$PYTHON_BIN" "$EXPERIMENT/scripts/run_one.py" "$@" --output-dir "$RUN_OUTPUT" 2>&1 | tee "$CAPTURE_LOG"
"$PYTHON_BIN" "$EXPERIMENT/scripts/replay.py" --run-dir "$RUN_OUTPUT" --suite 2>&1 | tee "$RUN_OUTPUT/replay.log"
"$PYTHON_BIN" "$EXPERIMENT/scripts/plot.py" --run-dir "$RUN_OUTPUT" 2>&1 | tee "$RUN_OUTPUT/plot.log"
