#!/usr/bin/env bash
# Run ONLY the formal LIBERO Goal x100 evaluation for the full Vision 72-Linear
# FP8 experiment. Assumes run_full_vision_linear.sh has already completed
# calibration and coverage gates successfully.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-smolvla_eval}"
CFG="experiments/2026-09-15_phaseI_vision-quantization/configs/vlin_full_vision_linear_fp8_goal.yaml"

if command -v conda >/dev/null 2>&1; then
  RUN=(conda run --no-capture-output -n "$CONDA_ENV")
else
  echo "ERROR: conda not found." >&2
  exit 2
fi

echo "[preflight] Vision calibration coverage"
"${RUN[@]}" python   experiments/2026-09-15_phaseI_vision-quantization/scripts/audit_full_vision_linear_calibration.py   --config "$CFG"

echo "[formal] LIBERO Goal: 10 tasks x 10 episodes = 100 episodes"
"${RUN[@]}" python main.py   --config "$CFG"   --skip-calibration
