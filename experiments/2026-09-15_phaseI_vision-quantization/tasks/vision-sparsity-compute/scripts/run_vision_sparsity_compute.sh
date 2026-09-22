#!/usr/bin/env bash
# Phase I task: Vision/VLM/Expert sparsity + compute characterization.
# Default variant is VSC-1, which reruns the 1-episode workload trace after
# the exact MatMul physical-MAC accounting fix.
#
# Uses the already-calibrated full VLIN scales and MUST NOT recalibrate.
# Primary scope:
#   - runtime native element sparsity
#   - runtime native bit sparsity
#   - static quantized-weight element/bit sparsity
#   - dense MAC/FLOP accounting per sample_actions()
#
# No accuracy claim is made from this 1-episode workload-characterization run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_ROOT="$(cd "$TASK_ROOT/../.." && pwd)"
REPO_ROOT="$(cd "$EXP_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-smolvla_eval}"
VARIANT="${VSC_VARIANT:-vsc1_vlin_fp8_task0_1ep_workloadfix}"
CFG="$TASK_ROOT/configs/${VARIANT}.yaml"
OUT="outputs/2026-09-15_phaseI_vision-quantization/tasks/vision-sparsity-compute/${VARIANT}"
SCALE_DIR="scales/2026-09-15_phaseI_vision-quantization/vlin_full_vision_linear_fp8"
PARENT_AUDIT="experiments/2026-09-15_phaseI_vision-quantization/scripts/audit_full_vision_linear_calibration.py"
SUMMARIZER="$TASK_ROOT/scripts/summarize_vision_sparsity_compute.py"

# robosuite/EGL convention used by the prior sparsity experiments: do not
# isolate CUDA_VISIBLE_DEVICES because conda's MUJOCO_EGL_DEVICE_ID may then
# fail the robosuite assertion.
unset CUDA_VISIBLE_DEVICES || true

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found." >&2
  exit 2
fi
RUN=(conda run --no-capture-output -n "$CONDA_ENV")

echo "======================================================================"
echo " Vision sparsity + compute characterization"
echo "======================================================================"
echo "repo       : $REPO_ROOT"
echo "variant    : $VARIANT"
echo "config     : $CFG"
echo "output     : $OUT"
echo "scale_dir  : $SCALE_DIR"
echo

echo "[Gate 0] static checks"
"${RUN[@]}" python -m py_compile "$SUMMARIZER"
"${RUN[@]}" python -m pytest tests/test_sparsity_accounting.py -q

echo
echo "[Gate 1] VLIN scale coverage (must already exist; no recalibration)"
if [[ ! -d "$SCALE_DIR" ]]; then
  echo "ERROR: VLIN scale directory missing: $SCALE_DIR" >&2
  echo "Run the completed VLIN calibration workflow first." >&2
  exit 2
fi
"${RUN[@]}" python "$PARENT_AUDIT" --config "$CFG"

echo
echo "[Gate 2] workload characterization: task0 x 1 episode, --skip-calibration"
mkdir -p "$OUT"
"${RUN[@]}" python main.py --config "$CFG" --skip-calibration   2>&1 | tee "$OUT/run.log"

echo
echo "[Gate 3] primary output presence"
SP="$OUT/sparsity"
required=(
  "$SP/module_sparsity.csv"
  "$SP/weight_sparsity_static.csv"
  "$SP/quantization_manifest.csv"
  "$SP/workload.csv"
)
for f in "${required[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: missing/empty primary output: $f" >&2
    exit 2
  fi
  lines="$(wc -l < "$f")"
  if (( lines <= 1 )); then
    echo "ERROR: header-only primary output: $f" >&2
    exit 2
  fi
  echo "  OK: $f  ($lines lines)"
done

echo
echo "[Gate 4] component sparsity + per-sample_actions compute summary"
"${RUN[@]}" python "$SUMMARIZER" --output "$OUT"

echo
echo "======================================================================"
echo " PASS — Vision sparsity + compute characterization complete"
echo "======================================================================"
echo "Primary summaries:"
echo "  $OUT/sparsity_compute_summary.csv"
echo "  $OUT/sparsity_by_operator.csv"
echo "  $OUT/compute_summary.csv"
