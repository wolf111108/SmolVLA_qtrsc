#!/usr/bin/env bash
# Full Vision Encoder Linear FP8 experiment.
#
# This runner is intentionally fail-loud and runs the infrastructure gates
# before the 1-episode smoke evaluation.
#
# What it does:
#   1. static/unit-test gate
#   2. copy canonical G6-A VLM/Expert FP8 scales into a fresh Phase-I scale dir
#   3. real-model routing/count audit (296 Linear / 64 MatMul; Vision=72)
#   4. calibration-only (ONLY the 72 Vision Linear sites recalibrate)
#   5. calibration coverage audit (72/72 sites; 216/216 scale files)
#   6. LIBERO Goal task0 x 1 smoke with --skip-calibration
#
# Optional:
#   RUN_GOAL=1 bash .../run_full_vision_linear.sh
# additionally runs Goal 10 tasks x 10 episodes = 100 episodes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-smolvla_eval}"
RUN_GOAL="${RUN_GOAL:-0}"

CFG="experiments/2026-09-15_phaseI_vision-quantization/configs/vlin_full_vision_linear_fp8.yaml"
GOAL_CFG="experiments/2026-09-15_phaseI_vision-quantization/configs/vlin_full_vision_linear_fp8_goal.yaml"

BASE_SCALE_DIR="${BASE_SCALE_DIR:-scales/2026-09-10_phaseG_w4-root-cause/vlm-selective-expert-fp8/g6a_all_fp8_control}"
TARGET_SCALE_DIR="${TARGET_SCALE_DIR:-scales/2026-09-15_phaseI_vision-quantization/vlin_full_vision_linear_fp8}"

if command -v conda >/dev/null 2>&1; then
  RUN=(conda run --no-capture-output -n "$CONDA_ENV")
else
  echo "ERROR: conda not found. Set up the local smolvla environment first." >&2
  exit 2
fi

echo "======================================================================"
echo " Full Vision 72-Linear FP8 integration"
echo "======================================================================"
echo "repo          : $REPO_ROOT"
echo "conda env     : $CONDA_ENV"
echo "config        : $CFG"
echo "base scales   : $BASE_SCALE_DIR"
echo "target scales : $TARGET_SCALE_DIR"
echo

echo "[Gate 1] py_compile + Vision routing unit tests"
"${RUN[@]}" python -m py_compile \
  src/vla_tcs2/model_wrapper.py \
  src/vla_tcs2/quant_linear.py \
  src/vla_tcs2/calibration.py \
  experiments/2026-09-15_phaseI_vision-quantization/scripts/audit_full_vision_linear_routing.py \
  experiments/2026-09-15_phaseI_vision-quantization/scripts/audit_full_vision_linear_calibration.py

"${RUN[@]}" python -m pytest tests/test_vision_quant_routing.py -q

echo
echo "[Gate 2] prepare isolated scale directory"
if [[ ! -d "$BASE_SCALE_DIR" ]]; then
  echo "ERROR: canonical G6-A scale directory not found:" >&2
  echo "  $BASE_SCALE_DIR" >&2
  echo "Set BASE_SCALE_DIR to the valid canonical FP8-all scale directory." >&2
  exit 2
fi

rm -rf "$TARGET_SCALE_DIR"
mkdir -p "$TARGET_SCALE_DIR"
cp -a "$BASE_SCALE_DIR"/. "$TARGET_SCALE_DIR"/

BASE_COUNT="$(find "$BASE_SCALE_DIR" -maxdepth 1 -type f -name '*.p' | wc -l | tr -d ' ')"
COPY_COUNT="$(find "$TARGET_SCALE_DIR" -maxdepth 1 -type f -name '*.p' | wc -l | tr -d ' ')"
echo "canonical scale files : $BASE_COUNT"
echo "copied scale files    : $COPY_COUNT"
if [[ "$BASE_COUNT" -ne "$COPY_COUNT" ]]; then
  echo "ERROR: copied scale file count differs from canonical source." >&2
  exit 2
fi

echo
echo "[Gate 3] real-model routing/count audit"
"${RUN[@]}" python \
  experiments/2026-09-15_phaseI_vision-quantization/scripts/audit_full_vision_linear_routing.py \
  --config "$CFG"

echo
echo "[Gate 4] calibration-only"
"${RUN[@]}" python main.py \
  --config "$CFG" \
  --skip-evaluation

echo
echo "[Gate 5] Vision calibration coverage"
"${RUN[@]}" python \
  experiments/2026-09-15_phaseI_vision-quantization/scripts/audit_full_vision_linear_calibration.py \
  --config "$CFG"

FINAL_COUNT="$(find "$TARGET_SCALE_DIR" -maxdepth 1 -type f -name '*.p' | wc -l | tr -d ' ')"
VISION_FILES="$((FINAL_COUNT - COPY_COUNT))"
echo "scale files after calibration : $FINAL_COUNT"
echo "new files vs canonical copy   : $VISION_FILES"
if [[ "$VISION_FILES" -ne 216 ]]; then
  echo "ERROR: expected exactly 216 new Vision scale files (72 sites x A/W/O)." >&2
  exit 2
fi

echo
echo "[Gate 6] LIBERO Goal task0 x 1 smoke"
"${RUN[@]}" python main.py \
  --config "$CFG" \
  --skip-calibration

echo
echo "======================================================================"
echo " Full Vision Linear smoke: PASS"
echo "======================================================================"

if [[ "$RUN_GOAL" == "1" ]]; then
  echo
  echo "[Optional Gate 7] LIBERO Goal x100 formal evaluation"
  "${RUN[@]}" python main.py \
    --config "$GOAL_CFG" \
    --skip-calibration
else
  echo
  echo "Formal Goal x100 was NOT run."
  echo "To run it after reviewing the smoke:"
  echo "  RUN_GOAL=1 bash $SCRIPT_DIR/run_full_vision_linear.sh"
fi
