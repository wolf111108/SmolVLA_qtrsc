#!/usr/bin/env bash
# Vision Encoder Linear FP8 single-variant runner (VLIN / V2 / V3).
#
# Generalizes run_full_vision_linear.sh: expected Vision site count is derived
# from the config's vision.linear.{mlp,attn_proj} switches, so the same gate
# chain serves:
#   VLIN vlin_full_vision_linear_fp8      (72 sites, 216 new scale files)
#   V2   v2_vision_mlp_fp8                (24 sites,  72 new scale files)
#   V3   v3_vision_attn_proj_fp8          (48 sites, 144 new scale files)
#
# Gate chain (fail-loud, any gate failure aborts):
#   1. py_compile + Vision routing unit tests
#   2. copy canonical G6-A VLM/Expert FP8 scales into a fresh scale dir
#   3. real-model routing/count audit
#   4. calibration-only (ONLY the selected Vision Linear sites recalibrate)
#   5. calibration coverage audit
#   6. LIBERO Goal task0 x 1 smoke with --skip-calibration
# Optional: RUN_GOAL=1 additionally runs Goal 10 tasks x 10 episodes.
#
# Usage:
#   bash experiments/2026-09-15_phaseI_vision-quantization/scripts/run_vision_linear_variant.sh \
#        experiments/2026-09-15_phaseI_vision-quantization/configs/v2_vision_mlp_fp8.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config.yaml>" >&2
  exit 2
fi
CFG="$1"

CONDA_ENV="${CONDA_ENV:-smolvla_eval}"
RUN_GOAL="${RUN_GOAL:-0}"

BASE_SCALE_DIR="${BASE_SCALE_DIR:-scales/2026-09-10_phaseG_w4-root-cause/vlm-selective-expert-fp8/g6a_all_fp8_control}"

if command -v conda >/dev/null 2>&1; then
  RUN=(conda run --no-capture-output -n "$CONDA_ENV")
else
  echo "ERROR: conda not found. Set up the local smolvla environment first." >&2
  exit 2
fi

# Derive per-variant expectations from the config (mirrors model_wrapper).
read -r EXP_SITES EXP_FILES GOAL_CFG <<<"$(
  "${RUN[@]}" python - "$CFG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
q = cfg.get("quantization", {})
vl = q.get("vision", {}).get("linear", {})
sites = 12 * (4 * bool(vl.get("attn_proj", False)) + 2 * bool(vl.get("mlp", False)))
goal = sys.argv[1].replace(".yaml", "_goal.yaml")
print(sites, sites * 3, goal)
PY
)"

TARGET_SCALE_DIR="$(\
  "${RUN[@]}" python - "$CFG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg["quantization"]["scale_dir"])
PY
)"

if [[ ! -f "$GOAL_CFG" ]]; then
  echo "ERROR: goal config not found: $GOAL_CFG" >&2
  exit 2
fi

echo "======================================================================"
echo " Vision Linear FP8 variant runner"
echo "======================================================================"
echo "repo            : $REPO_ROOT"
echo "conda env       : $CONDA_ENV"
echo "config          : $CFG"
echo "goal config     : $GOAL_CFG"
echo "expected sites  : $EXP_SITES (new scale files: $EXP_FILES)"
echo "base scales     : $BASE_SCALE_DIR"
echo "target scales   : $TARGET_SCALE_DIR"
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
if [[ "$VISION_FILES" -ne "$EXP_FILES" ]]; then
  echo "ERROR: expected exactly $EXP_FILES new Vision scale files ($EXP_SITES sites x A/W/O)." >&2
  exit 2
fi

echo
echo "[Gate 6] LIBERO Goal task0 x 1 smoke"
"${RUN[@]}" python main.py \
  --config "$CFG" \
  --skip-calibration

echo
echo "======================================================================"
echo " Vision Linear variant smoke: PASS ($EXP_SITES sites)"
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
  echo "  RUN_GOAL=1 bash $0 $CFG"
fi
