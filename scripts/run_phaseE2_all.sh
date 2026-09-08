#!/usr/bin/env bash
# Phase E2 — run all closed-loop sensitivity experiments sequentially.
#
# Each experiment:
#   - reuses the calibrated scales in scales/phaseE_sensitivity (864 files)
#   - injects noise into exactly ONE site (or runs raw baseline)
#   - evaluates 10 episodes on libero_object
#
# Logs:   outputs/phaseE2_<name>.log
# Result: outputs/experiments/<name>/result.json
#
# Usage:
#   bash scripts/run_phaseE2_all.sh [names...]   (default: all, in order)

set -u
cd "$(dirname "$0")/.."

PY=/home/lfwang/.conda/envs/smolvla_eval/bin/python

ALL=(
  phaseE2_baseline_raw
  phaseE2_vlm3_down_a001
  phaseE2_vlm3_down_a003
  phaseE2_vlm3_down_a010
  phaseE2_vlm0_down_a001
  phaseE2_vlm0_down_a003
  phaseE2_vlm0_down_a010
  phaseE2_exp1_up_a003
  phaseE2_exp1_up_a010
  phaseE2_exp7qk_a003
  phaseE2_exp7qk_a030
  phaseE2_exp7qk_qr
  phaseE2_vlm3qk_a003
  phaseE2_vlm3qk_a010
)

NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=("${ALL[@]}")

echo "=== Phase E2 full sweep: ${#NAMES[@]} experiments ==="
mkdir -p outputs/experiments

FAILED=()
SKIPPED=()
for name in "${NAMES[@]}"; do
  cfg="configs/experiments/${name}.yaml"
  log="outputs/${name}.log"
  res="outputs/experiments/${name}/result.json"

  if [ -f "$res" ]; then
    echo "[skip] $name (result.json exists)"
    SKIPPED+=("$name")
    continue
  fi
  if [ ! -f "$cfg" ]; then
    echo "[warn] $name: config not found, skipping"
    FAILED+=("$name")
    continue
  fi

  echo
  echo "==================================================================="
  echo "[$(date '+%F %T')] RUN $name"
  echo "==================================================================="
  if "$PY" main.py --config "$cfg" --skip-calibration > "$log" 2>&1; then
    echo "[$(date '+%F %T')] DONE $name -> $res"
  else
    echo "[$(date '+%F %T')] FAIL $name (see $log)"
    FAILED+=("$name")
  fi
done

echo
echo "=== summary ==="
for name in "${NAMES[@]}"; do
  res="outputs/experiments/${name}/result.json"
  if [ -f "$res" ]; then
    sr=$("$PY" -c "import json;print(json.load(open('$res'))['overall']['pc_success'])" 2>/dev/null)
    echo "  OK   $name  SR=${sr}%"
  else
    echo "  MISS $name"
  fi
done
[ ${#FAILED[@]} -gt 0 ] && echo "failed: ${FAILED[*]}"
exit 0
