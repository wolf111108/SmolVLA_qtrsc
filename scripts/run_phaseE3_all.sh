#!/usr/bin/env bash
# Phase E3 — component-wise (vlm vs expert) noise tolerance, 5x5 alpha grid.
#
# Each experiment:
#   - reuses the calibrated scales in scales/phaseE_sensitivity
#   - injects gaussian_rms_output noise into ALL vlm.* sites with alpha_vlm
#     and ALL expert.* sites with alpha_expert (288 sites total)
#   - evaluates 10 episodes on libero_object
#
# Logs:   outputs/phaseE3_<name>.log
# Result: outputs/experiments/<name>/result.json
#
# Usage:
#   bash scripts/run_phaseE3_all.sh [names...]   (default: all, in order)
# Resume: experiments with an existing result.json are skipped.

set -u
cd "$(dirname "$0")/.."

PY=/home/lfwang/.conda/envs/smolvla_eval/bin/python

# ordered: vlm alpha ascending outer, expert alpha ascending inner
ALL=()
for av in 01 05 10 20 50; do
  for ae in 01 05 10 20 50; do
    ALL+=("phaseE3_v${av}_e${ae}")
  done
done

NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=("${ALL[@]}")

echo "=== Phase E3 5x5 sweep: ${#NAMES[@]} experiments ==="
mkdir -p outputs/experiments

FAILED=()
for name in "${NAMES[@]}"; do
  cfg="configs/experiments/${name}.yaml"
  log="outputs/${name}.log"
  res="outputs/experiments/${name}/result.json"

  if [ -f "$res" ]; then
    echo "[skip] $name (result.json exists)"
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

# 5x5 grid view (vlm alpha rows x expert alpha cols)
echo
echo "=== 5x5 grid (rows=alpha_vlm, cols=alpha_expert, values=SR%) ==="
printf "       "
for ae in 01 05 10 20 50; do printf "e%-5s" "$ae"; done
echo
for av in 01 05 10 20 50; do
  printf "v%-5s " "$av"
  for ae in 01 05 10 20 50; do
    res="outputs/experiments/phaseE3_v${av}_e${ae}/result.json"
    if [ -f "$res" ]; then
      sr=$("$PY" -c "import json;print(json.load(open('$res'))['overall']['pc_success'])" 2>/dev/null)
      printf "%-6s" "$sr"
    else
      printf "%-6s" "--"
    fi
  done
  echo
done

[ ${#FAILED[@]} -gt 0 ] && echo "failed: ${FAILED[*]}"
exit 0
