#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
EXP=experiments/2026-11-09_phaseL_int8-goal
OUT=outputs/2026-11-09_phaseL_int8-goal
SCALE=scales/2026-11-09_phaseL_int8-goal
if [[ -e "$OUT" || -e "$SCALE" ]]; then
  echo 'Archive existing outputs/scales before starting a fresh run; use run_arm.py to resume stages.' >&2
  exit 1
fi
PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$EXP/scripts/run_arm.py" preflight
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1
git rev-parse HEAD > "$OUT/commit.txt"
git diff > "$OUT/worktree.patch"
"$PYTHON" -m pip freeze > "$OUT/packages.txt"
cp -r "$EXP/configs" "$OUT/source_configs"
"$PYTHON" "$EXP/scripts/run_arm.py" prepare
for arm in a8w8 a8w4_expert; do
  "$PYTHON" "$EXP/scripts/run_arm.py" calibrate --arm "$arm"
done
for arm in baseline a8w8 a8w4_expert; do
  "$PYTHON" "$EXP/scripts/run_arm.py" smoke --arm "$arm"
done
for arm in baseline a8w8 a8w4_expert; do
  "$PYTHON" "$EXP/scripts/run_arm.py" eval --arm "$arm"
done
"$PYTHON" "$EXP/scripts/summarize.py"
