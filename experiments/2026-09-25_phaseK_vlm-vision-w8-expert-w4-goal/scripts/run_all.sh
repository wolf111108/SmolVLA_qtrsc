#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
EXP=experiments/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal
OUT=outputs/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal
SCALE=scales/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal
if [[ -e "$OUT" || -e "$SCALE" ]]; then
  echo 'Archive the existing experiment output/scales before running again.' >&2
  exit 1
fi
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD > "$OUT/commit.txt"
git diff > "$OUT/worktree.patch"
python -m pip freeze > "$OUT/packages.txt"
cp -r "$EXP/configs" "$OUT/source_configs"
python "$EXP/scripts/run_arm.py" prepare
python "$EXP/scripts/run_arm.py" calibrate
python "$EXP/scripts/run_arm.py" smoke
python "$EXP/scripts/run_arm.py" baseline
python "$EXP/scripts/run_arm.py" quant
python "$EXP/scripts/summarize.py"
