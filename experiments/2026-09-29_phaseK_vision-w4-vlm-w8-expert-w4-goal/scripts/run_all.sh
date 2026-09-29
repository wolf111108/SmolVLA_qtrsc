#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
EXP=experiments/2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal
OUT=outputs/2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal
SCALE=scales/2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal
if [[ -e "$OUT" || -e "$SCALE" ]]; then
  echo 'Archive prior output/scales before a fresh run; see setup for per-stage continuation.' >&2
  exit 1
fi
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
git rev-parse HEAD > "$OUT/commit.txt"
git diff HEAD > "$OUT/worktree.patch"
git status --short > "$OUT/git_status.txt"
python -m pip freeze > "$OUT/packages.txt"
cp -r "$EXP/configs" "$OUT/source_configs"
python "$EXP/scripts/run_arm.py" prepare
python "$EXP/scripts/run_arm.py" calibrate
python "$EXP/scripts/run_arm.py" smoke
python "$EXP/scripts/run_arm.py" quant
python "$EXP/scripts/summarize.py"
