#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
EXP=experiments/2026-09-23_phaseI_vision-attention-smoke
OUT=outputs/2026-09-23_phaseI_vision-attention-smoke
CFG="$EXP/configs/vision_qkpv_fp8.yaml"
# Preserve previous evidence; rerun only after archiving these experiment outputs/scales.
if [[ -e "$OUT/run.log" || -d scales/2026-09-23_phaseI_vision-attention-smoke/vision_qkpv_fp8 ]]; then
  echo 'Existing run/scales found; archive them before rerunning.' >&2
  exit 1
fi
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD > "$OUT/commit.txt"
git diff > "$OUT/worktree.patch"
python -m pip freeze > "$OUT/packages.txt"
python -m pytest tests/test_vision_attention.py -q
python "$EXP/scripts/preflight.py" "$CFG"
python main.py --config "$CFG"
python "$EXP/scripts/check_outputs.py" "$CFG"
