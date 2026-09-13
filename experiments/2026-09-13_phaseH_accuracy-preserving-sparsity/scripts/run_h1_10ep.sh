#!/usr/bin/env bash
# Phase H — H1 10-episode pilot (experiment_setup.md §5.2).
# 10 tasks × 1 episode per config, for S0 and S1. Reuses Phase G scales.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export MUJOCO_GL=egl
STAGE=h1_10ep

cd "$SCRIPT_DIR"
python make_task_configs.py --base s0_fp8_all_base.yaml --stage "$STAGE"
python make_task_configs.py --base s1_expert_w4_base.yaml --stage "$STAGE"

cd "$REPO_ROOT"
for cfg in \
  experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/generated/"$STAGE"/s0_fp8_all/task*.yaml \
  experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/generated/"$STAGE"/s1_expert_w4/task*.yaml
do
  echo "=== running $cfg ==="
  python main.py --config "$cfg" --skip-calibration
done

echo "H1 10ep finished."
