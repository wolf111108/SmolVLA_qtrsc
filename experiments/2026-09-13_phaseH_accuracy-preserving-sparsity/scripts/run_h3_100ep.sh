#!/usr/bin/env bash
# Phase H — H3 Goal formal run (experiment_setup.md §5.4).
# 10 tasks × 10 episodes per config, for S0 and S1. Reuses Phase G scales.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export MUJOCO_GL=egl
STAGE=h3_100ep

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

echo "H3 100ep finished."
