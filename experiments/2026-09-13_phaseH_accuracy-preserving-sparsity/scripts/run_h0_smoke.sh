#!/usr/bin/env bash
# Phase H — H0 correctness smoke (experiment_setup.md §12).
# S0 + S1, libero_goal task0 × 1 episode, reuse Phase G scales (no calibration).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export MUJOCO_GL=egl
STAGE=h0_smoke

cd "$SCRIPT_DIR"
python make_task_configs.py --base s0_fp8_all_base.yaml --stage "$STAGE"
python make_task_configs.py --base s1_expert_w4_base.yaml --stage "$STAGE"

cd "$REPO_ROOT"
for cfg in \
  experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/generated/"$STAGE"/s0_fp8_all/task00.yaml \
  experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/generated/"$STAGE"/s1_expert_w4/task00.yaml
do
  echo "=== running $cfg ==="
  python main.py --config "$cfg" --skip-calibration
done

echo "H0 smoke finished."
