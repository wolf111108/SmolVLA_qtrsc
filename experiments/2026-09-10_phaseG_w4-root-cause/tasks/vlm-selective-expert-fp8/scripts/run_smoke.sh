#!/usr/bin/env bash
# G6 calibration-only smoke（Gate 4）。
# 四组各做一次 calibration（--skip-evaluation），验证 scale 生成无 missing/NaN。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TASK="$REPO_ROOT/experiments/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8"

cd "$REPO_ROOT"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=2

configs=(
  g6a_all_fp8_control
  g6b_vlm_attn_w4_expert_fp8
  g6c_vlm_mlp_w4_expert_fp8
  g6d_vlm_all_w4_expert_fp8
)

for name in "${configs[@]}"; do
  cfg="$TASK/configs/${name}.yaml"
  echo "[CALIBRATION SMOKE] $name"
  conda run -n smolvla_eval python main.py --config "$cfg" --skip-evaluation
done

echo "G6 calibration smoke finished."
