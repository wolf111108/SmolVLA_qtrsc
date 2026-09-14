#!/usr/bin/env bash
# G6 Goal ×100 正式运行（4 组串行）。
# 每组独立 calibration + 100ep evaluation。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
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
  out="$REPO_ROOT/outputs/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/${name}"

  if [[ -f "$out/result.json" ]]; then
    echo "[SKIP] $name already complete"
    continue
  fi

  echo "[RUN] $name"
  conda run -n smolvla_eval python main.py --config "$cfg"
done

echo "G6 Goal run finished."
