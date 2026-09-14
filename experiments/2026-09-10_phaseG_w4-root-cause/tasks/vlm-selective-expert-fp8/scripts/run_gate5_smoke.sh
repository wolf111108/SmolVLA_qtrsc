#!/usr/bin/env bash
# G6 Gate 5：task0×1 smoke（四组）。
# 复用 Gate 4 已生成的 scale（--skip-calibration），只跑 libero_goal task0 × 1ep。
# 验证 rollout 无 crash、输出非 NaN、routing manifest 与正式 config 一致。
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

# 生成 task0×1 临时 config
conda run -n smolvla_eval python "$TASK/scripts/make_gate5_configs.py"

for name in "${configs[@]}"; do
  tmp_cfg="$TASK/generated/gate5_task0x1/${name}.yaml"

  echo "[GATE5 SMOKE] $name (task0 × 1ep, --skip-calibration)"
  conda run -n smolvla_eval python main.py --config "$tmp_cfg" --skip-calibration
done

echo "G6 Gate 5 smoke finished."
