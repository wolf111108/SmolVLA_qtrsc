#!/usr/bin/env bash
# G2-P0 — weight-granularity 数值 preflight（无闭环 rollout）
# 比较 VLM per-tensor W4 vs per-output-channel W4 的数值误差，产出 5 个 CSV + PASS/STOP 判定。
# 幂等：CSV 存在则重算覆盖；模型 scale 由脚本内自行计算（离线，无需先校准）。
# 用法：bash experiments/2026-09-10_phaseG_w4-root-cause/tasks/weight-granularity/scripts/run_granularity_preflight.sh
set -euo pipefail

# scripts -> <sub> -> tasks -> <exp> -> experiments -> 仓库根 共 5 级
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
TASK_DIR="${REPO_ROOT}/experiments/2026-09-10_phaseG_w4-root-cause/tasks/weight-granularity"
OUT_DIR="${REPO_ROOT}/outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-granularity/preflight"

cd "$REPO_ROOT"
export MUJOCO_GL=egl

echo "commit: $(git rev-parse HEAD)" | tee "${OUT_DIR}/.preflight_meta" 2>/dev/null || mkdir -p "${OUT_DIR}"

conda run -n smolvla_eval python "${TASK_DIR}/scripts/granularity_preflight.py" \
    --config "${TASK_DIR}/configs/g2a_per_tensor_vlm.yaml" \
    --out-dir "${OUT_DIR}/per_tensor" --granularity per_tensor

conda run -n smolvla_eval python "${TASK_DIR}/scripts/granularity_preflight.py" \
    --config "${TASK_DIR}/configs/g2b_per_channel_vlm.yaml" \
    --out-dir "${OUT_DIR}/per_channel" --granularity per_output_channel

echo ""
echo "=== G2-P0 完成。CSV 位于 ${OUT_DIR}/{per_tensor,per_channel}/ ==="
echo "请按 setup §5.1 PASS 条件核验后，再决定是否启动 G2-B 闭环。"
