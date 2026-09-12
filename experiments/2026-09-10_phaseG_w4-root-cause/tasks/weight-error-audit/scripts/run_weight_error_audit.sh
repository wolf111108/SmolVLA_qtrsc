#!/usr/bin/env bash
# G0 weight-error-audit — 离线数值审计（无 rollout）
# 步骤：1) 两套 config 各自校准落 scale（--skip-evaluation）
#      2) audit_weight_error.py 计算 224 Linear 的误差指标并输出 5 个 CSV
# 幂等：scale 已存在则 main.py 校准自动跳过（auto→reuse）；CSV 存在则重算覆盖。
# 用法：bash experiments/2026-09-10_phaseG_w4-root-cause/tasks/weight-error-audit/scripts/run_weight_error_audit.sh
set -euo pipefail

# scripts -> <sub> -> tasks -> <exp> -> experiments -> 仓库根 共 5 级
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
TASK_DIR="${REPO_ROOT}/experiments/2026-09-10_phaseG_w4-root-cause/tasks/weight-error-audit"
OUT_DIR="${REPO_ROOT}/outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-error-audit"

cd "$REPO_ROOT"
export MUJOCO_GL=egl

echo "=== G0 step 1/3: calibrate FP8 protocol (scales only) ==="
conda run -n smolvla_eval python main.py \
    --config "${TASK_DIR}/configs/g0_fp8.yaml" --skip-evaluation

echo "=== G0 step 2/3: calibrate W4 protocol (scales only) ==="
conda run -n smolvla_eval python main.py \
    --config "${TASK_DIR}/configs/g0_w4.yaml" --skip-evaluation

echo "=== G0 step 3/3: audit -> CSV ==="
conda run -n smolvla_eval python "${TASK_DIR}/scripts/audit_weight_error.py" \
    --config-fp8 "${TASK_DIR}/configs/g0_fp8.yaml" \
    --config-w4  "${TASK_DIR}/configs/g0_w4.yaml" \
    --out-dir "${OUT_DIR}"

echo ""
echo "=== G0 done. CSV at: ${OUT_DIR} ==="
head -5 "${OUT_DIR}/linear_error_ranked.csv" || true
