#!/usr/bin/env bash
# G2 weight-granularity 主 runner（分阶段，强制 Gate）
# 用法：
#   bash run_weight_granularity.sh --stage per-channel   # G2-B（需 P0 PASS）
#   bash run_weight_granularity.sh --stage groupwise     # G2-C→D→E（需 Gate 2 允许）
# 幂等：完整 eval_info.json 存在则 skip；不完整目录不视为 done。
set -euo pipefail

# scripts -> <sub> -> tasks -> <exp> -> experiments -> 仓库根 共 5 级
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
TASK_DIR="${REPO_ROOT}/experiments/2026-09-10_phaseG_w4-root-cause/tasks/weight-granularity"
OUT_BASE="outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-granularity"
LOG="${OUT_BASE}/g2_run.log"

STAGE="${1:---help}"
cd "$REPO_ROOT"
export MUJOCO_GL=egl
mkdir -p "${OUT_BASE}"

run_one() {  # $1 = config 名（无 .yaml）
    local name="$1"
    local out="${OUT_BASE}/${name}"
    if [[ -f "${out}/eval_info.json" ]]; then
        echo "[skip] ${name} 已有完整结果"
        return 0
    fi
    echo "=============================================================="
    { echo "[$(date '+%F %T')] G2: ${name} | commit=$(git rev-parse HEAD)"; } | tee -a "$LOG"
    echo "=============================================================="
    conda run -n smolvla_eval python main.py \
        --config "${TASK_DIR}/configs/${name}.yaml" 2>&1 | tee -a "$LOG"
}

case "$STAGE" in
  --stage)
    STAGE_NAME="${2:?缺少 stage 名}"
    case "$STAGE_NAME" in
      per-channel)
        # Gate 前置：P0 preflight 结果必须存在
        if [[ ! -f "${OUT_BASE}/preflight/per_channel/granularity_stats.csv" ]]; then
          echo "STOP: 未找到 G2-P0 per-channel preflight 结果。请先运行 run_granularity_preflight.sh 并核验 PASS。" >&2
          exit 1
        fi
        run_one g2b_per_channel_vlm
        ;;
      groupwise)
        # Gate 前置：G2-B 闭环结果必须存在（Gate 2 决策依据）
        if [[ ! -f "${OUT_BASE}/g2b_per_channel_vlm/eval_info.json" ]]; then
          echo "STOP: G2-B 尚未完成。Gate 2 要求先看到 per-channel 闭环结果。" >&2
          exit 1
        fi
        run_one g2c_group128_vlm
        run_one g2d_group64_vlm
        run_one g2e_group32_vlm
        ;;
      *)
        echo "未知 stage: $STAGE_NAME（可用: per-channel / groupwise）" >&2; exit 1;;
    esac
    ;;
  *)
    echo "用法: $0 --stage <per-channel|groupwise>" >&2; exit 1;;
esac

echo ""
echo "=== G2 stage[${STAGE_NAME:-$2}] 汇总（libero_goal, 100ep）==="
for name in g2b_per_channel_vlm g2c_group128_vlm g2d_group64_vlm g2e_group32_vlm; do
    f="${OUT_BASE}/${name}/eval_info.json"
    if [[ -f "$f" ]]; then
        printf "  %-24s : " "$name"
        conda run -n smolvla_eval python -c "import json;d=json.load(open('$f'));o=d.get('overall',d);print(f\"pc_success={o['pc_success']}, n={o.get('n_episodes','?')}\")" 2>/dev/null \
            || echo "(见 $f)"
    else
        printf "  %-24s : (未完成)\n" "$name"
    fi
done
echo "(G2-A per_tensor anchor 复用 G1-C = 21.0%)"
