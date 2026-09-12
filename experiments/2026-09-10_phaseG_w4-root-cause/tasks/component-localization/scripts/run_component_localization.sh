#!/usr/bin/env bash
# G1 component-localization — 定位 F3 W4 退化来源（VLM vs Expert）| libero_goal × 100ep
# G1-A: 全 Linear FP8（F1 anchor，独立校准）
# G1-B: 全 Linear W4（F3 anchor）
# G1-C: 仅 vlm.* Linear W4（expert 为 raw FP）
# G1-D: 仅 expert.* Linear W4（vlm 为 raw FP）
# 幂等：已有 eval_info.json 的 config 自动跳过。
# 用法：bash experiments/2026-09-10_phaseG_w4-root-cause/tasks/component-localization/scripts/run_component_localization.sh
set -euo pipefail

# scripts -> <sub> -> tasks -> <exp> -> experiments -> 仓库根 共 5 级
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
TASK_DIR="${REPO_ROOT}/experiments/2026-09-10_phaseG_w4-root-cause/tasks/component-localization"

cd "$REPO_ROOT"
export MUJOCO_GL=egl

EXPS=(
    "g1a_fp8_all"
    "g1b_w4_all"
    "g1c_w4_vlm_only"
    "g1d_w4_expert_only"
)

for exp in "${EXPS[@]}"; do
    out="outputs/2026-09-10_phaseG_w4-root-cause/tasks/component-localization/${exp}"
    if [[ -f "$out/eval_info.json" ]]; then
        echo "[skip] ${exp} 已有结果: $out/eval_info.json"
        continue
    fi
    echo "=============================================================="
    echo "G1: ${exp}"
    echo "=============================================================="
    conda run -n smolvla_eval python main.py \
        --config "${TASK_DIR}/configs/${exp}.yaml"
done

echo ""
echo "=== G1 汇总（libero_goal, 100ep）==="
for exp in "${EXPS[@]}"; do
    f="outputs/2026-09-10_phaseG_w4-root-cause/tasks/component-localization/${exp}/eval_info.json"
    if [[ -f "$f" ]]; then
        printf "  %-20s : " "$exp"
        conda run -n smolvla_eval python -c "import json;d=json.load(open('$f'));o=d.get('overall',d);print(f\"pc_success={o['pc_success']}, n={o.get('n_episodes','?')}\")" 2>/dev/null \
            || echo "(见 $f)"
    else
        printf "  %-20s : (未完成)\n" "$exp"
    fi
done
