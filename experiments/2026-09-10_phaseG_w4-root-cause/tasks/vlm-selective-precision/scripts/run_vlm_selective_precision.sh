#!/usr/bin/env bash
# G5 vlm-selective-precision — VLM Attention vs MLP 的 W4 敏感度定位
# 3 个新 config（G5-D 复用 G1-C=21% 不重跑）。
# 幂等：完整 eval_info.json 存在则 skip；不完整目录不视为 done。
# 开跑前先做 routing smoke（见 g5_routing_smoke.py），通过后再跑 100ep。
# 用法：bash experiments/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-precision/scripts/run_vlm_selective_precision.sh
set -euo pipefail

# scripts -> <sub> -> tasks -> <exp> -> experiments -> 仓库根 共 5 级
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
TASK_DIR="${REPO_ROOT}/experiments/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-precision"
OUT_BASE="outputs/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-precision"

cd "$REPO_ROOT"
export MUJOCO_GL=egl
mkdir -p "${OUT_BASE}"

declare -A EXPS=(
  ["g5a_vlm_fp8_control"]="all_fp8"
  ["g5b_vlm_attn_w4_mlp_fp8"]="attn_w4"
  ["g5c_vlm_attn_fp8_mlp_w4"]="attn_fp8_mlp_w4"
)

# 1) routing smoke（全部先跑，快速失败）
for exp in "${!EXPS[@]}"; do
  echo "=== routing smoke: ${exp} ==="
  conda run -n smolvla_eval python "${TASK_DIR}/scripts/g5_routing_smoke.py" \
      "${TASK_DIR}/configs/${exp}.yaml" "${EXPS[$exp]}"
done

# 2) rollout
for exp in "${!EXPS[@]}"; do
  out="${OUT_BASE}/${exp}"
  if [[ -f "${out}/eval_info.json" ]]; then
    echo "[skip] ${exp} 已有完整结果"
    continue
  fi
  echo "=============================================================="
  echo "G5: ${exp} | commit=$(git rev-parse HEAD)"
  echo "=============================================================="
  conda run -n smolvla_eval python main.py \
      --config "${TASK_DIR}/configs/${exp}.yaml"
done

echo ""
echo "=== G5 汇总（libero_goal, 100ep）==="
for exp in g5a_vlm_fp8_control g5b_vlm_attn_w4_mlp_fp8 g5c_vlm_attn_fp8_mlp_w4; do
  f="${OUT_BASE}/${exp}/eval_info.json"
  if [[ -f "$f" ]]; then
    printf "  %-28s : " "$exp"
    conda run -n smolvla_eval python -c "import json;d=json.load(open('$f'));o=d.get('overall',d);print(f\"pc_success={o['pc_success']}, n={o.get('n_episodes','?')}\")" 2>/dev/null \
        || echo "(见 $f)"
  else
    printf "  %-28s : (未完成)\n" "$exp"
  fi
done
echo "  G5-D VLM all W4 (复用 G1-C)       : 21.0%"
