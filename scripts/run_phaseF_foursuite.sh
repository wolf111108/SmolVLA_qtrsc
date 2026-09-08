#!/usr/bin/env bash
# Phase F — 三组 fp8-pot per_site 量化协议 × LIBERO 四 suite 完整评测。
#
# 实验组：
#   F1  fp8 pot per_site + 0.01 outlier 保护（复用 phaseD per_site 的 scale，
#       量化协议与 D4 per_site 完全一致，直接可比）
#   F2  fp8 pot per_site 无 outlier 保护（新方法 pot_fp8_per_tensor：
#       纯 per-tensor absmax + PoT；注意 outlier_ratio=0 在 outlier 方法下
#       仍会保护 1 个通道，所以必须换方法而不是改 ratio）
#   F3  fp8 pot + weight int4 per_site + 0.01 outlier 保护（Linear w_bit=4 +
#       pot_ao_outlier：a/o 保持 FP8+PoT，w 连续 scale；MatMul 不变）
#
# 每 suite 一次运行（main.py 单 suite），顺序执行 3 组 × 4 suite = 12 次运行。
# F2/F3 首次运行会各做一次校准（各档独立 scale_dir）；F1 复用已有 scale，
# 校准阶段自动跳过（recalibrate=0 时直接复用）。
#
# 幂等：已生成 eval_info.json 的 suite 自动跳过。
#
# 用法：
#   bash scripts/run_phaseF_foursuite.sh
# 后台运行：
#   nohup bash scripts/run_phaseF_foursuite.sh > outputs/phaseF_foursuite.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
EXPS=(
    "phaseF1_fp8pot_site_outlier"
    "phaseF2_fp8pot_site_nooutlier"
    "phaseF3_fp8pot_w4_site_outlier"
)

for exp in "${EXPS[@]}"; do
    for suite in "${SUITES[@]}"; do
        out="outputs/experiments/${exp}_${suite}"
        if [[ -f "$out/eval_info.json" ]]; then
            echo "[skip] ${exp} ${suite} 已有结果: $out/eval_info.json"
            continue
        fi
        echo "=============================================================="
        echo "Phase F: ${exp} | ${suite}"
        echo "=============================================================="
        conda run -n smolvla_eval python main.py \
            --config "configs/experiments/${exp}_${suite}.yaml"
    done
done

echo ""
echo "=============================================================="
echo "Phase F 完成。汇总 SR（每组 4 suite）："
echo "=============================================================="
for exp in "${EXPS[@]}"; do
    echo "--- ${exp} ---"
    for suite in "${SUITES[@]}"; do
        f="outputs/experiments/${exp}_${suite}/eval_info.json"
        if [[ -f "$f" ]]; then
            printf "  %-16s : " "$suite"
            conda run -n smolvla_eval python -c "import json;d=json.load(open('$f'));o=d.get('overall',d);print(f\"pc_success={o['pc_success']}, n={o['n_episodes']}\")" 2>/dev/null \
                || echo "(见 $f)"
        else
            printf "  %-16s : (未完成)\n" "$suite"
        fi
    done
done
