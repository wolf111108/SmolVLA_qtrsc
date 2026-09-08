#!/usr/bin/env bash
# Phase D-w4 — 一键跑完 4 档 scale granularity ablation（Linear w_bit=4 / int4 版）。
#
# 与 run_phaseD_ablation.sh 的差异：
#   - 配置为 phaseD_w4_ablation_*.yaml（Linear 权重 int4 + method: pot_ao_outlier，
#     a/o 保持 FP8+PoT；MatMul 显式 pot_fp8_outlier/e4m3）
#   - 独立 scale_dir / output_dir（phaseD_w4_*），与 FP8 基线产物隔离
#
# 每个粒度独立 scale_dir + output_dir，顺序执行 calibration + evaluation。
# 全部完成后，从各 output_dir/result.json 汇总 SR。
#
# 用法：
#   bash scripts/run_phaseD_w4_ablation.sh
# 后台运行：
#   nohup bash scripts/run_phaseD_w4_ablation.sh > outputs/phaseD_w4_ablation.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."

GRANULARITIES=(per_site per_layer per_component global)

for g in "${GRANULARITIES[@]}"; do
    echo "=============================================================="
    echo "Phase D-w4 ablation (Linear w_bit=4): ${g}"
    echo "=============================================================="
    conda run -n smolvla_eval python main.py \
        --config "configs/experiments/phaseD_w4_ablation_${g}.yaml"
done

echo ""
echo "=============================================================="
echo "Phase D-w4 Ablation 完成。汇总 SR："
echo "=============================================================="
for g in "${GRANULARITIES[@]}"; do
    f="outputs/experiments/phaseD_w4_ablation_${g}/result.json"
    if [[ -f "$f" ]]; then
        printf "  %-16s : " "$g"
        conda run -n smolvla_eval python -c "import json;d=json.load(open('$f'));print(d)" 2>/dev/null \
            || echo "(见 $f)"
    else
        printf "  %-16s : (未生成 result.json)\n" "$g"
    fi
done
