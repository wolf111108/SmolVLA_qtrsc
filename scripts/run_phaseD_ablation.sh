#!/usr/bin/env bash
# Phase D — 一键跑完 4 档 scale granularity ablation。
#
# 每个粒度独立 scale_dir + output_dir，顺序执行 calibration + evaluation。
# 全部完成后，从各 output_dir/result.json 汇总 SR。
#
# 用法：
#   bash scripts/run_phaseD_ablation.sh
set -euo pipefail

cd "$(dirname "$0")/.."

GRANULARITIES=(per_site per_layer per_component global)

for g in "${GRANULARITIES[@]}"; do
    echo "=============================================================="
    echo "Phase D ablation: ${g}"
    echo "=============================================================="
    conda run -n smolvla_eval python main.py \
        --config "configs/experiments/phaseD_ablation_${g}.yaml"
done

echo ""
echo "=============================================================="
echo "Ablation 完成。汇总 SR："
echo "=============================================================="
for g in "${GRANULARITIES[@]}"; do
    f="outputs/experiments/phaseD_ablation_${g}/result.json"
    if [[ -f "$f" ]]; then
        printf "  %-16s : " "$g"
        conda run -n smolvla_eval python -c "import json;d=json.load(open('$f'));print(d)" 2>/dev/null \
            || echo "(见 $f)"
    else
        printf "  %-16s : (未生成 result.json)\n" "$g"
    fi
done
