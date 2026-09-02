#!/usr/bin/env bash
# int8 + outlier 保护 量化测试：校准 + 量化推理 eval
#
# 用法：
#   bash scripts/run_int8_outlier_quant_test.sh              # 校准 + eval
#   bash scripts/run_int8_outlier_quant_test.sh --reuse      # 跳过校准，复用已有 scales
#
# 说明：activation/weight/output 统一 int8，启用 outlier_ratio=0.01（见 config）。

set -euo pipefail

CONFIG=configs/experiments/smolvla_int8_outlier_quant_test.yaml
OUT_DIR=outputs/experiments/smolvla_int8_outlier_quant_test
FP_BASELINE=93.8   # verify_libero_object ep10 FP baseline（同口径）
SKIP_CALIB_FLAG=""

if [[ "${1:-}" == "--reuse" ]]; then
    SKIP_CALIB_FLAG="--skip-calibration"
    echo ">>> Reusing existing scales under $(grep 'scale_dir' $CONFIG | awk '{print $2}')"
fi

export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

echo "==================== int8 + outlier quantization test ===================="
echo "config : $CONFIG"
echo "output : $OUT_DIR"
echo "=========================================================================="

python main.py \
    --config "$CONFIG" \
    $SKIP_CALIB_FLAG

echo
echo "==================== RESULTS ===================="

python - "$OUT_DIR/result.json" "$FP_BASELINE" <<'EOF'
import json, sys

path, baseline = sys.argv[1], float(sys.argv[2])
result = json.load(open(path))

def collect(d):
    ok = n = 0
    for t in d.get("per_task", []):
        s = t["metrics"]["successes"]
        ok += sum(s); n += len(s)
    return ok, n

ok, n = collect(result)
if n == 0:
    print("no success data found"); sys.exit(1)

sr = 100 * ok / n
print(f"int8+outlier    : {ok}/{n} = {sr:.1f}%")
print(f"FP baseline     : {baseline:.1f}%  (libero_object ep10, 同口径)")
print(f"delta           : {sr - baseline:+.1f} pp")
print()
print("参考（同口径 ep10）：")
print("  FP     = 93.8%")
print("  int16  = 89.0%   (-4.8pp)")
print("  int12  = 7.0%    (-86.8pp)")
print("  int12+outlier = 95.0% (+1.2pp)")
print("  int8+outlier  = 本次结果")
EOF
