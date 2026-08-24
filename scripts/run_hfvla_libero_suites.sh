#!/usr/bin/env bash
# HuggingFaceVLA/smolvla_libero (官方 current reference, 32 层/0.5 宽) 在全部四个 LIBERO suite 上的 evaluation
# 用法: bash run_hfvla_libero_suites.sh
# 输出: ~/VLA_tcs2/outputs/hfvla_libero_<suite>/eval_info.json
#
# 与 run_lerobot_libero_suites.sh / run_tiantianx_libero_suites.sh 的区别:
# - POLICY 换成 HuggingFaceVLA/smolvla_libero (32 层 / 0.5 宽, official current contract)
# - 不需要 rename_map: 该 checkpoint config 就是 8D state + image/image2, 与 LIBERO processor 完全一致
#   (README handoff §8.3 与 §12.2 已验证 Task0/Task5 无需 rename 直接跑通)
#
# 已确认事项:
# - input: 8D state + image(image 主相机) + image2(wrist 相机)
# - output: 7D action
# - 该模型是"官方 current LIBERO reference", 非论文 Table-2 checkpoint(论文是 16 层/0.75 宽)
#
# 注意:
# - 每个 suite = 10 tasks × 10 episodes, 预计每个 suite ~1.5-4h(Long 最长 520 步), 四个 suite 共 ~8-12h
# - 需在 smolvla_eval 环境运行, 且 MUJOCO_GL=egl, MUJOCO_EGL_DEVICE_ID=2 已设置

set -euo pipefail

cd "$HOME/VLA_tcs2/lerobot_current"

POLICY="HuggingFaceVLA/smolvla_libero"
SEED=1000
BASE_OUT="$HOME/VLA_tcs2/outputs"
LOGDIR="$BASE_OUT/hfvla_libero_multisuite_logs"
mkdir -p "$LOGDIR"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for SUITE in "${SUITES[@]}"; do
    OUT="$BASE_OUT/hfvla_libero_${SUITE}"
    if [[ -f "$OUT/eval_info.json" ]]; then
        echo "[skip] $SUITE 已有结果: $OUT/eval_info.json"
        continue
    fi
    echo "[run ] $SUITE -> $OUT"
    lerobot-eval \
        --policy.path="$POLICY" \
        --policy.n_action_steps=1 \
        --policy.num_steps=10 \
        --env.type=libero \
        --env.task="$SUITE" \
        --eval.n_episodes=10 \
        --eval.batch_size=1 \
        --env.max_parallel_tasks=1 \
        --seed="$SEED" \
        --output_dir="$OUT" \
        2>&1 | tee "$LOGDIR/${SUITE}.log"
done

echo ""
echo "=== 全部完成, 汇总各 suite eval_info.json 的 pc_success ==="
python3 - <<'PY'
import json
from pathlib import Path

base = Path.home() / "VLA_tcs2/outputs"
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    f = base / f"hfvla_libero_{suite}/eval_info.json"
    if not f.exists():
        print(f"{suite}: 未完成")
        continue
    info = json.loads(f.read_text())
    per_task = info.get("per_task", [])
    if isinstance(per_task, dict):
        per_task = list(per_task.values())
    n_succ = sum(sum(t["metrics"]["successes"]) for t in per_task)
    n_tot = sum(len(t["metrics"]["successes"]) for t in per_task)
    print(f"{suite}: {n_succ}/{n_tot} = {100.0 * n_succ / max(n_tot, 1):.0f}%")
PY

echo ""
echo "=== 生成与论文及 A/C/D 类基线的对比报告 ==="
python3 "$HOME/VLA_tcs2/scripts/compare_hfvla_vs_baselines.py"
