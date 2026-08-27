#!/usr/bin/env bash
# PHASE 6 后续 — D 类 k1000dai/smolvla_libero_finetune 四 suite 在 mujoco 3.3.2 环境下的完整评测
# 目的：MuJoCo 版本降级后重测 D 类（recipe 最贴论文, batch64），对照 3.8.1 下的 S64/O82/G70/L46, 平均65.5%
#
# D 类 (PROVENANCE.md §4): k1000dai/smolvla_libero_finetune
#   16 层 / 0.75 宽 / 100k steps / batch64 / expert-only / from smolvla_base
#   相机 key 为 image/wrist_image, 需 rename_map（由 preflight 自动推导）
#
# 用法: 在 smolvla_eval 环境执行: bash run_phase6_mj332_D_foursuite.sh
# 输出: ~/VLA_tcs2/outputs/table2_repro_audit/06_simulator/mj332_D_<suite>/eval_info.json
#
# 注意:
# - 必须先 conda activate smolvla_eval (mujoco 3.3.2)
# - D 类相机 key 为 image/wrist_image（与 A/C 的 image/image2 不同），
#   rename_map 由 scripts/preflight_policy.py 从 config 自动推导
# - 四 suite = 400 episodes, Long(520步)最慢, 预计 8-12h
# - 幂等: 已有 eval_info.json 的 suite 自动跳过

set -euo pipefail

cd "$HOME/VLA_tcs2/lerobot_current"

POLICY="k1000dai/smolvla_libero_finetune"
SEED=1000
BASE_OUT="$HOME/VLA_tcs2/outputs/table2_repro_audit/06_simulator"
SCRIPTS="$HOME/VLA_tcs2/scripts"
LOGDIR="$BASE_OUT/mj332_D_logs"
mkdir -p "$LOGDIR"

# 由 preflight_policy.py 自动推导 rename_map（image/wrist_image -> camera1/camera2）
PREFLIGHT=$(python3 "$SCRIPTS/preflight_policy.py" "hub:$POLICY") || { echo "[error] preflight 失败: $POLICY"; exit 1; }
POLICY_PATH="${PREFLIGHT%%|*}"
RENAME_MAP="${PREFLIGHT##*|}"
echo "[preflight] policy.path = $POLICY_PATH"
echo "[preflight] rename_map  = $RENAME_MAP"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for SUITE in "${SUITES[@]}"; do
    OUT="$BASE_OUT/mj332_D_${SUITE}"
    if [[ -f "$OUT/eval_info.json" ]]; then
        echo "[skip] $SUITE 已有结果: $OUT/eval_info.json"
        continue
    fi
    echo "[run ] $SUITE -> $OUT"
    lerobot-eval \
        --policy.path="$POLICY_PATH" \
        --policy.n_action_steps=1 \
        --policy.num_steps=10 \
        --env.type=libero \
        --env.task="$SUITE" \
        --eval.n_episodes=10 \
        --eval.batch_size=1 \
        --env.max_parallel_tasks=1 \
        --rename_map="$RENAME_MAP" \
        --seed="$SEED" \
        --output_dir="$OUT" \
        2>&1 | tee "$LOGDIR/${SUITE}.log"
done

echo ""
echo "=== 全部完成, 汇总各 suite eval_info.json 的 pc_success ==="
python3 - <<'PY'
import json
from pathlib import Path

base = Path.home() / "VLA_tcs2/outputs/table2_repro_audit/06_simulator"
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    f = base / f"mj332_D_{suite}/eval_info.json"
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
