#!/usr/bin/env bash
# PHASE 6 后续 — B 类 HuggingFaceVLA/smolvla_libero 四 suite 在 mujoco 3.3.2 环境下的完整评测
# 目的：MuJoCo 版本降级后重测 B 类（官方 current reference），对照 3.8.1 下的 S65/O71/G72/L37, 平均61.2%
#
# B 类 = official_current_reference (README handoff §9):
#   HuggingFaceVLA/smolvla_libero — 32 层 / 0.5 宽 / 500M-Instruct / 8D state + image/image2
#   该 checkpoint config 与 LIBERO processor 完全一致，无需 rename_map
#
# 用法: 在 smolvla_eval_mj332 环境执行: bash run_phase6_mj332_B_foursuite.sh
# 输出: ~/VLA_tcs2/outputs/table2_repro_audit/06_simulator/mj332_B_<suite>/eval_info.json
#
# 注意:
# - 必须先 conda activate smolvla_eval_mj332 (mujoco 3.3.2)
# - 四 suite = 400 episodes, Long(520步)最慢, 预计 8-12h
# - 幂等: 已有 eval_info.json 的 suite 自动跳过

set -euo pipefail

cd "$HOME/VLA_tcs2/lerobot_current"

POLICY="HuggingFaceVLA/smolvla_libero"
SEED=1000
# B 类无需 rename_map（config 即 8D state + image/image2，与 LIBERO processor 一致）
RENAME_MAP='{}'
BASE_OUT="$HOME/VLA_tcs2/outputs/table2_repro_audit/06_simulator"
LOGDIR="$BASE_OUT/mj332_B_logs"
mkdir -p "$LOGDIR"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for SUITE in "${SUITES[@]}"; do
    OUT="$BASE_OUT/mj332_B_${SUITE}"
    if [[ -f "$OUT/eval_info.json" ]]; then
        echo "[skip] $SUITE 已有结果: $OUT/eval_info.json"
        continue
    fi
    echo "[run ] $SUITE -> $OUT"
    lerobot-eval \
        --policy.path="$POLICY" \
        --policy.n_action_steps=10 \
        --policy.num_steps=10 \
        --env.type=libero \
        --env.task="$SUITE" \
        --eval.n_episodes=50 \
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
    f = base / f"mj332_B_{suite}/eval_info.json"
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
