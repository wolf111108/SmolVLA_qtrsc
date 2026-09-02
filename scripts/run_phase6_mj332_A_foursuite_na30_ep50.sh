#!/usr/bin/env bash
# 消融补充 — A 类 lerobot/smolvla_libero 四 suite，n_action_steps=30, n_episodes=50
# 目的：补齐消融矩阵的 na=30 数据点，观察 n_action_steps 从 10 → 30 → 50 是否继续提升或饱和
#
# 用法: 在 smolvla_eval 环境执行: bash run_phase6_mj332_A_foursuite_na30_ep50.sh
# 输出: ~/VLA_tcs2/outputs/table2_repro_audit/06_simulator/mj332_A_na30_ep50_<suite>/eval_info.json
#
# 注意:
# - 必须先 conda activate smolvla_eval (mujoco 3.3.2)
# - A 类需要 rename_map(camera1/2 -> image/image2)
# - 四 suite = 4*50 = 200 episodes, Long(520步)最慢
# - 幂等: 已有 eval_info.json 的 suite 自动跳过

set -euo pipefail

cd "$HOME/VLA_tcs2/lerobot_current"

POLICY="lerobot/smolvla_libero"
SEED=1000
NA=30
EP=50
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
BASE_OUT="$HOME/VLA_tcs2/outputs/table2_repro_audit/06_simulator"
LOGDIR="$BASE_OUT/mj332_A_na${NA}_ep${EP}_logs"
mkdir -p "$LOGDIR"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for SUITE in "${SUITES[@]}"; do
    OUT="$BASE_OUT/mj332_A_na${NA}_ep${EP}_${SUITE}"
    if [[ -f "$OUT/eval_info.json" ]]; then
        echo "[skip] $SUITE 已有结果: $OUT/eval_info.json"
        continue
    fi
    echo "[run ] $SUITE -> $OUT"
    lerobot-eval \
        --policy.path="$POLICY" \
        --policy.n_action_steps="$NA" \
        --policy.num_steps=10 \
        --env.type=libero \
        --env.task="$SUITE" \
        --eval.n_episodes="$EP" \
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
    f = base / f"mj332_A_na30_ep50_{suite}/eval_info.json"
    if not f.exists():
        print(f"{suite}: 未完成")
        continue
    info = json.loads(f.read_text())
    per_task = info.get("per_task", [])
    if isinstance(per_task, dict):
        per_task = list(per_task.values())
    n_succ = sum(sum(t["metrics"]["successes"]) for t in per_task)
    n_tot = sum(len(t["metrics"]["successes"]) for t in per_task)
    print(f"{suite}: {n_succ}/{n_tot} = {100.0 * n_succ / max(n_tot, 1):.1f}%")
PY
