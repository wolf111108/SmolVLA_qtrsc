#!/usr/bin/env bash
# PHASE 6 后续 — C 类 tiantianx/smolvla_libero 四 suite 在 mujoco 3.3.2 环境下的完整评测
# 目的：MuJoCo 版本降级后重测 C 类（paper-like 复现候选），对照 3.8.1 下的 S76/O59/G63/L40, 平均59.5%
#
# C 类 = paper_like_reproduction_candidate (README handoff §9):
#   tiantianx/smolvla_libero — 16 层 / 0.75 宽 / 100k steps / expert-only / frozen VLM
#   需 rename_map: image->camera1, image2->camera2 (config 中 camera3 为 stale, 无需补)
#
# 用法: 在 smolvla_eval 环境执行: bash run_phase6_mj332_C_foursuite.sh
# 输出: ~/VLA_tcs2/outputs/table2_repro_audit/06_simulator/mj332_C_<suite>/eval_info.json
#
# 注意:
# - 必须先 conda activate smolvla_eval (mujoco 3.3.2)
# - 四 suite = 400 episodes, Long(520步)最慢, 预计 8-12h
# - 幂等: 已有 eval_info.json 的 suite 自动跳过

set -euo pipefail

cd "$HOME/VLA_tcs2/lerobot_current"

POLICY="tiantianx/smolvla_libero"
SEED=1000
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
BASE_OUT="$HOME/VLA_tcs2/outputs/table2_repro_audit/06_simulator"
LOGDIR="$BASE_OUT/mj332_C_logs"
mkdir -p "$LOGDIR"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for SUITE in "${SUITES[@]}"; do
    OUT="$BASE_OUT/mj332_C_${SUITE}"
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
    f = base / f"mj332_C_{suite}/eval_info.json"
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
