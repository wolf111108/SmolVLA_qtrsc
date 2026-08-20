#!/usr/bin/env bash
# lerobot/smolvla_libero 在 LIBERO Object / Goal / Long 上的分 suite evaluation
# 用法: bash run_lerobot_libero_suites.sh [--skip-spatial]
# 输出: ~/VLA_tcs2/outputs/lerobot_smolvla_libero_<suite>/eval_info.json
#
# 注意:
# - Spatial 已有历史结果(livariate 81/100), 默认跳过; 如需重跑加 --skip-spatial 去掉
# - 每个 suite = 10 tasks × 10 episodes, 预计每个 suite ~1.5-4h(Long 最长 520 步)
# - 需在 smolvla_eval 环境运行, 且 MUJOCO_GL=egl, MUJOCO_EGL_DEVICE_ID=2 已设置

set -euo pipefail

cd "$HOME/VLA_tcs2/lerobot_current"

POLICY="lerobot/smolvla_libero"
SEED=1000
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
BASE_OUT="$HOME/VLA_tcs2/outputs"
LOGDIR="$BASE_OUT/lerobot_smolvla_libero_multisuite_logs"
mkdir -p "$LOGDIR"

SUITES=("libero_object" "libero_goal" "libero_10")

for SUITE in "${SUITES[@]}"; do
    OUT="$BASE_OUT/lerobot_smolvla_libero_${SUITE}"
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

base = Path.home() / "VLA_tcs2/outputs"
for suite in ["libero_object", "libero_goal", "libero_10"]:
    f = base / f"lerobot_smolvla_libero_{suite}/eval_info.json"
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
