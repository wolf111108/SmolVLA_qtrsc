#!/usr/bin/env bash
# PHASE 5 实验 B (mj332 版) — 四 suite combined（共享 RNG 流），A 类 lerobot/smolvla_libero
# 目的：在 mujoco 3.3.2 环境下重跑 combined，与逐 suite 串行(Phase 6 mj332_A)对比 ΔLong，
#       以及 combined 在 3.8.1 vs 3.3.2 的版本差异。
#
# 用法: 先 conda activate smolvla_eval (mujoco 3.3.2)，再 bash run_phase5_combined_mj332.sh
# 输出: ~/VLA_tcs2/outputs/table2_repro_audit/05_rng_ordering/combined_mj332/eval_info.json
#
# 注意:
# - 400 episodes, Long(520步)最慢, 预计 8-12h, 建议 nohup 后台
# - A 类需要 rename_map(camera1/2/3 -> image/image2)
# - 需 smolvla_eval 环境 + EGL(device 需在本机重新枚举, 不一定是 2)

set -euo pipefail

cd "$HOME/VLA_tcs2/lerobot_current"

POLICY="lerobot/smolvla_libero"
SEED=1000
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
OUT="$HOME/VLA_tcs2/outputs/table2_repro_audit/05_rng_ordering/combined_mj332"

if [[ -f "$OUT/eval_info.json" ]]; then
    echo "[skip] 已有结果: $OUT/eval_info.json"
    exit 0
fi

echo "[run ] combined 四 suite (mujoco 3.3.2) -> $OUT"
lerobot-eval \
    --policy.path="$POLICY" \
    --policy.n_action_steps=1 \
    --policy.num_steps=10 \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --env.max_parallel_tasks=1 \
    --rename_map="$RENAME_MAP" \
    --seed="$SEED" \
    --output_dir="$OUT"