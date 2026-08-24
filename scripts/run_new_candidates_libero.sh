#!/usr/bin/env bash
# 跑两个新候选 checkpoint 的完整 LIBERO 四 suite benchmark + 记录结果
# 候选 1: HuggingFaceVLA/smolvla_libero_ckpts 的 100000/pretrained_model (官方, 16/0.75, DEPRECIATED)
# 候选 2: k1000dai/smolvla_libero_finetune (社区, 16/0.75, 100k steps, batch64, from smolvla_base)
#
# 流程(per 模型):
#   1. preflight: 下载/解析 config, 自动推导 rename_map
#   2. smoke test: Spatial task0 x 1 episode, 失败则跳过该模型(不阻塞另一个)
#   3. 四 suite x (10 tasks x 10 episodes), 幂等(已有 eval_info.json 跳过)
#   4. 汇总 + 写入 logs/ 当日日志
#
# 输出:
#   outputs/<shortname>_<suite>/eval_info.json
#   logs/<date>_<shortname>_vs_baselines.md
#
# 运行环境: conda activate smolvla_eval (MUJOCO_GL=egl, MUJOCO_EGL_DEVICE_ID=2)

set -uo pipefail

cd "$HOME/VLA_tcs2/lerobot_current"
BASE_OUT="$HOME/VLA_tcs2/outputs"
SCRIPTS="$HOME/VLA_tcs2/scripts"
LOGS="$HOME/VLA_tcs2/logs"
SEED=1000
mkdir -p "$BASE_OUT"

# spec 格式见 preflight_policy.py: hub:<repo> / hfsub:<repo>:<subfolder> / dir:<本地路径>
# hfvla  : 老版训练轨迹 checkpoint, 修复三处后可用:
#          1) config 剔除 gradient_accumulation_steps
#          2) config 里训练机本地 vlm 路径 /raid/... 改为 HuggingFaceTB/SmolVLM2-2.2B-Instruct
#          3) migrate_policy_normalization.py 迁移到 processor 格式
#          -> checkpoints/hfvla_smolvla_libero_100k_migrated
# k1000dai: 旧格式, 已迁移 -> lerobot_current/k1000dai_smolvla_libero_finetune_migrated
MODEL_SPECS=(
    "dir:$HOME/VLA_tcs2/checkpoints/hfvla_smolvla_libero_100k_migrated|hfvla_ckpts100k"
    "dir:$HOME/VLA_tcs2/lerobot_current/k1000dai_smolvla_libero_finetune_migrated|k1000dai_ft100k"
)

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for entry in "${MODEL_SPECS[@]}"; do
    SPEC="${entry%%|*}"
    SHORT="${entry##*|}"
    echo ""
    echo "=============================== MODEL: $SHORT ==============================="
    echo "[preflight] $SPEC"

    # 1. preflight: 解析 policy.path + rename_map
    PREFLIGHT=$(python3 "$SCRIPTS/preflight_policy.py" "$SPEC") || { echo "[skip] preflight 失败: $SHORT"; continue; }
    POLICY_PATH="${PREFLIGHT%%|*}"
    RENAME_MAP="${PREFLIGHT##*|}"
    echo "[preflight] policy.path = $POLICY_PATH"
    echo "[preflight] rename_map  = $RENAME_MAP"

    # 2. smoke test: 连加载都过不了就跳过该模型
    #    (注意: 之前失败运行的 smoke 目录要清掉才会重试)
    SMOKE_OUT="$BASE_OUT/${SHORT}_smoke"
    rm -rf "$SMOKE_OUT"
    if [[ -f "$SMOKE_OUT/eval_info.json" ]]; then
        echo "[skip] smoke 已通过: $SMOKE_OUT/eval_info.json"
    else
        echo "[smoke ] Spatial task0 x 1 episode -> $SMOKE_OUT"
        if ! lerobot-eval \
            --policy.path="$POLICY_PATH" \
            --policy.n_action_steps=1 \
            --policy.num_steps=10 \
            --env.type=libero \
            --env.task=libero_spatial \
            --env.task_ids='[0]' \
            --eval.n_episodes=1 \
            --eval.batch_size=1 \
            --env.max_parallel_tasks=1 \
            --rename_map="$RENAME_MAP" \
            --seed="$SEED" \
            --output_dir="$SMOKE_OUT" \
            > "$SMOKE_OUT.log" 2>&1; then
            echo "[skip] smoke 失败, 跳过 $SHORT (日志: $SMOKE_OUT.log)"
            tail -5 "$SMOKE_OUT.log"
            continue
        fi
        echo "[smoke ] 通过"
    fi

    # 3. 四 suite
    for SUITE in "${SUITES[@]}"; do
        OUT="$BASE_OUT/${SHORT}_${SUITE}"
        if [[ -f "$OUT/eval_info.json" ]]; then
            echo "[skip] $SUITE 已有结果"
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
            2>&1 | tee "$BASE_OUT/${SHORT}_${SUITE}.log"
    done

    # 4. 汇总该模型
    echo ""
    echo "=== $SHORT 四 suite 汇总 ==="
    python3 - "$SHORT" <<'PY'
import json
import sys
from pathlib import Path

short = sys.argv[1]
base = Path.home() / "VLA_tcs2/outputs"
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    f = base / f"{short}_{suite}/eval_info.json"
    if not f.exists():
        print(f"{suite}: 未完成")
        continue
    per_task = json.loads(f.read_text()).get("per_task", [])
    if isinstance(per_task, dict):
        per_task = list(per_task.values())
    n = sum(sum(t["metrics"]["successes"]) for t in per_task)
    d = sum(len(t["metrics"]["successes"]) for t in per_task)
    print(f"{suite}: {n}/{d} = {100.0*n/max(d,1):.0f}%")
PY
done

echo ""
echo "=== 生成对比报告(新候选 vs A/C 类基线) ==="
python3 "$SCRIPTS/compare_new_candidates_vs_baselines.py"
