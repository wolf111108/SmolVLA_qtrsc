#!/usr/bin/env bash
# tiantianx/smolvla_libero (paper-like 复现候选, C 类) 在全部四个 LIBERO suite 上的 evaluation
# 用法: bash run_tiantianx_libero_suites.sh
# 输出: ~/VLA_tcs2/outputs/tiantianx_smolvla_libero_<suite>/eval_info.json
#
# 与 run_lerobot_libero_suites.sh 的区别:
# - POLICY 换成 tiantianx/smolvla_libero (16 层 / 0.75 宽 / 100k steps / expert-only / frozen VLM)
# - Spatial 也要跑(tiantianx 此前只有 Task0/Task5 的零星 smoke test)
# - 跑完自动调用 compare_tiantianx_vs_lerobot.py 生成与 lerobot/smolvla_libero 的对比报告
#
# 已确认事项(审计见 outputs/audit_tiantianx_normalizer.txt 与 outputs/compare_libero_stats.txt):
# - normalizer 实际是 8D state, 与 HuggingFaceVLA/libero dataset stats 完全一致
# - config 中 state=6D 是 stale metadata, 不影响推理
# - 需要 rename_map(image->camera1, image2->camera2); 不需要补 camera3(Task0 已验证 1/1 success)
#
# 注意:
# - 每个 suite = 10 tasks × 10 episodes, 预计每个 suite ~1.5-4h(Long 最长 520 步), 四个 suite 共 ~8-12h
# - 需在 smolvla_eval 环境运行, 且 MUJOCO_GL=egl, MUJOCO_EGL_DEVICE_ID=2 已设置

set -euo pipefail

cd "$HOME/VLA_tcs2/lerobot_current"

POLICY="tiantianx/smolvla_libero"
SEED=1000
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
BASE_OUT="$HOME/VLA_tcs2/outputs"
LOGDIR="$BASE_OUT/tiantianx_smolvla_libero_multisuite_logs"
mkdir -p "$LOGDIR"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for SUITE in "${SUITES[@]}"; do
    OUT="$BASE_OUT/tiantianx_smolvla_libero_${SUITE}"
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
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    f = base / f"tiantianx_smolvla_libero_{suite}/eval_info.json"
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
echo "=== 生成与 lerobot/smolvla_libero 的对比报告 ==="
python3 "$HOME/VLA_tcs2/scripts/compare_tiantianx_vs_lerobot.py"
