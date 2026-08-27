#!/usr/bin/env bash
# ============================================================
# VLA_tcs2 新设备重建后验证脚本
# 用法:
#   bash scripts/verify_rebuild.sh          # 阶段 A(版本核对) + B(smoke)，~5 分钟
#   bash scripts/verify_rebuild.sh full     # 阶段 A + B + C(完整四 suite 验收，~8-12h)
#
# 验收标准: 完整 LIBERO 四 suite (spatial/object/goal/10)，
#           spatial pc_success ~84% (mujoco 3.3.2)
# ============================================================
set -euo pipefail

LEROBOT_COMMIT=6adf51511b7625090eade8d82d9f61a1846ebe56
MUJOCO_TARGET=3.3.2
POLICY="lerobot/smolvla_libero"
SEED=1000
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
BASE_OUT="$HOME/VLA_tcs2/outputs/table2_repro_audit/06_simulator"
LOGDIR="$BASE_OUT/rebuild_verify_logs"
MODE="${1:-}"

cd "$HOME/VLA_tcs2/lerobot_current"

echo "============================================================"
echo "阶段 A: 版本核对"
echo "============================================================"

fail=0

echo "--- git commit (预期 $LEROBOT_COMMIT) ---"
actual_commit=$(git rev-parse HEAD)
echo "$actual_commit"
if [[ "$actual_commit" != "$LEROBOT_COMMIT" ]]; then
    echo "[FAIL] lerobot commit 不一致!"; fail=1
else
    echo "[OK] lerobot commit"
fi

echo "--- mujoco (预期 $MUJOCO_TARGET) ---"
actual_mj=$(python -c "import mujoco; print(mujoco.__version__)")
echo "$actual_mj"
if [[ "$actual_mj" != "$MUJOCO_TARGET" ]]; then
    echo "[FAIL] mujoco 版本错误! 需 $MUJOCO_TARGET"; fail=1
else
    echo "[OK] mujoco $MUJOCO_TARGET"
fi

echo "--- torch + cuda ---"
python -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "--- EGL 环境变量 ---"
printenv MUJOCO_GL PYOPENGL_PLATFORM MUJOCO_EGL_DEVICE_ID || echo "[WARN] EGL 变量未完全设置"

echo "--- pip check ---"
pip check || echo "[WARN] pip check 有冲突(可能非致命)"

if [[ "$fail" -eq 1 ]]; then
    echo ""
    echo "[ABORT] 版本核对未通过，请先修复再继续。"
    exit 1
fi

echo ""
echo "============================================================"
echo "阶段 B: 单任务 smoke test (spatial task0, 1 episode)"
echo "============================================================"

SMOKE_OUT=$(mktemp -d)
lerobot-eval \
    --policy.path="$POLICY" \
    --policy.n_action_steps=10 --policy.num_steps=10 \
    --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
    --eval.n_episodes=1 --eval.batch_size=1 \
    --env.max_parallel_tasks=1 \
    --rename_map="$RENAME_MAP" \
    --seed="$SEED" \
    --output_dir="$SMOKE_OUT" 2>&1 | tail -5

if [[ -f "$SMOKE_OUT/eval_info.json" ]]; then
    echo "[OK] smoke test 通过 (eval_info.json 已生成)"
    python -c "import json,sys; d=json.load(open('$SMOKE_OUT/eval_info.json')); print('  overall pc_success =', d.get('overall',{}).get('pc_success'))"
else
    echo "[FAIL] smoke test 失败，检查 EGL/CUDA/驱动。"
    exit 1
fi

if [[ "$MODE" != "full" ]]; then
    echo ""
    echo "============================================================"
    echo "快速验证完成（阶段 A + B）。"
    echo "运行完整四 suite 验收: bash scripts/verify_rebuild.sh full"
    echo "============================================================"
    exit 0
fi

echo ""
echo "============================================================"
echo "阶段 C: 完整 LIBERO 四 suite 验收 (~8-12h)"
echo "============================================================"
mkdir -p "$LOGDIR"

# 复用官方四 suite 脚本（幂等：已有结果自动跳过）
bash "$HOME/VLA_tcs2/scripts/run_phase6_mj332_A_foursuite_10_10.sh" \
    2>&1 | tee "$LOGDIR/foursuite.log"

echo ""
echo "=== 四 suite 汇总 ==="
python3 - "$BASE_OUT" <<'PY'
import json, sys
from pathlib import Path

base = Path(sys.argv[1])
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
results = {}

for suite in suites:
    f = base / f"mj332_A_na10_ep10_{suite}/eval_info.json"
    if not f.exists():
        results[suite] = None
        print(f"{suite:>16}: 未完成")
        continue
    info = json.loads(f.read_text())
    pt = info.get("per_task", [])
    if isinstance(pt, dict):
        pt = list(pt.values())
    n_succ = sum(sum(t["metrics"]["successes"]) for t in pt)
    n_tot = sum(len(t["metrics"]["successes"]) for t in pt)
    pct = 100.0 * n_succ / max(n_tot, 1)
    results[suite] = pct
    print(f"{suite:>16}: {n_succ}/{n_tot} = {pct:.1f}%")

print()
# 验收判定：spatial 达到 ~84%
spatial = results.get("libero_spatial")
if spatial is None:
    print("[FAIL] spatial 未完成")
    sys.exit(1)

target = 84.0
if abs(spatial - target) <= 3.0:
    print(f"[PASS] spatial {spatial:.1f}% 达到 ~{target:.0f}% 目标 (容差 ±3pp)")
else:
    print(f"[CHECK] spatial {spatial:.1f}% 偏离 {target:.0f}% 目标超过 3pp，请排查 (mujoco 版本/EGL/GPU)")

# 打印完整对照（历史 3.8.1 参考值）
print()
print("参考: 历史 3.8.1 为 S81/O60/G77/L59 (平均 69.2%)；mujoco 3.3.2 目标整体提升")
PY

echo ""
echo "验证完成。"
