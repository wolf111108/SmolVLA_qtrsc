#!/usr/bin/env bash
# ============================================================
# 用量化框架跑完整 LIBERO 四 suite（FP baseline，验证 eval 等价性）
# 用法: bash scripts/run_verify_libero.sh
# 每个 suite 一个 config（n_episodes=50, ep50 协议），串行跑，最后汇总 pc_success
# 对照: 已有 lerobot-eval 结果 mj332_A_na10_ep10_<suite>（ep10 口径，spatial ~84%）
# ============================================================
set -euo pipefail

cd "$HOME/VLA_tcs2"

CFG_DIR="configs/experiments"
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for suite in "${SUITES[@]}"; do
    cfg="$CFG_DIR/verify_${suite}.yaml"
    out="outputs/verify_libero/${suite}"

    # 幂等：已有结果跳过
    if [[ -f "$out/eval_info.json" ]]; then
        echo "[skip] $suite 已有结果: $out/eval_info.json"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "跑 $suite (config: $cfg)"
    echo "============================================================"
    python main.py --config "$cfg"
done

echo ""
echo "============================================================"
echo "四 suite 汇总"
echo "============================================================"
python3 - <<'PY'
import json
from pathlib import Path

base = Path.home() / "VLA_tcs2/outputs/verify_libero"
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

total_succ = 0
total_eps = 0
for suite in suites:
    f = base / suite / "eval_info.json"
    if not f.exists():
        print(f"{suite:>16}: 未完成")
        continue
    info = json.loads(f.read_text())
    pt = info.get("per_task", [])
    if isinstance(pt, dict):
        pt = list(pt.values())
    n_succ = sum(sum(t["metrics"]["successes"]) for t in pt)
    n_tot = sum(len(t["metrics"]["successes"]) for t in pt)
    total_succ += n_succ
    total_eps += n_tot
    pct = 100.0 * n_succ / max(n_tot, 1)
    print(f"{suite:>16}: {n_succ}/{n_tot} = {pct:.1f}%")

if total_eps > 0:
    print("-" * 40)
    print(f"{'四 suite 合计':>16}: {total_succ}/{total_eps} = {100.0*total_succ/total_eps:.1f}%")

print()
print("对照 (lerobot-eval, mujoco 3.3.2):")
print("  ep50 协议 (每 task 50 episodes)，与附件 run_phase6_mj332_A_foursuite.sh 一致")
print("若四 suite 结果接近历史 mj332 值，说明 rename 环境 + 重写代码正确、eval 链路等价。")
PY
