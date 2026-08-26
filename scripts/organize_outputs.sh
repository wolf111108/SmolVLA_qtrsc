#!/usr/bin/env bash
# organize_outputs.sh — 将 outputs/ 按类别归入子目录（条目保持原名，便于追踪）
#
# 用法: bash scripts/organize_outputs.sh   （在仓库根目录执行）
#
# 说明:
#   - 已跟踪文件用 git mv（保留历史），未跟踪用 mv，自动判断
#   - outputs/table2_repro_audit/ 不动（AUDIT_RECORD 编号体系 + 活跃任务 + 脚本引用）
#   - 移动后请 git add -A 并提交
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="outputs"

# --- 1. 建分类目录 ---
mkdir -p \
    "$OUT/libero/logs" \
    "$OUT/metaworld" \
    "$OUT/reports" \
    "$OUT/figures" \
    "$OUT/misc"

# --- 2. move helper: git mv 优先, 未跟踪则 mv ---
move() {
    local src="$1" dst="$2"
    if [[ ! -e "$src" ]]; then
        echo "[skip] 不存在: $src"
        return 0
    fi
    if git ls-files --error-unmatch -- "$src" >/dev/null 2>&1; then
        git mv "$src" "$dst"
    else
        mv "$src" "$dst"
    fi
    echo "[mv  ] $src -> $dst"
}

# --- 3. LIBERO 评测（A / hfvla / D / C + 早期 baseline + rerun + 日志） ---
for s in libero_spatial libero_object libero_goal libero_10; do
    move "$OUT/lerobot_smolvla_libero_libero_$s"    "$OUT/libero/lerobot_smolvla_libero_libero_$s"
    move "$OUT/hfvla_libero_libero_$s"              "$OUT/libero/hfvla_libero_libero_$s"
    move "$OUT/k1000dai_ft100k_libero_$s"           "$OUT/libero/k1000dai_ft100k_libero_$s"
    move "$OUT/tiantianx_smolvla_libero_libero_$s"  "$OUT/libero/tiantianx_smolvla_libero_libero_$s"
done
move "$OUT/baseline_smolvla450m_libero_spatial"     "$OUT/libero/baseline_smolvla450m_libero_spatial"
move "$OUT/libero_spatial_rerun_v2"                "$OUT/libero/libero_spatial_rerun_v2"
move "$OUT/lerobot_smolvla_libero_multisuite_logs" "$OUT/libero/logs/lerobot_smolvla_libero_multisuite_logs"
move "$OUT/multisuite_nohup.log"                   "$OUT/libero/logs/multisuite_nohup.log"

# --- 4. Meta-World 评测 ---
move "$OUT/metaworld_mt50_lerobot_smolvla" "$OUT/metaworld/metaworld_mt50_lerobot_smolvla"
move "$OUT/metaworld_mt50_run.log"         "$OUT/metaworld/metaworld_mt50_run.log"
move "$OUT/metaworld_push_v3_10ep"         "$OUT/metaworld/metaworld_push_v3_10ep"
move "$OUT/metaworld_smoke_push_mj332"     "$OUT/metaworld/metaworld_smoke_push_mj332"
move "$OUT/metaworld_smoke_push_origcfg"   "$OUT/metaworld/metaworld_smoke_push_origcfg"

# --- 5. 汇总报告 / 图表 / misc ---
for f in checkpoint_baselines_summary.md hfvla_vs_baselines.md new_candidates_vs_baselines.md \
         lerobot_smolvla_libero_vs_paper.md tiantianx_vs_lerobot_libero.md \
         audit_hfvla_ckpts100k.txt audit_tiantianx_normalizer.txt compare_libero_stats.txt; do
    move "$OUT/$f" "$OUT/reports/$f"
done
move "$OUT/benchmark_libero.png"    "$OUT/figures/benchmark_libero.png"
move "$OUT/benchmark_metaworld.png" "$OUT/figures/benchmark_metaworld.png"
move "$OUT/eval"                    "$OUT/misc/eval"
move "$OUT/vla_tcs2_identity_task0" "$OUT/misc/vla_tcs2_identity_task0"

# --- 6. 完成 ---
echo ""
echo "=== 完成。请检查并提交 ==="
echo "    git status"
echo "    git add -A && git commit -m 'reorganize outputs/ into categorized subdirs'"
