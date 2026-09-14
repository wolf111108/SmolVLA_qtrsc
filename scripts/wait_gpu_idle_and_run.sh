#!/usr/bin/env bash
# 轮询 GPU 显存使用量，连续 N 次满足空闲条件后启动目标程序。
#
# 用法:
#   bash scripts/wait_gpu_idle_and_run.sh '<要启动的命令>'
#
# 示例:
#   bash scripts/wait_gpu_idle_and_run.sh \
#     'bash experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/run_h0_smoke.sh'
#
# 后台运行（推荐，配合 nohup 可脱离终端）:
#   nohup bash scripts/wait_gpu_idle_and_run.sh '<命令>' > wait_gpu.log 2>&1 &
#
# 环境变量（可选覆盖默认值）:
#   INTERVAL     轮询间隔（秒），默认 300（5 分钟）
#   REQUIRED     需连续满足的次数，默认 10
#   MEM_THRESHOLD  空闲显存阈值（MiB），默认 20000
#   GPU_INDEX    指定检查的 GPU 序号，默认检查所有 GPU（任一超阈值即不满足）
set -euo pipefail

INTERVAL="${INTERVAL:-300}"
REQUIRED="${REQUIRED:-10}"
MEM_THRESHOLD="${MEM_THRESHOLD:-20000}"
GPU_INDEX="${GPU_INDEX:-}"

CMD="${1:-}"
if [[ -z "$CMD" ]]; then
    echo "用法: $0 '<要启动的命令>'" >&2
    echo "示例: $0 'bash experiments/.../run_h0_smoke.sh'" >&2
    exit 1
fi

# 查询目标 GPU 的最大显存使用量（MiB）。
# 返回全局变量 GPU_USED_MAX。
get_max_used() {
    local q
    if [[ -n "$GPU_INDEX" ]]; then
        q=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "$GPU_INDEX" 2>/dev/null)
    else
        q=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    fi
    if [[ -z "$q" ]]; then
        return 1
    fi
    GPU_USED_MAX=$(echo "$q" | tr -d ' ' | sort -n | tail -1)
    return 0
}

is_idle() {
    # 返回 0 表示满足空闲条件（所有目标 GPU 显存使用 < 阈值）。
    if ! get_max_used; then
        echo "[$(date '+%F %T')] nvidia-smi 查询失败，视为不满足" >&2
        return 1
    fi
    if (( GPU_USED_MAX < MEM_THRESHOLD )); then
        return 0
    fi
    return 1
}

consecutive=0
while true; do
    if is_idle; then
        consecutive=$((consecutive + 1))
        echo "[$(date '+%F %T')] GPU 空闲 (max_used=${GPU_USED_MAX}MiB < ${MEM_THRESHOLD}MiB)，连续满足 ${consecutive}/${REQUIRED} 次"
    else
        consecutive=0
        echo "[$(date '+%F %T')] GPU 占用中 (max_used=${GPU_USED_MAX}MiB >= ${MEM_THRESHOLD}MiB)，计数重置"
    fi

    if (( consecutive >= REQUIRED )); then
        echo "[$(date '+%F %T')] 连续 ${REQUIRED} 次满足空闲条件，启动程序: ${CMD}"
        eval "$CMD"
        echo "[$(date '+%F %T')] 程序已结束，脚本退出"
        break
    fi

    sleep "$INTERVAL"
done
