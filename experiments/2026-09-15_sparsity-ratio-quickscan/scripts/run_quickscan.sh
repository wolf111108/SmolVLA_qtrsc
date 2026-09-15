#!/usr/bin/env bash
# Q0-Q3 sparsity quick scan runner.
#
# 4 组串行：Q0 INT8 / Q1 INT16 / Q2 FP8(A/O)-W4 / Q3 FP8。
# 协议：libero_goal task0 × 1ep，n_action_steps=10，num_steps=10。
# 只做 workload characterization，不做正式 SR claim。
#
# 用法：
#   GPU=0 bash experiments/2026-09-15_sparsity-ratio-quickscan/scripts/run_quickscan.sh
#
# GPU：CUDA_VISIBLE_DEVICES 的目标 GPU 索引。**必填**由调用方显式指定，
#      以免与其它实验（如 Phase H 的 H3）抢占同一张卡。
#
# 失败语义（按 experiments/README 的 2026-09-15 sweep 标准）：
#   - 不用 `set -e`：单组失败不连坐后续组；末尾若有过失败则以非零码退出。
#   - 每组 tee 到 <out>/run.log，标记行带时间戳。
#   - conda run --no-capture-output 让输出流式落盘，避免块缓冲导致的 0 字节日志。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_NAME="2026-09-15_sparsity-ratio-quickscan"
TASK="$REPO_ROOT/experiments/${EXP_NAME}"
OUT_ROOT="$REPO_ROOT/outputs/${EXP_NAME}"

cd "$REPO_ROOT"

if [[ -z "${GPU:-}" ]]; then
  echo "ERROR: 必须显式指定 GPU，例如 GPU=0 bash $0" >&2
  echo "       当前机器 GPU 列表：" >&2
  nvidia-smi -L >&2 || true
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU"

# EGL / headless 渲染（与 Phase G/H 一致）
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-2}"
export PYTHONUNBUFFERED=1

CONDA_RUN=(conda run --no-capture-output -n smolvla_eval)

# config : skip_calibration
#   Q0/Q1 需要 fresh quick calibration（各自独立 scale_dir）
#   Q2/Q3 复用本地只读 scale 副本 ⇒ 必须 --skip-calibration，否则会覆盖副本
RUNS=(
  "q0_int8:0"
  "q1_int16:0"
  "q2_fp8w4_po2:1"
  "q3_fp8_po2:1"
)

echo "=================================================================="
echo "Sparsity quick scan — ${EXP_NAME}"
echo "  GPU (CUDA_VISIBLE_DEVICES) = ${GPU}"
echo "  MUJOCO_EGL_DEVICE_ID       = ${MUJOCO_EGL_DEVICE_ID}"
echo "  start                      = $(date '+%F %T')"
echo "=================================================================="

failed=()

for entry in "${RUNS[@]}"; do
  name="${entry%%:*}"
  skip_cal="${entry##*:}"

  cfg="$TASK/configs/${name}.yaml"
  out="$OUT_ROOT/${name}"
  mkdir -p "$out"

  if [[ ! -f "$cfg" ]]; then
    echo "[FAIL] $name — config 不存在: $cfg"
    failed+=("$name")
    continue
  fi

  # 已完成的组跳过（幂等续跑）
  if [[ -f "$out/sparsity/module_sparsity.csv" ]]; then
    echo "[SKIP] $name 已产出 sparsity/module_sparsity.csv"
    continue
  fi

  args=(python main.py --config "$cfg")
  if [[ "$skip_cal" == "1" ]]; then
    args+=(--skip-calibration)
  fi

  echo ""
  echo "[RUN] $name  ($(date '+%F %T'))  [skip-calibration=$skip_cal]"
  if "${CONDA_RUN[@]}" "${args[@]}" 2>&1 | tee "$out/run.log"; then
    echo "[DONE] $name  ($(date '+%F %T'))"
  else
    echo "[FAIL] $name  ($(date '+%F %T')) — 见 $out/run.log"
    failed+=("$name")
  fi
done

echo ""
echo "=================================================================="
echo "汇总（$(date '+%F %T')）"
echo "=================================================================="
"${CONDA_RUN[@]}" python "$TASK/scripts/summarize_quick_sparsity.py" \
  --outputs "$OUT_ROOT" || {
    echo "[FAIL] summarize_quick_sparsity.py"
    failed+=("summarize")
  }

if (( ${#failed[@]} > 0 )); then
  echo ""
  echo "Quick scan finished WITH FAILURES: ${failed[*]}"
  exit 1
fi

echo ""
echo "Quick scan finished. end = $(date '+%F %T')"
