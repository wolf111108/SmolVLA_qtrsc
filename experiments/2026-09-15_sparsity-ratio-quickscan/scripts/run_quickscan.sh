#!/usr/bin/env bash
# Q0-Q3 sparsity quick scan runner.
#
# 4 组串行：Q0 INT8 / Q1 INT16 / Q2 FP8(A/O)-W4 / Q3 FP8。
# 协议：libero_goal task0 × 1ep，n_action_steps=10，num_steps=10。
# 只做 workload characterization，不做正式 SR claim。
#
# 用法：
#   bash experiments/2026-09-15_sparsity-ratio-quickscan/scripts/run_quickscan.sh
#
# 关于 GPU 选择（2026-09-15 修订，原因见 docs/logs.md §4.8）：
#   **本脚本不设置 CUDA_VISIBLE_DEVICES，也不修改 MUJOCO_EGL_DEVICE_ID。**
#   这两个变量在本机是耦合的：
#     1. robosuite 断言 `MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`（子串判断，
#        site-packages/robosuite/utils/binding_utils.py:35）；
#     2. `smolvla_eval` 的 `conda env config vars` 里固定了
#        `MUJOCO_EGL_DEVICE_ID=2`，`conda run` 会把 shell export 覆盖回 2。
#   因此一旦限制 CUDA_VISIBLE_DEVICES=0，conda run 仍注入 EGL=2 → "2" not in "0"
#   → AssertionError。且即使绕开断言（用 env 前缀强制 EGL=0），EGL 设备索引与
#   CUDA 索引也未必指向同一张物理卡，隔离语义不可靠。
#   Phase F/G/H 的脚本一律不设 CUDA_VISIBLE_DEVICES（断言被跳过），本机只有 1 张
#   H100 NVL，故沿用同一约定；GPU 占用情况在启动时打印供人工确认。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_NAME="2026-09-15_sparsity-ratio-quickscan"
TASK="$REPO_ROOT/experiments/${EXP_NAME}"
OUT_ROOT="$REPO_ROOT/outputs/${EXP_NAME}"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 渲染 / 设备环境：显式重置，不继承调用者的污染值
# ---------------------------------------------------------------------------
# 为什么必须 unset CUDA_VISIBLE_DEVICES：
#   robosuite 有硬断言（site-packages/robosuite/utils/binding_utils.py:33）
#     if CUDA_VISIBLE_DEVICES != "":
#         assert MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES   # 字符串子串判断
#   只要 CUDA_VISIBLE_DEVICES 非空而 MUJOCO_EGL_DEVICE_ID 不在其中，就 AssertionError。
#   而 `smolvla_eval` 的 conda env config vars 固定注入 MUJOCO_EGL_DEVICE_ID=2，
#   `conda run` 会覆盖 shell 的 export —— 所以只要父会话残留
#   `CUDA_VISIBLE_DEVICES=0`，就会 "2" not in "0" 直接崩。
#   交互式 bash 会话的变量跨命令持久（曾多次踩到），故这里主动清除。
# 为什么不做设备隔离：CUDA_VISIBLE_DEVICES 只重映射 CUDA runtime 的可见设备，
#   不影响 MuJoCo/EGL 自己的设备枚举，二者索引未必指向同一物理卡。本机只有
#   1 张 H100 NVL，隔离既无收益又引入静默错卡风险。详见 docs/logs.md §4.8。
unset CUDA_VISIBLE_DEVICES

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=2   # 与 conda env config vars 保持一致
export PYTHONUNBUFFERED=1

echo "渲染环境：MUJOCO_GL=${MUJOCO_GL} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

CONDA_RUN=(conda run --no-capture-output -n smolvla_eval)

# config : skip_calibration
#   **四组一律 1（= 一律 --skip-calibration）**，原因见 docs/logs.md §4.9：
#   `calibrate()`（src/vla_tcs2/calibration.py:102）会无条件 new 一个
#   QuantStatManager 并覆盖 `module._stat_manager`，而该新实例的
#   `sparsity_enabled` 默认 False（main.py 只对 `wrapper.stat_manager` 调过
#   enable_sparsity）。于是「同一次 run 里既校准又统计」会让 runtime sparsity
#   静默全空（CSV 只有表头、workload rows: 0），且不报错。
#   因此把「生成 scale」与「采集 sparsity」拆成两个 run：
#     阶段 1  --skip-evaluation  → 只校准、生成 scale（不导出 sparsity）
#     阶段 2  --skip-calibration → 只 eval、正常采集 sparsity
#   下方 ensure_scales() 自动完成阶段 1（幂等，scale 齐全则跳过）。
RUNS=(
  "q0_int8:1"
  "q1_int16:1"
  "q2_fp8w4_po2:1"
  "q3_fp8_po2:1"
)

# 需要 fresh calibration 的组（阶段 1 生成 scale）；其余组复用既有 scale。
NEEDS_CALIBRATION=(
  "q0_int8"
  "q1_int16"
)

EXPECTED_SCALE_FILES=864   # 224 Linear x 3 + 64 MatMul x 3

echo "=================================================================="
echo "Sparsity quick scan — ${EXP_NAME}"
echo "  start = $(date '+%F %T')"
echo "=================================================================="
echo "GPU 占用（供人工确认，本脚本不做设备隔离，见文件头注释）："
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader | sed 's/^/  /' || true
echo "同一时刻在跑的本仓库 main.py 任务："
pgrep -af "main.py --config" 2>/dev/null | grep -v "conda run" \
  | sed 's/.*configs\///;s/.*generated\///;s/ --skip-calibration//;s/^/  /' || echo "  （无）"
echo "=================================================================="

failed=()

# ---------------------------------------------------------------------------
# 阶段 1：为 NEEDS_CALIBRATION 的组生成 scale（只校准，不 eval、不导出）
# ---------------------------------------------------------------------------
for name in "${NEEDS_CALIBRATION[@]}"; do
  cfg="$TASK/configs/${name}.yaml"
  out="$OUT_ROOT/${name}"
  scale_dir="$REPO_ROOT/scales/2026-09-15_sparsity-ratio-quickscan/${name}"
  mkdir -p "$out"

  # 将 config 里的 scale_dir 转成绝对路径以核对文件数
  n_have=$(find "$scale_dir" -type f -name '*.p' 2>/dev/null | wc -l)
  if (( n_have >= EXPECTED_SCALE_FILES )); then
    echo "[SCALE-OK] $name — 已存在 ${n_have} 个 scale 文件，跳过阶段 1"
    continue
  fi

  echo ""
  echo "[CALIB] $name  ($(date '+%F %T')) — 阶段 1：仅校准生成 scale（--skip-evaluation）"
  if "${CONDA_RUN[@]}" python main.py --config "$cfg" --skip-evaluation \
       2>&1 | tee "$out/calib.log"; then
    n_after=$(find "$scale_dir" -type f -name '*.p' 2>/dev/null | wc -l)
    echo "[CALIB-DONE] $name — 生成 ${n_after} 个 scale 文件"
    if (( n_after < EXPECTED_SCALE_FILES )); then
      echo "[FAIL] $name — scale 文件数 ${n_after} < 期望 ${EXPECTED_SCALE_FILES}"
      failed+=("$name(calib)")
    fi
  else
    echo "[FAIL] $name — 阶段 1 校准失败，见 $out/calib.log"
    failed+=("$name(calib)")
  fi
done

# ---------------------------------------------------------------------------
# 阶段 2：四组统一 --skip-calibration 跑 eval 并采集 sparsity
# ---------------------------------------------------------------------------
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

  # 已完成的组跳过（幂等续跑）。
  # 判据必须同时满足三点，否则会误判：
  #   1) module + weight 两个 primary 产物都存在（main.py 的 export 顺序是
  #      module → ... → weight，只查前者会在中途崩溃时误判完成）；
  #   2) module_sparsity.csv 有数据行（> 1 行）——config-only 的 header-only
  #      CSV 是真实出现过的失败形态（见 docs/logs.md §4.9）；
  #   3) 该组所依赖的 scale 齐全。
  if [[ -f "$out/sparsity/module_sparsity.csv" &&
        -f "$out/sparsity/weight_sparsity_static.csv" ]] \
     && (( $(wc -l < "$out/sparsity/module_sparsity.csv") > 1 )); then
    echo "[SKIP] $name 已完成且 sparsity 非空"
    continue
  fi

  args=(python main.py --config "$cfg")
  if [[ "$skip_cal" == "1" ]]; then
    args+=(--skip-calibration)
  fi

  echo ""
  echo "[RUN] $name  ($(date '+%F %T'))  [skip-calibration=$skip_cal]"
  if "${CONDA_RUN[@]}" "${args[@]}" 2>&1 | tee "$out/run.log"; then
    if (( $(wc -l < "$out/sparsity/module_sparsity.csv" 2>/dev/null || echo 0) > 1 )); then
      echo "[DONE] $name  ($(date '+%F %T'))"
    else
      echo "[FAIL] $name — module_sparsity.csv 为空（仅表头），见 §4.9 同类问题"
      failed+=("$name(empty-sparsity)")
    fi
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
