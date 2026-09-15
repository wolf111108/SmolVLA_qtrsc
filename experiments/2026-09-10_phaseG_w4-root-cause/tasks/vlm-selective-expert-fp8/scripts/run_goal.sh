#!/usr/bin/env bash
# G6 Goal ×100 正式运行（4 组串行）。
# 每组独立 calibration + 100ep evaluation。
#
# 失败语义（2026-09-15 修订，见 docs/logs.md §4）：
#   原实现用 `set -e` + 无每组日志，导致 2026-09-14 因共享模块被编辑成
#   半成品（stat_manager.py SyntaxError）而**整条 sweep 静默死掉**，只留下
#   一行 `[RUN] <next>`，无法分辨失败点。
#   现改为：每组独立 tee 到 `<out>/run.log`；失败时记录 `[FAIL]` 但**继续**
#   跑后续组；全部结束后若有失败则以非零码退出（fail-loud, no cascading）。
set -uo pipefail   # 故意不用 -e：单组失败不应连坐后续组

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
TASK="$REPO_ROOT/experiments/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8"

cd "$REPO_ROOT"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=2
# conda run 默认捕获子进程输出并块缓冲：会导致 (a) 运行中 <out>/run.log 一直为 0
# 字节直到结束才落盘，(b) stderr(traceback) 与 stdout 错序，难以定位失败点。
# --no-capture-output 让输出直通，配合 tee 得到实时日志。
export PYTHONUNBUFFERED=1
CONDA_RUN=(conda run --no-capture-output -n smolvla_eval)

configs=(
  g6a_all_fp8_control
  g6b_vlm_attn_w4_expert_fp8
  g6c_vlm_mlp_w4_expert_fp8
  g6d_vlm_all_w4_expert_fp8
)

failed=()

for name in "${configs[@]}"; do
  cfg="$TASK/configs/${name}.yaml"
  out="$REPO_ROOT/outputs/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/${name}"
  mkdir -p "$out"

  if [[ -f "$out/result.json" ]]; then
    echo "[SKIP] $name already complete"
    continue
  fi

  echo "[RUN] $name  ($(date '+%F %T'))"
  if "${CONDA_RUN[@]}" python main.py --config "$cfg" --skip-calibration 2>&1 | tee "$out/run.log"; then
    echo "[DONE] $name  ($(date '+%F %T'))"
  else
    echo "[FAIL] $name  ($(date '+%F %T')) — 见 $out/run.log"
    failed+=("$name")
  fi
done

if (( ${#failed[@]} > 0 )); then
  echo "G6 Goal run finished WITH FAILURES: ${failed[*]}"
  exit 1
fi

echo "G6 Goal run finished."
