#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/lerobot_current/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export VLA_TENSOR_DUMP_DIR=""
exec "${PYTHON:-python}" experiments/2026-11-10_phaseN_bg-p-sweep/scripts/run.py "$@"
