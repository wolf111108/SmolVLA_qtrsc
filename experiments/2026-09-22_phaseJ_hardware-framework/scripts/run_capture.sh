#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$root"
case "${1:-}" in
  raw) config=hw0_raw; extra=() ;;
  vlin) config=hw1_vlin_trace; extra=(--skip-calibration) ;;
  *) echo "Usage: bash $0 raw|vlin (activate smolvla_eval first)" >&2; exit 2 ;;
esac
python -c 'import torch; print("torch", torch.__version__)'
python -m unittest discover -s tests -p test_hardware.py -v
python main.py --config "experiments/2026-09-22_phaseJ_hardware-framework/configs/${config}.yaml" "${extra[@]}"
