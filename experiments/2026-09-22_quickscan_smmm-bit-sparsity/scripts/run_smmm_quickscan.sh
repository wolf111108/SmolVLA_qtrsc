#!/usr/bin/env bash
# SMMM bit-sparsity quickscan: 1 config (FP8 PoT), task0 x 1ep, eval-only.
#
# Re-measures runtime/weight sparsity under the NEW S|MMM bit metric
# (sign + mantissa, no hidden leading 1) in stat_manager.
#
# Scale isolation: copy the canonical q3_fp8_po2 scales (2026-09-15
# quickscan) into a fresh per-experiment scale dir, sha256-verify the
# copy, and force --skip-calibration (triple protection).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-smolvla_eval}"
CFG="experiments/2026-09-22_quickscan_smmm-bit-sparsity/configs/s4_fp8_smmm.yaml"
OUT="outputs/2026-09-22_quickscan_smmm-bit-sparsity/s4_fp8_smmm"

BASE_SCALE_DIR="${BASE_SCALE_DIR:-outputs/2026-09-15_sparsity-ratio-quickscan/q3_fp8_po2/sparsity}"
# The q3 run did not persist its scale dir; fall back to the canonical G6-A copy.
ALT_SCALE_DIR="scales/2026-09-15_sparsity-ratio-quickscan/q3_fp8_po2"
TARGET_SCALE_DIR="scales/2026-09-22_quickscan_smmm-bit-sparsity/s4_fp8_smmm"

if command -v conda >/dev/null 2>&1; then
  RUN=(conda run --no-capture-output -n "$CONDA_ENV")
else
  echo "ERROR: conda not found." >&2
  exit 2
fi

echo "[Gate 0] unit tests (incl. T10 SMMM encoding)"
"${RUN[@]}" python -m pytest tests/test_sparsity_accounting.py -q

echo
echo "[Gate 1] prepare isolated scale directory"
SRC=""
if [[ -d "$ALT_SCALE_DIR" ]]; then
  SRC="$ALT_SCALE_DIR"
elif [[ -d "$BASE_SCALE_DIR" ]]; then
  SRC="$BASE_SCALE_DIR"
else
  echo "ERROR: no canonical FP8 PoT scale source found." >&2
  echo "  tried: $ALT_SCALE_DIR" >&2
  echo "        $BASE_SCALE_DIR" >&2
  echo "Set BASE_SCALE_DIR to the valid q3/G6-A FP8 scale directory." >&2
  exit 2
fi
rm -rf "$TARGET_SCALE_DIR"
mkdir -p "$TARGET_SCALE_DIR"
cp -a "$SRC"/. "$TARGET_SCALE_DIR"/
BASE_COUNT="$(find "$SRC" -maxdepth 1 -type f -name '*.p' | wc -l | tr -d ' ')"
COPY_COUNT="$(find "$TARGET_SCALE_DIR" -maxdepth 1 -type f -name '*.p' | wc -l | tr -d ' ')"
echo "scale source      : $SRC"
echo "canonical files   : $BASE_COUNT"
echo "copied files      : $COPY_COUNT"
if [[ "$BASE_COUNT" -ne "$COPY_COUNT" || "$BASE_COUNT" -eq 0 ]]; then
  echo "ERROR: scale copy mismatch." >&2
  exit 2
fi
(cd "$SRC" && find . -maxdepth 1 -type f -name '*.p' -exec sha256sum {} \; | sort) > /tmp/smmm_base.sha
(cd "$TARGET_SCALE_DIR" && find . -maxdepth 1 -type f -name '*.p' -exec sha256sum {} \; | sort) > /tmp/smmm_copy.sha
if ! diff -q /tmp/smmm_base.sha /tmp/smmm_copy.sha >/dev/null; then
  echo "ERROR: sha256 mismatch between canonical and copied scales." >&2
  exit 2
fi
echo "sha256 verify     : OK (bit-identical)"
cp /tmp/smmm_copy.sha "$OUT".scale_hashes.txt 2>/dev/null || \
  mkdir -p outputs/2026-09-22_quickscan_smmm-bit-sparsity && \
  cp /tmp/smmm_copy.sha outputs/2026-09-22_quickscan_smmm-bit-sparsity/s4_scale_hashes.txt

echo
echo "[Gate 2] eval-only run (task0 x 1ep, --skip-calibration)"
"${RUN[@]}" python main.py --config "$CFG" --skip-calibration

echo
echo "[Gate 3] summary"
"${RUN[@]}" python \
  experiments/2026-09-22_quickscan_smmm-bit-sparsity/scripts/summarize_smmm_sparsity.py \
  --sparsity-dir "$OUT/sparsity"

echo
echo "SMMM quickscan: PASS"
