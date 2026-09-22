#!/usr/bin/env bash
# Full Vision+VLM+Expert S|MMM sparsity characterization, task0 x 1 episode.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-smolvla_eval}"
CFG="$EXP_ROOT/configs/vs1_full_vlin_smmm_1ep.yaml"
OUT="outputs/2026-09-22_phaseI_vision-smmm-sparsity/vs1_full_vlin_smmm_1ep"

SRC_SCALE_DIR="${SRC_SCALE_DIR:-scales/2026-09-15_phaseI_vision-quantization/vlin_full_vision_linear_fp8}"
TARGET_SCALE_DIR="scales/2026-09-22_phaseI_vision-smmm-sparsity/vs1_full_vlin_smmm_1ep"
EXPECTED_SCALE_FILES=1080  # 864 legacy + 216 Vision (72 sites x A/W/O)

PARENT_AUDIT="experiments/2026-09-15_phaseI_vision-quantization/scripts/audit_full_vision_linear_calibration.py"
SUMMARIZER="$EXP_ROOT/scripts/summarize_vision_smmm_sparsity.py"

unset CUDA_VISIBLE_DEVICES || true

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found." >&2
  exit 2
fi
RUN=(conda run --no-capture-output -n "$CONDA_ENV")

echo "======================================================================"
echo " Full VLIN S|MMM sparsity — task0 x 1 episode"
echo "======================================================================"
echo "config : $CFG"
echo "output : $OUT"
echo

echo "[Gate 0] code/tests"
"${RUN[@]}" python -m py_compile "$SUMMARIZER"
"${RUN[@]}" python -m pytest tests/test_sparsity_accounting.py -q

echo
echo "[Gate 1] isolated VLIN scale copy"
if [[ ! -d "$SRC_SCALE_DIR" ]]; then
  echo "ERROR: source VLIN scales missing: $SRC_SCALE_DIR" >&2
  exit 2
fi
rm -rf "$TARGET_SCALE_DIR"
mkdir -p "$TARGET_SCALE_DIR"
cp -a "$SRC_SCALE_DIR"/. "$TARGET_SCALE_DIR"/

SRC_COUNT="$(find "$SRC_SCALE_DIR" -maxdepth 1 -type f -name '*.p' | wc -l | tr -d ' ')"
DST_COUNT="$(find "$TARGET_SCALE_DIR" -maxdepth 1 -type f -name '*.p' | wc -l | tr -d ' ')"
echo "source scale files : $SRC_COUNT"
echo "copied scale files : $DST_COUNT"
if [[ "$SRC_COUNT" -ne "$EXPECTED_SCALE_FILES" || "$DST_COUNT" -ne "$EXPECTED_SCALE_FILES" ]]; then
  echo "ERROR: expected exactly $EXPECTED_SCALE_FILES scale files." >&2
  exit 2
fi

src_hash="$(mktemp)"
dst_hash="$(mktemp)"
trap 'rm -f "$src_hash" "$dst_hash"' EXIT
(cd "$SRC_SCALE_DIR" && find . -maxdepth 1 -type f -name '*.p' -exec sha256sum {} \; | sort) > "$src_hash"
(cd "$TARGET_SCALE_DIR" && find . -maxdepth 1 -type f -name '*.p' -exec sha256sum {} \; | sort) > "$dst_hash"
if ! diff -q "$src_hash" "$dst_hash" >/dev/null; then
  echo "ERROR: scale sha256 mismatch." >&2
  exit 2
fi
echo "sha256              : bit-identical"

mkdir -p "$OUT"
cp "$dst_hash" "$OUT/scale_hashes.txt"

echo
echo "[Gate 2] routing + Vision scale coverage preflight"
"${RUN[@]}" python "$PARENT_AUDIT" --config "$CFG"

echo
echo "[Gate 3] task0 x 1 episode, eval-only, S|MMM collector"
"${RUN[@]}" python main.py --config "$CFG" --skip-calibration \
  2>&1 | tee "$OUT/run.log"

echo
echo "[Gate 4] primary artifact presence"
SP="$OUT/sparsity"
required=(
  "$SP/module_sparsity.csv"
  "$SP/weight_sparsity_static.csv"
  "$SP/quantization_manifest.csv"
)
for f in "${required[@]}"; do
  if [[ ! -s "$f" || "$(wc -l < "$f")" -le 1 ]]; then
    echo "ERROR: missing/header-only artifact: $f" >&2
    exit 2
  fi
  echo "  OK: $f ($(wc -l < "$f") lines)"
done

echo
echo "[Gate 5] Vision/VLM/Expert S|MMM summary"
"${RUN[@]}" python "$SUMMARIZER" --sparsity-dir "$SP"

echo
echo "======================================================================"
echo " PASS — full Vision S|MMM 1-episode characterization complete"
echo "======================================================================"
echo "summary: $OUT/vision_smmm_sparsity_summary.csv"
