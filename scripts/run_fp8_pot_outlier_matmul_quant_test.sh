#!/usr/bin/env bash
# PoT-FP8 + Outlier Protection（Linear + MatMul）：calibration + LIBERO eval + 审计
#
# 与 run_fp8_pot_outlier_quant_test.sh 同协议，唯一差异：
#   quantize_matmul=true（attention QK^T / PV 也走 pot_fp8_outlier 量化）
#
# 默认配置：
#   configs/experiments/smolvla_fp8_pot_outlier_matmul_quant_test_full.yaml
#
# 用法：
#   # 1) 首次：重新校准 + evaluation + 收集结果
#   bash scripts/run_fp8_pot_outlier_matmul_quant_test.sh
#
#   # 2) 复用已有 PoT scales，只做 evaluation + 收集结果
#   bash scripts/run_fp8_pot_outlier_matmul_quant_test.sh --reuse
#
#   # 3) 只重新汇总已有 result.json / scale 文件，不运行模型
#   bash scripts/run_fp8_pot_outlier_matmul_quant_test.sh --summary-only
#
# 输出（写入 YAML 的 output_dir）：
#   result.json / config.yaml / run_*.log      main.py 产物
#   pot_scale_audit.csv / pot_scale_audit.json PoT scale 审计（含 A/B/O）
#   per_task_sr.csv / summary.json / summary.md 结果汇总
#
# 说明：
#   - matmul 层的 scale 文件名是 {qk,pv}_matmul_{A,B,O}_scale_0.p，
#     audit_scales 已同时识别 linear 的 a/w/o 与 matmul 的 A/B/O；
#   - matmul 量化器是全模型共享一对（模型级 attention 接口），
#     scale 对所有 attention 层的激活取 absmax 聚合。
#
set -euo pipefail

CONFIG="configs/experiments/smolvla_fp8_pot_outlier_matmul_quant_test_full.yaml"
FP_BASELINE="93.8"
REUSE=0
SUMMARY_ONLY=0

usage() {
    sed -n '2,30p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reuse)
            REUSE=1
            shift
            ;;
        --summary-only)
            SUMMARY_ONLY=1
            shift
            ;;
        --config)
            [[ $# -ge 2 ]] || { echo "ERROR: --config requires a path" >&2; exit 2; }
            CONFIG="$2"
            shift 2
            ;;
        --baseline)
            [[ $# -ge 2 ]] || { echo "ERROR: --baseline requires a number" >&2; exit 2; }
            FP_BASELINE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Read authoritative settings from YAML.
# ---------------------------------------------------------------------------
mapfile -t CFG_VALUES < <(
python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
q = cfg.get("quantization", {})
e = cfg.get("evaluation", {})
env = e.get("env", {})
m = cfg.get("model", {})
print(cfg.get("output_dir", "outputs/default_run"))
print(q.get("scale_dir", ""))
print(q.get("method", q.get("scale_method", "per_tensor")))
print(q.get("outlier_ratio", 0.0))
print(env.get("task", ""))
print(e.get("n_episodes", ""))
print(m.get("path", ""))
print(q.get("quantize_matmul", False))
PY
)

OUT_DIR="${CFG_VALUES[0]}"
SCALE_DIR="${CFG_VALUES[1]}"
METHOD="${CFG_VALUES[2]}"
OUTLIER_RATIO="${CFG_VALUES[3]}"
EVAL_TASK="${CFG_VALUES[4]}"
N_EPISODES="${CFG_VALUES[5]}"
MODEL_PATH="${CFG_VALUES[6]}"
QUANTIZE_MATMUL="${CFG_VALUES[7]}"

if [[ -z "$SCALE_DIR" ]]; then
    echo "ERROR: quantization.scale_dir is empty in $CONFIG" >&2
    exit 2
fi

if [[ "$METHOD" != "pot_fp8_outlier" ]]; then
    echo "ERROR: expected quantization.method=pot_fp8_outlier, got: $METHOD" >&2
    exit 2
fi

if [[ "$QUANTIZE_MATMUL" != "True" ]]; then
    echo "ERROR: this runner expects quantization.quantize_matmul=true, got: $QUANTIZE_MATMUL" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/run_${TIMESTAMP}.log"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "================================================================================"
echo "              PoT-FP8 + Outlier Protection Test (Linear + MatMul)"
echo "================================================================================"
echo "config          : $CONFIG"
echo "model           : $MODEL_PATH"
echo "method          : $METHOD"
echo "outlier_ratio   : $OUTLIER_RATIO"
echo "scale_dir       : $SCALE_DIR"
echo "output_dir      : $OUT_DIR"
echo "eval task       : $EVAL_TASK"
echo "eval episodes   : $N_EPISODES"
echo "quantize_matmul : $QUANTIZE_MATMUL"
echo "FP baseline     : ${FP_BASELINE}%"
echo "reuse scales    : $REUSE"
echo "summary only    : $SUMMARY_ONLY"
echo "MUJOCO_GL       : $MUJOCO_GL"
echo "conda env       : ${CONDA_DEFAULT_ENV:-<none>}"
echo "================================================================================"

if [[ "${CONDA_DEFAULT_ENV:-}" != "smolvla_eval" ]]; then
    echo "WARNING: expected conda env 'smolvla_eval', current='${CONDA_DEFAULT_ENV:-<none>}'" >&2
fi

# ---------------------------------------------------------------------------
# Preflight: linear + matmul 方法都已注册，YAML 全 FP8，matmul 已启用。
# ---------------------------------------------------------------------------
python - "$CONFIG" <<'PY'
import sys, yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
q = cfg["quantization"]

from vla_tcs2.quant.scale_methods import get_scale_method, get_matmul_scale_method
from vla_tcs2.quant.quant_methods import get_quant_method, get_matmul_quant_method

name = q.get("method", q.get("scale_method"))
get_scale_method(name)
get_quant_method(name)
get_matmul_scale_method(name)
get_matmul_quant_method(name)

if not q.get("quantize_matmul", False):
    raise SystemExit("ERROR: quantization.quantize_matmul must be true for this runner.")

linear_names = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
matmul_names = ("qk_matmul", "pv_matmul")
valid_fp8 = {"e4m3", "e4m3fn", "e5m2"}

bad = []
for layer in linear_names:
    d = q.get(layer, {})
    for field in ("a_bit", "w_bit", "o_bit"):
        value = str(d.get(field, "")).lower()
        # Allow explicit fp16/bf16 passthrough for future A8W8O16 ablations.
        if value not in valid_fp8 | {"fp16", "bf16"}:
            bad.append(f"{layer}.{field}={value!r}")

for layer in matmul_names:
    d = q.get(layer, {})
    for field in ("A_bit", "B_bit", "O_bit"):
        value = str(d.get(field, "")).lower()
        if value not in valid_fp8 | {"fp16", "bf16"}:
            bad.append(f"{layer}.{field}={value!r}")

if bad:
    raise SystemExit(
        "Invalid precision for pot_fp8_outlier:\n  " + "\n  ".join(bad)
    )

print("Preflight OK: linear + matmul methods registered, all specs FP8/passthrough.")
PY

# ---------------------------------------------------------------------------
# PoT scale audit (shared with the linear-only runner; recognizes both
# linear a/w/o and matmul A/B/O scale files).
# ---------------------------------------------------------------------------
audit_scales() {
python - "$SCALE_DIR" "$OUT_DIR" <<'PY'
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

scale_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

if not scale_dir.exists():
    raise SystemExit(f"Scale directory not found: {scale_dir}")

files = sorted(scale_dir.glob("*.p"))
if not files:
    raise SystemExit(f"No scale pickle files found under: {scale_dir}")

records = []
stats = defaultdict(lambda: {"count": 0, "exponents": []})
invalid = []

def scalar_value(obj):
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.numel() != 1:
                raise TypeError(f"non-scalar tensor shape={tuple(obj.shape)}")
            return float(obj.item())
    except Exception:
        pass
    return float(obj)

for path in files:
    try:
        with path.open("rb") as f:
            value = scalar_value(pickle.load(f))
    except Exception as exc:
        invalid.append((path.name, f"unreadable/non-scalar: {exc}"))
        continue

    if "_a_scale_" in path.name:
        scale_type = "a_scale"
    elif "_w_scale_" in path.name:
        scale_type = "w_scale"
    elif "_o_scale_" in path.name:
        scale_type = "o_scale"
    elif "_A_scale_" in path.name:
        scale_type = "A_scale"
    elif "_B_scale_" in path.name:
        scale_type = "B_scale"
    elif "_O_scale_" in path.name:
        scale_type = "O_scale"
    else:
        scale_type = "unknown"

    if not math.isfinite(value) or value <= 0:
        is_pot = False
        exponent = None
        nearest = None
    else:
        raw_exp = math.log2(value)
        exponent = int(round(raw_exp))
        nearest = math.ldexp(1.0, exponent)
        # Values written by math.ldexp are exact binary powers of two.
        # Keep a tiny tolerance so old pickle/python round-trips are harmless.
        is_pot = abs(value - nearest) <= max(1e-15, abs(value) * 1e-12)

    rec = {
        "file": path.name,
        "scale_type": scale_type,
        "scale": value,
        "exponent": exponent,
        "is_power_of_two": is_pot,
    }
    records.append(rec)

    stats[scale_type]["count"] += 1
    if exponent is not None and is_pot:
        stats[scale_type]["exponents"].append(exponent)

    if not is_pot:
        invalid.append((path.name, f"value={value!r}"))

csv_path = out_dir / "pot_scale_audit.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file", "scale_type", "scale", "exponent", "is_power_of_two"],
    )
    writer.writeheader()
    writer.writerows(records)

summary_by_type = {}
for typ, d in sorted(stats.items()):
    exps = d["exponents"]
    summary_by_type[typ] = {
        "count": d["count"],
        "pot_count": len(exps),
        "min_exponent": min(exps) if exps else None,
        "max_exponent": max(exps) if exps else None,
        "unique_exponents": sorted(set(exps)),
    }

audit = {
    "scale_dir": str(scale_dir),
    "file_count": len(files),
    "valid_power_of_two_count": sum(r["is_power_of_two"] for r in records),
    "invalid_count": len(invalid),
    "all_power_of_two": len(invalid) == 0,
    "by_type": summary_by_type,
    "invalid": [{"file": f, "reason": r} for f, r in invalid],
}
json_path = out_dir / "pot_scale_audit.json"
json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

print("\n==================== PoT SCALE AUDIT ====================")
print(f"scale files : {len(files)}")
print(f"valid PoT   : {audit['valid_power_of_two_count']}")
print(f"invalid     : {audit['invalid_count']}")
for typ, d in summary_by_type.items():
    print(
        f"{typ:8s}: count={d['count']:4d}, "
        f"k_range=[{d['min_exponent']}, {d['max_exponent']}], "
        f"unique_k={d['unique_exponents']}"
    )
print(f"csv  : {csv_path}")
print(f"json : {json_path}")

if invalid:
    print("\nERROR: non-PoT scales detected:")
    for name, reason in invalid[:20]:
        print(f"  {name}: {reason}")
    if len(invalid) > 20:
        print(f"  ...and {len(invalid)-20} more")
    raise SystemExit(3)

print("PoT scale audit: PASS")
PY
}

# If reusing scale files, reject arbitrary-scale files *before* spending eval time.
if [[ "$REUSE" -eq 1 && "$SUMMARY_ONLY" -eq 0 ]]; then
    echo
    echo ">>> Pre-auditing reused scales ..."
    audit_scales
fi

# ---------------------------------------------------------------------------
# Run pipeline.
# ---------------------------------------------------------------------------
if [[ "$SUMMARY_ONLY" -eq 0 ]]; then
    CMD=(python main.py --config "$CONFIG")
    if [[ "$REUSE" -eq 1 ]]; then
        CMD+=(--skip-calibration)
    fi

    echo
    echo ">>> Running:"
    printf ' %q' "${CMD[@]}"
    echo
    echo ">>> Full log: $LOG_FILE"
    echo

    set +e
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    RC=${PIPESTATUS[0]}
    set -e

    if [[ "$RC" -ne 0 ]]; then
        echo "ERROR: main.py failed with exit code $RC" >&2
        echo "Log preserved at: $LOG_FILE" >&2
        exit "$RC"
    fi
else
    echo
    echo ">>> --summary-only: model execution skipped."
fi

# ---------------------------------------------------------------------------
# Post-run PoT scale audit.
# ---------------------------------------------------------------------------
echo
echo ">>> Auditing calibrated/reused PoT scales ..."
audit_scales

RESULT_JSON="$OUT_DIR/result.json"
if [[ ! -f "$RESULT_JSON" ]]; then
    echo "ERROR: result.json not found: $RESULT_JSON" >&2
    exit 4
fi

# ---------------------------------------------------------------------------
# Collect SR + per-task results + reproducibility metadata.
# ---------------------------------------------------------------------------
python - "$CONFIG" "$RESULT_JSON" "$OUT_DIR" "$SCALE_DIR" "$FP_BASELINE" <<'PY'
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
scale_dir = Path(sys.argv[4])
fp_baseline = float(sys.argv[5])

cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
result = json.loads(result_path.read_text(encoding="utf-8"))
scale_audit_path = out_dir / "pot_scale_audit.json"
scale_audit = json.loads(scale_audit_path.read_text(encoding="utf-8"))

qcfg = cfg.get("quantization", {})
ecfg = cfg.get("evaluation", {})
envcfg = ecfg.get("env", {})
mcfg = cfg.get("model", {})

def task_label(task, idx):
    for key in ("task_id", "task", "task_name", "name", "id"):
        if key in task and task[key] is not None:
            return str(task[key])
    return str(idx)

rows = []
total_ok = 0
total_n = 0

for idx, task in enumerate(result.get("per_task", [])):
    metrics = task.get("metrics", {})
    successes = metrics.get("successes", [])
    if hasattr(successes, "tolist"):
        successes = successes.tolist()
    successes = list(successes)
    ok = sum(bool(x) for x in successes)
    n = len(successes)
    sr = 100.0 * ok / n if n else float("nan")
    total_ok += ok
    total_n += n
    rows.append(
        {
            "task_index": idx,
            "task": task_label(task, idx),
            "successes": ok,
            "episodes": n,
            "success_rate_percent": sr,
        }
    )

if total_n == 0:
    raise SystemExit("No success data found in result['per_task'][*]['metrics']['successes'].")

sr = 100.0 * total_ok / total_n
delta = sr - fp_baseline

per_task_csv = out_dir / "per_task_sr.csv"
with per_task_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "task_index",
            "task",
            "successes",
            "episodes",
            "success_rate_percent",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

def cmd_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

git_commit = cmd_output(["git", "rev-parse", "HEAD"])
git_status = cmd_output(["git", "status", "--porcelain"])

software = {
    "python": platform.python_version(),
    "git_commit": git_commit,
    "git_dirty": bool(git_status),
    "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
}
try:
    import torch
    software.update(
        {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )
except Exception as exc:
    software["torch_error"] = str(exc)

try:
    import mujoco
    software["mujoco"] = mujoco.__version__
except Exception as exc:
    software["mujoco_error"] = str(exc)

summary = {
    "timestamp": datetime.now().astimezone().isoformat(),
    "experiment": "PoT-FP8 + Outlier Protection (Linear + MatMul)",
    "config": str(config_path),
    "output_dir": str(out_dir),
    "scale_dir": str(scale_dir),
    "model": mcfg.get("path"),
    "quantization": {
        "method": qcfg.get("method", qcfg.get("scale_method")),
        "outlier_ratio": qcfg.get("outlier_ratio"),
        "quantize_matmul": qcfg.get("quantize_matmul", False),
        "q_proj": qcfg.get("q_proj", {}),
        "k_proj": qcfg.get("k_proj", {}),
        "v_proj": qcfg.get("v_proj", {}),
        "o_proj": qcfg.get("o_proj", {}),
        "gate_proj": qcfg.get("gate_proj", {}),
        "up_proj": qcfg.get("up_proj", {}),
        "down_proj": qcfg.get("down_proj", {}),
        "qk_matmul": qcfg.get("qk_matmul", {}),
        "pv_matmul": qcfg.get("pv_matmul", {}),
    },
    "evaluation": {
        "suite": envcfg.get("task"),
        "configured_n_episodes": ecfg.get("n_episodes"),
        "seed": ecfg.get("seed"),
        "batch_size": ecfg.get("batch_size"),
        "successes": total_ok,
        "episodes_observed": total_n,
        "success_rate_percent": sr,
        "fp_baseline_percent": fp_baseline,
        "delta_vs_fp_pp": delta,
        "per_task": rows,
    },
    "pot_scale_audit": scale_audit,
    "software": software,
}

summary_json = out_dir / "summary.json"
summary_json.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)

def fnum(x):
    return "n/a" if x is None else str(x)

md = []
md.append("# PoT-FP8 + Outlier Protection (Linear + MatMul) — Result Summary")
md.append("")
md.append("## Configuration")
md.append("")
md.append(f"- Model: `{mcfg.get('path')}`")
md.append(f"- Method: `{summary['quantization']['method']}`")
md.append(f"- Outlier ratio: `{qcfg.get('outlier_ratio')}`")
md.append(f"- LIBERO suite: `{envcfg.get('task')}`")
md.append(f"- Seed: `{ecfg.get('seed')}`")
md.append(f"- MatMul quantization: `{qcfg.get('quantize_matmul', False)}`")
md.append("")
md.append("## Success Rate")
md.append("")
md.append(f"- PoT-FP8 + OP (linear+matmul): **{total_ok}/{total_n} = {sr:.1f}%**")
md.append(f"- FP baseline: **{fp_baseline:.1f}%**")
md.append(f"- Delta vs FP: **{delta:+.1f} pp**")
md.append("")
md.append("| Task | Success | Episodes | SR |")
md.append("|---:|---:|---:|---:|")
for r in rows:
    md.append(
        f"| {r['task']} | {r['successes']} | {r['episodes']} | "
        f"{r['success_rate_percent']:.1f}% |"
    )

md.append("")
md.append("## PoT Scale Audit")
md.append("")
md.append(
    f"- All scales are power-of-two: **{scale_audit['all_power_of_two']}**"
)
md.append(f"- Scale files: **{scale_audit['file_count']}**")
md.append(f"- Invalid scales: **{scale_audit['invalid_count']}**")
md.append("")
md.append("| Scale type | Count | Min k | Max k | Unique k |")
md.append("|---|---:|---:|---:|---|")
for typ, d in scale_audit.get("by_type", {}).items():
    md.append(
        f"| {typ} | {d['count']} | {fnum(d['min_exponent'])} | "
        f"{fnum(d['max_exponent'])} | `{d['unique_exponents']}` |"
    )

md.append("")
md.append("## Reproducibility")
md.append("")
md.append(f"- Git commit: `{software.get('git_commit')}`")
md.append(f"- Git dirty: `{software.get('git_dirty')}`")
md.append(f"- Conda env: `{software.get('conda_env')}`")
md.append(f"- Python: `{software.get('python')}`")
md.append(f"- PyTorch: `{software.get('torch')}`")
md.append(f"- CUDA runtime: `{software.get('cuda_runtime')}`")
md.append(f"- GPU: `{software.get('gpu')}`")
md.append(f"- MuJoCo: `{software.get('mujoco')}`")
md.append(f"- MUJOCO_GL: `{software.get('MUJOCO_GL')}`")
md.append("")
md.append("Artifacts:")
md.append(f"- `{result_path}`")
md.append(f"- `{per_task_csv}`")
md.append(f"- `{scale_audit_path}`")
md.append(f"- `{summary_json}`")

summary_md = out_dir / "summary.md"
summary_md.write_text("\n".join(md) + "\n", encoding="utf-8")

print("\n==================== FINAL RESULTS ====================")
print(f"PoT-FP8 L+M       : {total_ok}/{total_n} = {sr:.1f}%")
print(f"FP baseline       : {fp_baseline:.1f}%")
print(f"delta vs FP       : {delta:+.1f} pp")
print(f"PoT scale audit   : {'PASS' if scale_audit['all_power_of_two'] else 'FAIL'}")
print()
print("Per-task:")
for r in rows:
    print(
        f"  task {r['task']:>4}: "
        f"{r['successes']}/{r['episodes']} = "
        f"{r['success_rate_percent']:.1f}%"
    )
print()
print(f"summary JSON : {summary_json}")
print(f"summary MD   : {summary_md}")
print(f"per-task CSV : {per_task_csv}")
print("=======================================================")
PY

echo
echo "DONE."
echo "Result summary: $OUT_DIR/summary.md"
