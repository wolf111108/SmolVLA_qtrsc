#!/usr/bin/env python
"""Sparsity quick scan aggregation (README_EXPERIMENT.md §12-§13).

Reads each config's `sparsity/module_sparsity.csv` + `sparsity/weight_sparsity_static.csv`
and produces:

  quick_sparsity_summary.csv     — Primary 表（Config × Stage，4 个指标）
  quick_sparsity_by_role.csv     — Secondary 表（按 tensor_role / weight operator）

Aggregation rule (§13): NEVER average per-row ratios; always sum numerator and
denominator across rows first, then divide.

Stage definition (§1) — prefill/denoise and vlm/expert must not be merged:
  PREFILL_VLM    : phase == "prefill" AND component == "vlm"
  DENOISE_EXPERT : phase == "denoise" AND component == "expert"

Native counters only for runtime metrics (§2.1): PoT/outlier protected FP
sidepaths manufacture artificial zeros on the normal quant path, so `reported`
would understate sparsity.

Static weights are attributed by component (vlm -> PREFILL_VLM,
expert -> DENOISE_EXPERT); this is a deployment/workload attribution, not a
statement that weights vary over time.

Bit-metric caveat (§3): INT family uses a sign-aware sparse-bit metric; FP8
E4M3 uses the 4-bit significand zero-bit ratio. These are NOT the same encoding
definition — compare element ratios across formats, but not the bit ratios.

Usage:
    python summarize_quick_sparsity.py [--outputs <dir>]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXP_NAME = "2026-09-15_sparsity-ratio-quickscan"

# config dir name -> display label (Primary 表顺序)
CONFIGS = [
    ("q0_int8", "INT8"),
    ("q1_int16", "INT16"),
    ("q2_fp8w4_po2", "FP8W4 PoT"),
    ("q3_fp8_po2", "FP8 PoT"),
]
STAGES = [("PREFILL_VLM", "VLM prefill"), ("DENOISE_EXPERT", "Expert denoise")]

# fp/bf kinds use the significand zero-bit metric; int uses sign-aware bits.
FP_KINDS = {"e4m3", "e5m2", "e2m1", "fp8_e4m3", "fp8_e5m2"}


def _read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row, key):
    try:
        return float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _ratio(num, den):
    return (num / den) if den > 0 else 0.0


def _stage_of_runtime(row):
    """Map a module_sparsity row to a headline stage, or None."""
    phase = (row.get("phase") or "").strip()
    comp = (row.get("component") or "").strip()
    if phase == "prefill" and comp == "vlm":
        return "PREFILL_VLM"
    if phase == "denoise" and comp == "expert":
        return "DENOISE_EXPERT"
    return None


def _stage_of_weight(row):
    comp = (row.get("component") or "").strip()
    if comp == "vlm":
        return "PREFILL_VLM"
    if comp == "expert":
        return "DENOISE_EXPERT"
    return None


def collect(config_dir, out_root):
    """Return (runtime_stage_agg, weight_stage_agg, gate_report)."""
    sp = os.path.join(out_root, config_dir, "sparsity")
    mod_path = os.path.join(sp, "module_sparsity.csv")
    w_path = os.path.join(sp, "weight_sparsity_static.csv")

    mod_rows = _read_csv(mod_path)
    w_rows = _read_csv(w_path)

    # ---- runtime：元素/bit 级 native，先求和再相除 ----
    rt = defaultdict(lambda: {"ze": 0.0, "te": 0.0, "zb": 0.0, "tb": 0.0})
    # ---- secondary：按 tensor_role 细分 ----
    rt_role = defaultdict(lambda: {"ze": 0.0, "te": 0.0, "zb": 0.0, "tb": 0.0})
    # ---- Gate 校验用 ----
    phases, flow_steps, components, roles, akinds = set(), set(), set(), set(), set()
    neg = 0

    for r in mod_rows:
        st = _stage_of_runtime(r)
        ze, te = _f(r, "zero_elements_native"), _f(r, "total_elements_native")
        zb, tb = _f(r, "sparse_bits_native"), _f(r, "total_bits_native")
        if ze < 0 or te < 0 or zb < 0 or tb < 0:
            neg += 1
        role = (r.get("tensor_role") or "?").strip()
        phases.add((r.get("phase") or "").strip())
        flow_steps.add((r.get("flow_step") or "").strip())
        components.add((r.get("component") or "").strip())
        roles.add(role)
        akinds.add((r.get("attention_kind") or "unknown").strip())
        if st is None:
            continue
        for acc in (rt[st], rt_role[(st, role)]):
            acc["ze"] += ze
            acc["te"] += te
            acc["zb"] += zb
            acc["tb"] += tb

    # ---- static weight：按 component 归属到 stage ----
    wt = defaultdict(lambda: {"ze": 0.0, "te": 0.0, "zb": 0.0, "tb": 0.0})
    wt_role = defaultdict(lambda: {"ze": 0.0, "te": 0.0, "zb": 0.0, "tb": 0.0})
    for r in w_rows:
        st = _stage_of_weight(r)
        if st is None:
            continue
        ze, te = _f(r, "zero_elements"), _f(r, "total_elements")
        zb, tb = _f(r, "sparse_bits"), _f(r, "total_bits")
        op = (r.get("operator") or "?").strip()
        spec = (r.get("weight_spec") or "?").strip()
        if ze < 0 or te < 0 or zb < 0 or tb < 0:
            neg += 1
        for acc in (wt[st], wt_role[(st, f"weight:{op}[{spec}]")]):
            acc["ze"] += ze
            acc["te"] += te
            acc["zb"] += zb
            acc["tb"] += tb

    gate = {
        "n_module_rows": len(mod_rows),
        "n_weight_rows": len(w_rows),
        "phases": sorted(p for p in phases if p),
        "flow_steps": sorted((f for f in flow_steps if f), key=lambda x: int(x) if x.lstrip("-").isdigit() else 0),
        "components": sorted(c for c in components if c),
        "roles": sorted(r for r in roles if r),
        "attention_kinds": sorted(k for k in akinds if k),
        "negative_counter_rows": neg,
        "has_summary": bool(mod_rows) and bool(w_rows),
    }
    return rt, rt_role, wt, wt_role, gate


def check_gate(cfg_dir, gate):
    """README §11 output gate. Returns list of (level, message)."""
    msgs = []
    if not gate["has_summary"]:
        msgs.append(("FAIL", "缺少 module_sparsity.csv 或 weight_sparsity_static.csv"))
        return msgs

    if gate["negative_counter_rows"]:
        msgs.append(("FAIL", f"native counters 出现负值（{gate['negative_counter_rows']} 行）"))
    if "prefill" in gate["phases"] and "denoise" in gate["phases"]:
        msgs.append(("OK", f"phase 覆盖 prefill+denoise"))
    else:
        msgs.append(("FAIL", f"phase 缺失：{gate['phases']}"))

    fs = set(gate["flow_steps"])
    want = {str(i) for i in range(10)}
    if want.issubset(fs):
        msgs.append(("OK", "denoise flow_step 覆盖 0..9"))
    else:
        msgs.append(("WARN", f"flow_step 未覆盖 0..9：缺 {sorted(want - fs)}"))

    comps = set(gate["components"])
    if {"vlm", "expert"}.issubset(comps):
        msgs.append(("OK", "component 覆盖 vlm+expert"))
    else:
        msgs.append(("FAIL", f"component 缺失：{sorted(comps)}"))

    roles = set(gate["roles"])
    if {"activation", "output"}.issubset(roles) and {"A", "B", "O"}.issubset(roles):
        msgs.append(("OK", "tensor_role 覆盖 Linear activation/output + MatMul A/B/O"))
    else:
        msgs.append(("WARN", f"tensor_role 不全：{sorted(roles)}"))
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outputs",
        default=os.path.join(REPO_ROOT, "outputs", EXP_NAME),
        help="quick scan 输出根目录（默认为 outputs/<EXP_NAME>）",
    )
    args = ap.parse_args()
    out_root = os.path.abspath(args.outputs)

    if not os.path.isdir(out_root):
        print(f"输出目录不存在: {out_root}")
        return 1

    primary_rows = []
    role_rows = []
    all_fail = False

    for cfg_dir, label in CONFIGS:
        rt, rt_role, wt, wt_role, gate = collect(cfg_dir, out_root)

        print("=" * 78)
        print(f"[{label}]  {cfg_dir}")
        print("=" * 78)
        for level, msg in check_gate(cfg_dir, gate):
            print(f"  [{level}] {msg}")
        if not gate["has_summary"]:
            all_fail = True
            print()
            continue
        print(
            f"  module rows={gate['n_module_rows']}  weight rows={gate['n_weight_rows']}"
            f"  attention_kind={gate['attention_kinds']}"
        )

        for st, st_label in STAGES:
            a = rt.get(st)
            w = wt.get(st)
            if not a or not w:
                print(f"  {st_label}: 无数据")
                continue
            r_e = _ratio(a["ze"], a["te"])
            r_b = _ratio(a["zb"], a["tb"])
            w_e = _ratio(w["ze"], w["te"])
            w_b = _ratio(w["zb"], w["tb"])
            primary_rows.append(
                {
                    "config": label,
                    "stage": st_label,
                    "runtime_element_sparsity": f"{r_e:.6f}",
                    "runtime_bit_sparsity": f"{r_b:.6f}",
                    "weight_element_sparsity": f"{w_e:.6f}",
                    "weight_bit_sparsity": f"{w_b:.6f}",
                }
            )
            print(
                f"  {st_label:16s} elem={r_e * 100:6.2f}%  bit={r_b * 100:6.2f}%"
                f"  | w_elem={w_e * 100:6.2f}%  w_bit={w_b * 100:6.2f}%"
            )

        # secondary：role 细分
        for st, st_label in STAGES:
            merged = defaultdict(lambda: {"ze": 0.0, "te": 0.0, "zb": 0.0, "tb": 0.0})
            for (s, role), acc in rt_role.items():
                if s != st:
                    continue
                for k in ("ze", "te", "zb", "tb"):
                    merged[role][k] += acc[k]
            for (s, role), acc in wt_role.items():
                if s != st:
                    continue
                for k in ("ze", "te", "zb", "tb"):
                    merged[role][k] += acc[k]
            for role in sorted(merged):
                a = merged[role]
                role_rows.append(
                    {
                        "config": label,
                        "stage": st_label,
                        "role": role,
                        "element_sparsity": f"{_ratio(a['ze'], a['te']):.6f}",
                        "bit_sparsity": f"{_ratio(a['zb'], a['tb']):.6f}",
                        "total_elements": f"{a['te']:.0f}",
                        "total_bits": f"{a['tb']:.0f}",
                    }
                )
        print()

    summary_path = os.path.join(out_root, "quick_sparsity_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "stage",
                "runtime_element_sparsity",
                "runtime_bit_sparsity",
                "weight_element_sparsity",
                "weight_bit_sparsity",
            ],
        )
        w.writeheader()
        w.writerows(primary_rows)
    print(f"Primary 表 → {summary_path}  ({len(primary_rows)} 行)")

    role_path = os.path.join(out_root, "quick_sparsity_by_role.csv")
    with open(role_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "stage",
                "role",
                "element_sparsity",
                "bit_sparsity",
                "total_elements",
                "total_bits",
            ],
        )
        w.writeheader()
        w.writerows(role_rows)
    print(f"Secondary 表 → {role_path}  ({len(role_rows)} 行)")

    print()
    print("注意（README §3）：INT 的 bit sparsity 是 sign-aware sparse-bit；")
    print("FP8 是 E4M3 4-bit significand zero-bit。二者不是同一编码定义，")
    print("不可把二者的差值直接写成同一指标的 pp 差。")

    return 1 if all_fail else 0


if __name__ == "__main__":
    sys.exit(main())
