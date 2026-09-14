#!/usr/bin/env python
"""Phase H — build the §14 result tables (experiment_setup.md §14) from raw CSVs.

Aggregation rule (experiment_setup.md §8): never average percentages, always
sum numerators / sum denominators.

Outputs GitHub-flavored markdown to stdout so it can be pasted into results.md.
Also prints the §13 convergence gate (|Δ| < 0.5 pp for H2 vs H1).

Usage:
    python build_results_tables.py --stage h2_30ep
    python build_results_tables.py --stage h2_30ep --compare h1_10ep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXP_NAME = "2026-09-13_phaseH_accuracy-preserving-sparsity"
CONFIGS = ["s0_fp8_all", "s1_expert_w4"]
LINEAR_OPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _read(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load(stage, cfg, fname, exclude_roles=()):
    root = os.path.join(REPO_ROOT, "outputs", EXP_NAME, stage, cfg)
    rows = []
    if not os.path.isdir(root):
        return rows
    for d in sorted(os.listdir(root)):
        for r in _read(os.path.join(root, d, "sparsity", fname)):
            if r.get("tensor_role") in exclude_roles:
                continue
            rows.append(r)
    return rows


def _sr(stage, cfg):
    """Episode-weighted SR across tasks (sum successes / sum episodes)."""
    root = os.path.join(REPO_ROOT, "outputs", EXP_NAME, stage, cfg)
    ok = tot = 0
    n_tasks = 0
    if not os.path.isdir(root):
        return None
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "eval_info.json")
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            o = json.load(f)["overall"]
        n = int(o["n_episodes"])
        ok += round(o["pc_success"] / 100.0 * n)
        tot += n
        n_tasks += 1
    return (ok, tot, n_tasks, 100.0 * ok / tot if tot else 0.0)


def _unit_component_key(r):
    """unit_sparsity.csv has no component/operator columns → derive from module_id."""
    comp, op = _parse_module_id(r["module_id"])
    if op in ("qk", "pv"):
        return (f"{comp}.{op}".upper(), r["phase"], r["tensor_role"])
    return (comp.upper(), r["phase"], r["tensor_role"])


def _ratio(rows, num, den):
    n = sum(float(r[num]) for r in rows)
    d = sum(float(r[den]) for r in rows)
    return (n / d if d else 0.0), n, d


def _pct(x):
    return f"{x * 100:.2f}%"


def _parse_module_id(module_id):
    """module_id → (component, operator).

    Examples:
        vlm.layers.0.self_attn.q_proj   → ("vlm", "q_proj")
        vlm.layers.3.mlp.gate_proj      → ("vlm", "gate_proj")
        expert.layer.0.pv               → ("expert", "pv")
        expert.layers.0.mlp.down_proj   → ("expert", "down_proj")
    """
    parts = module_id.split(".")
    comp = parts[0]
    op = parts[-1]
    return comp, op


def _component_key(r):
    """Map a module_sparsity row to a (component, phase, role) display key."""
    comp = r["component"]
    op = r["operator"]
    role = r["tensor_role"]
    if op in ("qk", "pv"):
        return (f"{comp}.{op}".upper(), r["phase"], role)
    return (comp.upper(), r["phase"], role)


def _section_summary(rows_all):
    """§14.1 summary table."""
    print("### Summary (accuracy + coverage)")
    print()
    print("| Config | Stage | Tasks | Episodes | SR | Native bit sparsity (all) | Unit sparsity | FP sidepath |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for cfg, stage, ms, us in rows_all:
        sr = _sr(stage, cfg)
        bit, _, _ = _ratio(ms, "sparse_bits_native", "total_bits_native")
        unit, _, _ = _ratio(us, "zero_units", "total_units")
        fp, _, _ = _ratio(ms, "protected_elements", "total_elements_reported")
        sr_s = f"{sr[3]:.1f}%" if sr else "n/a"
        n_ep = sr[1] if sr else 0
        n_task = sr[2] if sr else 0
        print(f"| {cfg} | {stage} | {n_task} | {n_ep} | {sr_s} | {_pct(bit)} | {_pct(unit)} | {_pct(fp)} |")
    print()


def _section_components(rows_all):
    """§14.2 component table."""
    print("### Component / role breakdown")
    print()
    print("| Config | Component | Phase | Role | Zero native | Bit sparse native | Unit sparse | FP sidepath |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    for cfg, stage, ms, us in rows_all:
        if not ms:
            continue
        agg_ms = defaultdict(list)
        for r in ms:
            agg_ms[_component_key(r)].append(r)
        agg_us = defaultdict(list)
        for r in us:
            agg_us[_unit_component_key(r)].append(r)

        def sort_key(k):
            comp, phase, role = k
            return (0 if comp.startswith("VLM") else 1, comp, 0 if phase == "prefill" else 1, role)

        for k in sorted(agg_ms, key=sort_key):
            comp, phase, role = k
            z, _, _ = _ratio(agg_ms[k], "zero_elements_native", "total_elements_native")
            b, _, _ = _ratio(agg_ms[k], "sparse_bits_native", "total_bits_native")
            fp, _, _ = _ratio(agg_ms[k], "protected_elements", "total_elements_reported")
            urows = agg_us.get(k, [])
            u, _, _ = _ratio(urows, "zero_units", "total_units") if urows else (0.0, 0, 0)
            print(f"| {cfg} | {comp} | {phase} | {role} | {_pct(z)} | {_pct(b)} | {_pct(u)} | {_pct(fp)} |")
    print()


def _section_flow_step(rows_all):
    """§14.3 flow-step table (expert denoise, per step, per MatMul role)."""
    print("### Flow-step table (expert denoise, bit sparse native)")
    print()
    roles = [("qk", "A"), ("qk", "B"), ("qk", "O"), ("pv", "A"), ("pv", "B"), ("pv", "O")]
    header = "| Config | step | " + " | ".join(f"{op.upper()} {r}" for op, r in roles) + " |"
    print(header)
    print("|---|---:|" + "---:|" * len(roles))
    for cfg, stage, ms, _us in rows_all:
        if not ms:
            continue
        per = defaultdict(list)
        for r in ms:
            if r["phase"] != "denoise" or r["operator"] not in ("qk", "pv"):
                continue
            per[(int(float(r["flow_step"])), r["operator"], r["tensor_role"])].append(r)
        for step in range(10):
            cells = []
            for op, role in roles:
                rws = per.get((step, op, role), [])
                v, _, _ = _ratio(rws, "sparse_bits_native", "total_bits_native") if rws else (0.0, 0, 0)
                cells.append(_pct(v))
            print(f"| {cfg} | {step} | " + " | ".join(cells) + " |")
    print()


def _section_weight(rows_all):
    print("### Weight static sparsity (episode-independent)")
    print()
    print("| Config | Rows | zero_rate | sparse_bit_rate | ideal_sparse_upper_bound |")
    print("|---|---:|---:|---:|---:|")
    for cfg, stage, rows in rows_all:
        if not rows:
            print(f"| {cfg} | 0 | n/a | n/a | n/a |")
            continue
        z, _, _ = _ratio(rows, "zero_elements", "total_elements")
        b, _, _ = _ratio(rows, "sparse_bits", "total_bits")
        print(f"| {cfg} ({stage}) | {len(rows)} | {_pct(z)} | {_pct(b)} | {1 / (1 - b):.3f} |")
    print()


def _section_convergence(cur, prev, excl=()):
    """§13 convergence gate |Δ| < 0.5 pp between two stages."""
    print(f"### Convergence gate ({cur} vs {prev})")
    print()
    print(f"Gate: |Δ bit sparsity| < 0.5 pp; |Δ unit zero rate| < 0.5 pp")
    if excl:
        print(f"(excluding tensor_role ∈ {{{', '.join(excl)}}} — pre-fix invalid rows)")
    print()
    print("| Config | Component | Phase | Role | Prev | Cur | Δ (pp) | Gate |")
    print("|---|---|---|---|---:|---:|---:|---|")

    def agg(ms, us, cfg, stage):
        m, u = {}, {}
        for r in ms:
            m.setdefault(_component_key(r), []).append(r)
        for r in us:
            u.setdefault(_unit_component_key(r), []).append(r)
        out = {}
        for k, rows in m.items():
            b, _, _ = _ratio(rows, "sparse_bits_native", "total_bits_native")
            urows = u.get(k, [])
            uz, _, _ = _ratio(urows, "zero_units", "total_units") if urows else (0.0, 0, 0)
            out[k] = (b, uz)
        return out

    for cfg in CONFIGS:
        a = agg(_load(prev, cfg, "module_sparsity.csv", excl),
                _load(prev, cfg, "unit_sparsity.csv", excl), cfg, prev)
        b = agg(_load(cur, cfg, "module_sparsity.csv", excl),
                _load(cur, cfg, "unit_sparsity.csv", excl), cfg, cur)
        for k in sorted(set(a) | set(b), key=lambda k: (0 if k[0].startswith("VLM") else 1, k[0], k[1], k[2])):
            if k not in a or k not in b:
                continue
            dbit = (b[k][0] - a[k][0]) * 100
            dunit = (b[k][1] - a[k][1]) * 100
            gate = "PASS" if max(abs(dbit), abs(dunit)) < 0.5 else "FAIL"
            print(f"| {cfg} | {k[0]} | {k[1]} | {k[2]} | {_pct(a[k][0])} | {_pct(b[k][0])} | "
                  f"{dbit:+.2f} | {gate} |")
    print("  (Δ shown for bit sparsity; unit rate Δ printed below)")
    for cfg in CONFIGS:
        a = agg(_load(prev, cfg, "module_sparsity.csv", excl),
                _load(prev, cfg, "unit_sparsity.csv", excl), cfg, prev)
        b = agg(_load(cur, cfg, "module_sparsity.csv", excl),
                _load(cur, cfg, "unit_sparsity.csv", excl), cfg, cur)
        for k in sorted(set(a) & set(b)):
            dunit = (b[k][1] - a[k][1]) * 100
            if abs(dunit) >= 0.5:
                print(f"  unit-rate gate FAIL: {cfg} {k}: {_pct(a[k][1])} → {_pct(b[k][1])} ({dunit:+.2f} pp)")
    print()


def _section_operators(rows_all):
    """Operator-level (q/k/v/o/gate/up/down) linear breakdown."""
    print("### Operator breakdown (linear modules, aggregated over layers)")
    print()
    print("| Config | Component | Operator | Phase | Role | Bit sparse native | Unit sparse | FP sidepath |")
    print("|---|---|---|---|---|---:|---:|---:|")
    for cfg, stage, ms, us in rows_all:
        if not ms:
            continue
        ops, uops = defaultdict(list), defaultdict(list)
        for r in ms:
            if r["operator"] not in LINEAR_OPS:
                continue
            ops[(r["component"].upper(), r["operator"], r["phase"], r["tensor_role"])].append(r)
        for r in us:
            comp, op = _parse_module_id(r["module_id"])
            if op not in LINEAR_OPS:
                continue
            uops[(comp.upper(), op, r["phase"], r["tensor_role"])].append(r)
        for k in sorted(ops, key=lambda k: (k[0], LINEAR_OPS.index(k[1]), k[2], k[3])):
            b, _, _ = _ratio(ops[k], "sparse_bits_native", "total_bits_native")
            fp, _, _ = _ratio(ops[k], "protected_elements", "total_elements_reported")
            urows = uops.get(k, [])
            u, _, _ = _ratio(urows, "zero_units", "total_units") if urows else (0.0, 0, 0)
            print(f"| {cfg} | {k[0]} | {k[1]} | {k[2]} | {k[3]} | {_pct(b)} | {_pct(u)} | {_pct(fp)} |")
    print()


def _section_workload(stage):
    """Workload-level BOP proxies (quantized operators only)."""
    print("### Workload / BOP proxy (quantized operators only)")
    print()
    print("| Config | Rows | BOP dense proxy | BOP active proxy | Effective active/dense | Calls |")
    print("|---|---:|---:|---:|---:|---:|")
    for cfg in CONFIGS:
        rows = _load(stage, cfg, "workload.csv")
        if not rows:
            continue
        dense = sum(float(r["BOP_dense_proxy"]) for r in rows)
        active = sum(float(r["BOP_active_proxy"]) for r in rows)
        calls = sum(float(r["calls"]) for r in rows)
        ratio = active / dense if dense else 0.0
        print(f"| {cfg} | {len(rows)} | {dense:.4e} | {active:.4e} | {ratio:.4f} "
              f"({1 - ratio:.2%} sparsity) | {calls:.0f} |")
    print()
    print("> `BOP_dense_proxy`/`BOP_active_proxy` 是 dense vs active bit-op 代理量，"
          "仅在已量化算子集合内可加；**不构成 whole-model MAC coverage**"
          "（raw-op runtime denominator 未 instrument，按 experiment_setup.md §14.1 记 N/A）。")
    print()


def _section_taskwise(stage):
    """§3.5 task-wise correlation: SR vs native bit sparsity per task."""
    print("### Task-wise SR vs sparsity")
    print()
    print("| Config | Task | SR | Native bit sparsity | FP sidepath |")
    print("|---|---|---:|---:|---:|")
    for cfg in CONFIGS:
        root = os.path.join(REPO_ROOT, "outputs", EXP_NAME, stage, cfg)
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            ms = _read(os.path.join(root, d, "sparsity", "module_sparsity.csv"))
            if not ms:
                continue
            p = os.path.join(root, d, "eval_info.json")
            sr = "n/a"
            if os.path.isfile(p):
                with open(p) as f:
                    sr = f"{json.load(f)['overall']['pc_success']:.1f}%"
            bit, _, _ = _ratio(ms, "sparse_bits_native", "total_bits_native")
            fp, _, _ = _ratio(ms, "protected_elements", "total_elements_reported")
            print(f"| {cfg} | {d} | {sr} | {_pct(bit)} | {_pct(fp)} |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--compare", default=None, help="earlier stage for the convergence gate")
    ap.add_argument(
        "--exclude-roles",
        default="",
        help="comma-separated tensor_role values to drop before aggregating "
             "(used to strip pre-fix output/O rows for H0/H1)",
    )
    args = ap.parse_args()
    excl = tuple(x for x in args.exclude_roles.split(",") if x)
    load = lambda stage, cfg, f: _load(stage, cfg, f, excl)  # noqa: E731

    rows_all = [
        (cfg, args.stage, load(args.stage, cfg, "module_sparsity.csv"),
         load(args.stage, cfg, "unit_sparsity.csv"))
        for cfg in CONFIGS
    ]

    if excl:
        print(f"> Excluding tensor_role ∈ {{{', '.join(excl)}}} from all aggregates.\n")

    _section_summary(rows_all)
    _section_components(rows_all)
    _section_operators(rows_all)
    _section_flow_step(rows_all)
    _section_weight([(cfg, args.stage, _load(args.stage, cfg, "weight_sparsity_static.csv"))
                     for cfg in CONFIGS])
    _section_workload(args.stage)
    _section_taskwise(args.stage)

    if args.compare:
        _section_convergence(args.stage, args.compare, excl)


if __name__ == "__main__":
    main()
