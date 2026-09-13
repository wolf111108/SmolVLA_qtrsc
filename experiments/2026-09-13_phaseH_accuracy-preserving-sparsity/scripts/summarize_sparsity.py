#!/usr/bin/env python
"""Phase H sparsity aggregation (experiment_setup.md §13).

Aggregates module_sparsity.csv / weight_sparsity_static.csv across tasks by
summing numerators/denominators (never averaging per-layer ratios). Prints
the headline metrics used for H1/H2 convergence checks and the final tables.

Usage:
    python summarize_sparsity.py --stage h1_10ep [--config s0_fp8_all]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP_NAME = "2026-09-13_phaseH_accuracy-preserving-sparsity"


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _agg(rows, keyfn):
    """Sum numerator/denominator pairs grouped by keyfn."""
    acc = defaultdict(lambda: {"num": 0, "den": 0})
    for r in rows:
        k = keyfn(r)
        acc[k]["num"] += float(r["sparse_bits_native"])
        acc[k]["den"] += float(r["total_bits_native"])
    return {k: v["num"] / v["den"] if v["den"] > 0 else 0.0 for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["h0_smoke", "h1_10ep", "h2_30ep", "h3_100ep"])
    ap.add_argument("--config", default=None, help="s0_fp8_all | s1_expert_w4 (default: both)")
    args = ap.parse_args()

    configs = [args.config] if args.config else ["s0_fp8_all", "s1_expert_w4"]

    for cfg in configs:
        task_root = os.path.join(
            REPO_ROOT, "outputs", EXP_NAME, args.stage, cfg
        )
        rows = []
        for d in sorted(os.listdir(task_root)):
            p = os.path.join(task_root, d, "sparsity", "module_sparsity.csv")
            if os.path.isfile(p):
                rows.extend(_read(p))

        if not rows:
            print(f"[{cfg}] no module_sparsity.csv found under {task_root}")
            continue

        print(f"\n===== {cfg} ({args.stage}) =====")

        def keyfn(r):
            comp = r["component"]
            phase = r["phase"]
            role = r["tensor_role"]
            op = r["operator"]
            if op in ("qk", "pv"):
                comp = f"{comp}.{op}"
            return (comp, phase, role)

        agg = _agg(rows, keyfn)
        for (comp, phase, role) in sorted(agg):
            print(f"  {comp:16s} {phase:8s} {role:11s} "
                  f"sparse_bit_rate_native = {agg[(comp, phase, role)]:.4%}")

        # fp sidepath ratio (aggregate)
        fp_num = sum(float(r["protected_elements"]) for r in rows)
        fp_den = sum(float(r["total_elements_reported"]) for r in rows)
        print(f"  overall fp_sidepath_ratio = {fp_num / fp_den:.4%}" if fp_den else "  n/a")


if __name__ == "__main__":
    main()
