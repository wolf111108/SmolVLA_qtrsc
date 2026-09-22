#!/usr/bin/env python
"""Summarize S|MMM bit sparsity (new metric) from module/weight CSVs.

Reports, per component (vlm/expert) and stage:
  - runtime element sparsity (native)
  - runtime S|MMM bit sparsity (native)      <- new metric
  - static weight element / bit sparsity (native)

Gate: the module CSV must contain both components with the expected
module-row count (3520 = 416 VLM + 3104 Expert, matching the 09-15
quickscan protocol); otherwise exit non-zero.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def ratio(num: int, den: int) -> float:
    return 100.0 * num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity-dir", type=Path, required=True)
    args = ap.parse_args()

    module_csv = args.sparsity_dir / "module_sparsity.csv"
    weight_csv = args.sparsity_dir / "weight_sparsity_static.csv"
    for p in (module_csv, weight_csv):
        if not p.exists():
            raise SystemExit(f"MISSING: {p}")

    run = defaultdict(lambda: [0, 0, 0, 0])   # elem_n, elem_z, bit_n, bit_sparse
    n_rows = defaultdict(int)
    with module_csv.open() as f:
        for r in csv.DictReader(f):
            comp = r["component"]
            if comp not in ("vlm", "expert"):
                continue
            a = run[comp]
            a[0] += int(r["total_elements_native"])
            a[1] += int(r["zero_elements_native"])
            a[2] += int(r["total_bits_native"])
            a[3] += int(r["sparse_bits_native"])
            n_rows[comp] += 1

    wrun = defaultdict(lambda: [0, 0, 0, 0])
    with weight_csv.open() as f:
        for r in csv.DictReader(f):
            comp = "vlm" if r.get("component", "vlm") in ("vlm", "") else r["component"]
            # weight CSV may key by layer; component comes from prefix
            layer = r.get("layer_key", r.get("module_id", ""))
            if layer.startswith("vlm") or "text_model" in layer:
                comp = "vlm"
            elif layer.startswith("expert") or "lm_expert" in layer:
                comp = "expert"
            a = wrun[comp]
            for nk, dk in (("total_elements", "zero_elements"),
                           ("total_bits", "sparse_bits")):
                pass
            a[0] += int(r.get("total_elements", 0))
            a[1] += int(r.get("zero_elements", 0))
            a[2] += int(r.get("total_bits", 0))
            a[3] += int(r.get("sparse_bits", 0))

    errors = []
    if n_rows.get("vlm", 0) != 416:
        errors.append(f"vlm module rows = {n_rows.get('vlm', 0)}, expected 416")
    if n_rows.get("expert", 0) != 3104:
        errors.append(f"expert module rows = {n_rows.get('expert', 0)}, expected 3104")

    out_csv = args.sparsity_dir.parent / "smmm_sparsity_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "runtime_elem_sparsity_native_pct",
                    "runtime_smmm_bit_sparsity_native_pct",
                    "weight_elem_sparsity_native_pct",
                    "weight_smmm_bit_sparsity_native_pct"])
        for comp in ("vlm", "expert", "pooled"):
            if comp == "pooled":
                m = [sum(run[c][i] for c in ("vlm", "expert")) for i in range(4)]
                t = [sum(wrun[c][i] for c in ("vlm", "expert")) for i in range(4)]
            else:
                m, t = run[comp], wrun[comp]
            w.writerow([comp, f"{ratio(m[1], m[0]):.2f}", f"{ratio(m[3], m[2]):.2f}",
                        f"{ratio(t[1], t[0]):.2f}", f"{ratio(t[3], t[2]):.2f}"])

    print(f"\n=== S|MMM bit sparsity (native) — {args.sparsity_dir} ===")
    for comp in ("vlm", "expert", "pooled"):
        if comp == "pooled":
            m = [sum(run[c][i] for c in ("vlm", "expert")) for i in range(4)]
            t = [sum(wrun[c][i] for c in ("vlm", "expert")) for i in range(4)]
        else:
            m, t = run[comp], wrun[comp]
        print(f"{comp:8s} runtime elem {ratio(m[1], m[0]):6.2f}%  "
              f"runtime S|MMM bit {ratio(m[3], m[2]):6.2f}%  "
              f"weight elem {ratio(t[1], t[0]):6.2f}%  "
              f"weight S|MMM bit {ratio(t[3], t[2]):6.2f}%")
    print(f"summary written: {out_csv}")

    if errors:
        print("\nSUMMARY GATE: FAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(2)
    print("\nSUMMARY GATE: PASS")


if __name__ == "__main__":
    main()
