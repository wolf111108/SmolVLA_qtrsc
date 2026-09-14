#!/usr/bin/env python
"""H1-Audit: aggregate fp_code_audit.csv by (tensor_role, phase).

Reads the expanded-column CSV produced by StatManager.export_fp_code_audit_csv
and prints per-(role, phase) summary rates to judge whether the output/O
~0.5% bit sparsity is a real post-quant E4M3 code distribution or a bug.
"""

import csv
import sys
from collections import defaultdict


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <fp_code_audit.csv>", file=sys.stderr)
        return 2

    path = sys.argv[1]
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"no rows in {path}", file=sys.stderr)
        return 1

    def add(a, k, v):
        a[k] += int(float(v))

    groups = defaultdict(lambda: defaultdict(int))

    for r in rows:
        key = (r["tensor_role"], r["phase"])
        g = groups[key]

        for name in [
            "valid_elements",
            "nonzero_elements",
            "zero_code_count",
            "subnormal_count",
            "nan_count",
            "saturation_count",
            "sig_zero_bits",
            "sig_total_bits",
            "sig_nonzero_zero_bits",
            "sig_nonzero_total_bits",
        ]:
            add(g, name, r[name])

        for i in range(8):
            add(g, f"mant_nonzero_{i:03b}", r[f"mant_nonzero_{i:03b}"])

    print(
        f"{'role':12s} "
        f"{'phase':10s} "
        f"{'zero':>10s} "
        f"{'sig-all':>10s} "
        f"{'sig-nz':>10s} "
        f"{'mant111':>10s} "
        f"{'sat':>10s}"
    )

    for (role, phase), g in sorted(groups.items()):
        valid = g["valid_elements"]
        sig_total = g["sig_total_bits"]
        sig_nz_total = g["sig_nonzero_total_bits"]

        mant_nz_total = sum(
            g[f"mant_nonzero_{i:03b}"] for i in range(8)
        )

        zero_rate = g["zero_code_count"] / valid if valid else 0
        sig_rate = g["sig_zero_bits"] / sig_total if sig_total else 0
        sig_nz_rate = (
            g["sig_nonzero_zero_bits"] / sig_nz_total
            if sig_nz_total else 0
        )
        mant111 = (
            g["mant_nonzero_111"] / mant_nz_total
            if mant_nz_total else 0
        )
        sat = g["saturation_count"] / valid if valid else 0

        print(
            f"{role:12s} "
            f"{phase:10s} "
            f"{zero_rate:10.3%} "
            f"{sig_rate:10.3%} "
            f"{sig_nz_rate:10.3%} "
            f"{mant111:10.3%} "
            f"{sat:10.3%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
