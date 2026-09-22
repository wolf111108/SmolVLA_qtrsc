#!/usr/bin/env python
"""Summarize full Vision+VLM+Expert S|MMM sparsity.

Primary metric:
  E4M3 raw S EEEE MMM -> S|MMM (4 bits)
  exponent excluded; hidden leading 1 excluded.

Uses native counters only, so outlier-protection artificial zeros do not inflate
sparsity. Pooled values are sum(numerators)/sum(denominators), never arithmetic
means of per-row ratios.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_RUNTIME_ROWS = {
    "vision": 144,
    "vlm": 320,
    "expert": 3200,
}
EXPECTED_WEIGHT_ROWS = {
    "vision": 72,
    "vlm": 112,
    "expert": 112,
}
EXPECTED_WEIGHT_TOTAL = 296


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else float("nan")


def read_csv(path: Path):
    if not path.is_file():
        raise SystemExit(f"MISSING: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def add(counter, elem_total, elem_zero, bit_total, bit_zero):
    counter[0] += int(elem_total)
    counter[1] += int(elem_zero)
    counter[2] += int(bit_total)
    counter[3] += int(bit_zero)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity-dir", type=Path, required=True)
    args = ap.parse_args()

    module_rows = read_csv(args.sparsity_dir / "module_sparsity.csv")
    weight_rows = read_csv(args.sparsity_dir / "weight_sparsity_static.csv")
    manifest_rows = read_csv(args.sparsity_dir / "quantization_manifest.csv")

    runtime = defaultdict(lambda: [0, 0, 0, 0])
    runtime_rows = defaultdict(int)
    phases = defaultdict(set)
    flow_steps = defaultdict(set)

    for r in module_rows:
        comp = (r.get("component") or "").strip()
        if comp not in EXPECTED_RUNTIME_ROWS:
            continue
        add(
            runtime[comp],
            r["total_elements_native"],
            r["zero_elements_native"],
            r["total_bits_native"],
            r["sparse_bits_native"],
        )
        runtime_rows[comp] += 1
        phases[comp].add((r.get("phase") or "").strip())
        fs = (r.get("flow_step") or "").strip()
        if fs:
            flow_steps[comp].add(fs)

    weights = defaultdict(lambda: [0, 0, 0, 0])
    weight_counts = defaultdict(int)
    for r in weight_rows:
        comp = (r.get("component") or "").strip()
        if comp not in EXPECTED_WEIGHT_ROWS:
            # Backward-compatible prefix fallback.
            mid = (r.get("module_id") or r.get("layer_key") or "").strip()
            if mid.startswith("vision"):
                comp = "vision"
            elif mid.startswith("vlm") or "text_model" in mid:
                comp = "vlm"
            elif mid.startswith("expert") or "lm_expert" in mid:
                comp = "expert"
        if comp not in EXPECTED_WEIGHT_ROWS:
            continue
        add(
            weights[comp],
            r.get("total_elements", 0),
            r.get("zero_elements", 0),
            r.get("total_bits", 0),
            r.get("sparse_bits", 0),
        )
        weight_counts[comp] += 1

    errors = []
    for comp, expected in EXPECTED_RUNTIME_ROWS.items():
        got = runtime_rows.get(comp, 0)
        if got != expected:
            errors.append(f"{comp} runtime rows={got}, expected={expected}")
    for comp, expected in EXPECTED_WEIGHT_ROWS.items():
        got = weight_counts.get(comp, 0)
        if got != expected:
            errors.append(f"{comp} weight rows={got}, expected={expected}")

    if len(weight_rows) != EXPECTED_WEIGHT_TOTAL:
        errors.append(
            f"static weight rows={len(weight_rows)}, expected={EXPECTED_WEIGHT_TOTAL}"
        )

    # Routing gate from manifest.
    route = defaultdict(int)
    for r in manifest_rows:
        comp = (r.get("component") or "").strip()
        op_type = (r.get("op_type") or "").strip()
        route[(comp, op_type)] += 1
    expected_route = {
        ("vision", "linear"): 72,
        ("vlm", "linear"): 112,
        ("vlm", "matmul"): 32,
        ("expert", "linear"): 112,
        ("expert", "matmul"): 32,
    }
    if len(manifest_rows) != 360:
        errors.append(f"manifest rows={len(manifest_rows)}, expected=360")
    for key, expected in expected_route.items():
        if route[key] != expected:
            errors.append(f"routing {key}={route[key]}, expected={expected}")

    # Expert must cover all ten denoise steps.
    expert_steps = {str(i) for i in range(10)}
    missing_steps = expert_steps - flow_steps.get("expert", set())
    if missing_steps:
        errors.append(f"expert flow_step missing={sorted(missing_steps)}")

    summary_rows = []
    for comp in ("vision", "vlm", "expert"):
        r = runtime[comp]
        w = weights[comp]
        summary_rows.append({
            "component": comp,
            "runtime_rows": runtime_rows[comp],
            "runtime_total_elements_native": r[0],
            "runtime_zero_elements_native": r[1],
            "runtime_element_sparsity_native_pct": f"{pct(r[1], r[0]):.4f}",
            "runtime_total_smmm_bits_native": r[2],
            "runtime_zero_smmm_bits_native": r[3],
            "runtime_smmm_bit_sparsity_native_pct": f"{pct(r[3], r[2]):.4f}",
            "weight_rows": weight_counts[comp],
            "weight_total_elements": w[0],
            "weight_zero_elements": w[1],
            "weight_element_sparsity_pct": f"{pct(w[1], w[0]):.6f}",
            "weight_total_smmm_bits": w[2],
            "weight_zero_smmm_bits": w[3],
            "weight_smmm_bit_sparsity_pct": f"{pct(w[3], w[2]):.4f}",
            "fp_bit_metric": "S|MMM",
            "fp_bit_metric_version": "1",
        })

    pooled_r = [sum(runtime[c][i] for c in ("vision", "vlm", "expert")) for i in range(4)]
    pooled_w = [sum(weights[c][i] for c in ("vision", "vlm", "expert")) for i in range(4)]
    summary_rows.append({
        "component": "all_quantized",
        "runtime_rows": sum(runtime_rows.values()),
        "runtime_total_elements_native": pooled_r[0],
        "runtime_zero_elements_native": pooled_r[1],
        "runtime_element_sparsity_native_pct": f"{pct(pooled_r[1], pooled_r[0]):.4f}",
        "runtime_total_smmm_bits_native": pooled_r[2],
        "runtime_zero_smmm_bits_native": pooled_r[3],
        "runtime_smmm_bit_sparsity_native_pct": f"{pct(pooled_r[3], pooled_r[2]):.4f}",
        "weight_rows": sum(weight_counts.values()),
        "weight_total_elements": pooled_w[0],
        "weight_zero_elements": pooled_w[1],
        "weight_element_sparsity_pct": f"{pct(pooled_w[1], pooled_w[0]):.6f}",
        "weight_total_smmm_bits": pooled_w[2],
        "weight_zero_smmm_bits": pooled_w[3],
        "weight_smmm_bit_sparsity_pct": f"{pct(pooled_w[3], pooled_w[2]):.4f}",
        "fp_bit_metric": "S|MMM",
        "fp_bit_metric_version": "1",
    })

    out = args.sparsity_dir.parent / "vision_smmm_sparsity_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("\n=== FULL VLIN S|MMM SPARSITY ===")
    for r in summary_rows:
        print(
            f"{r['component']:13s} "
            f"runtime elem={float(r['runtime_element_sparsity_native_pct']):7.3f}% "
            f"S|MMM={float(r['runtime_smmm_bit_sparsity_native_pct']):7.3f}% | "
            f"weight elem={float(r['weight_element_sparsity_pct']):9.6f}% "
            f"S|MMM={float(r['weight_smmm_bit_sparsity_pct']):7.3f}%"
        )
    print(f"summary: {out}")
    print("metric : S|MMM v1 (sign+mantissa; exponent/hidden-1 excluded)")

    if errors:
        print("\nSUMMARY GATE: FAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(2)

    print("\nSUMMARY GATE: PASS")


if __name__ == "__main__":
    main()
