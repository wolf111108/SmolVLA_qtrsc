#!/usr/bin/env python
"""Audit G6 routing: verify each config wraps the expected Linear/MatMul counts.

For each QuantizedLinear, dump module_id / component / layer_idx / operator /
a_bit / w_bit / o_bit / method / scale_group / override_names to CSV, and print
aggregate FP8 vs W4 Linear counts + MatMul count.

Usage:
  python audit_routing.py --config <path-to-config.yaml>
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

import yaml  # noqa: E402

from vla_tcs2.model_wrapper import ModelWrapper  # noqa: E402
from vla_tcs2.quant_linear import QuantizedLinear  # noqa: E402
from vla_tcs2.quant_matmul import QuantizedMatMul  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    wrapper = ModelWrapper(config=config)
    model = wrapper.build()

    rows = []
    n_fp8 = 0
    n_w4 = 0
    n_matmul = 0

    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            mid = getattr(module, "module_id", "")
            component = mid.split(".")[0] if mid else ""
            operator = mid.split(".")[-1] if mid else ""
            layer_idx = getattr(module, "layer_idx", -1)
            w_bit = getattr(module, "w_bit", None)
            is_w4 = str(w_bit) == "4"
            if is_w4:
                n_w4 += 1
            else:
                n_fp8 += 1

            rows.append([
                mid, component, layer_idx, operator,
                getattr(module, "a_bit", None),
                w_bit,
                getattr(module, "o_bit", None),
                getattr(module, "method", ""),
                getattr(module, "scale_group_name", ""),
                getattr(module, "scale_group_idx", ""),
                ",".join(getattr(module, "quant_override_names", [])),
            ])
        elif isinstance(module, QuantizedMatMul):
            n_matmul += 1

    # CSV -> <repo_root>/outputs/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/routing/
    # __file__ = .../experiments/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/scripts/audit_routing.py
    # 6 dirnames up = repo root
    repo_root = os.path.abspath(__file__)
    for _ in range(6):
        repo_root = os.path.dirname(repo_root)
    out_dir = os.path.join(
        repo_root, "outputs", "2026-09-10_phaseG_w4-root-cause",
        "tasks", "vlm-selective-expert-fp8", "routing",
    )
    os.makedirs(out_dir, exist_ok=True)
    cfg_name = os.path.basename(args.config).replace(".yaml", "")
    csv_path = os.path.join(out_dir, f"{cfg_name}_routing.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "module_id", "component", "layer_idx", "operator",
            "a_bit", "w_bit", "o_bit", "method",
            "scale_group_name", "scale_group_idx", "override_names",
        ])
        w.writerows(rows)

    print(f"=== {cfg_name} routing ===")
    print(f"  FP8 Linear  : {n_fp8}")
    print(f"  W4  Linear  : {n_w4}")
    print(f"  Total Linear: {n_fp8 + n_w4}")
    print(f"  MatMul      : {n_matmul}")
    print(f"  CSV         : {csv_path}")


if __name__ == "__main__":
    main()
