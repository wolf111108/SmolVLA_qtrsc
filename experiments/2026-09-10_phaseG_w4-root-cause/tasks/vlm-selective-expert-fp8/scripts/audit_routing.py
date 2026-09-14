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
from pathlib import Path

# __file__ = .../experiments/<exp>/tasks/<task>/scripts/audit_routing.py
# 6 parents up = repo root.
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from vla_tcs2.model_wrapper import ModelWrapper  # noqa: E402
from vla_tcs2.quant_linear import QuantizedLinear  # noqa: E402
from vla_tcs2.quant_matmul import QuantizedMatMul  # noqa: E402

# Expected (FP8 Linear, W4 Linear, MatMul) per config (manual §6.2).
EXPECTED = {
    "g6a_all_fp8_control": (224, 0, 64),
    "g6b_vlm_attn_w4_expert_fp8": (160, 64, 64),
    "g6c_vlm_mlp_w4_expert_fp8": (176, 48, 64),
    "g6d_vlm_all_w4_expert_fp8": (112, 112, 64),
}


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
            method = getattr(module, "method", "")

            # Strict classification: only e4m3 counts as FP8, only 4 as W4
            # (audit §9). Any other precision fails loudly instead of being
            # silently folded into FP8.
            if str(w_bit) == "4":
                assert method == "pot_ao_outlier", (
                    f"W4 module {mid} has unexpected method {method!r}"
                )
                n_w4 += 1
            elif str(w_bit).lower() == "e4m3":
                assert method == "pot_fp8_outlier", (
                    f"FP8 module {mid} has unexpected method {method!r}"
                )
                n_fp8 += 1
            else:
                raise AssertionError(
                    f"Unexpected w_bit={w_bit!r} at {mid}"
                )

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

    # Hard Gate (audit §10): exit non-zero on any mismatch.
    if cfg_name not in EXPECTED:
        raise SystemExit(
            f"ERROR: no expected routing defined for {cfg_name}"
        )
    exp_fp8, exp_w4, exp_mm = EXPECTED[cfg_name]
    assert n_fp8 == exp_fp8, (
        f"{cfg_name}: FP8 Linear {n_fp8} != expected {exp_fp8}"
    )
    assert n_w4 == exp_w4, (
        f"{cfg_name}: W4 Linear {n_w4} != expected {exp_w4}"
    )
    assert n_matmul == exp_mm, (
        f"{cfg_name}: MatMul {n_matmul} != expected {exp_mm}"
    )
    print(f"  ROUTING GATE: PASS")


if __name__ == "__main__":
    main()
