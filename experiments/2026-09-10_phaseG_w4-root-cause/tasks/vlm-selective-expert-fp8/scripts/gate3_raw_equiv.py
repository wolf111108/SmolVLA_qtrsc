#!/usr/bin/env python
"""Gate 3: Linear raw-equivalence for G6 component-aware routing upgrade.

Verifies the routing upgrade (resolve_linear_quant_config + module_id
threading) did NOT change the raw (unquantized) forward: for every wrapped
QuantizedLinear, mode="raw" must be bit-exact with F.linear(x, weight, bias).

Usage:
  python gate3_raw_equiv.py [--config <g6 config>] [--seed 1234]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from vla_tcs2.model_wrapper import ModelWrapper  # noqa: E402
from vla_tcs2.quant_linear import QuantizedLinear  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    wrapper = ModelWrapper(config=config)
    model = wrapper.build(mode="raw")

    torch.manual_seed(args.seed)

    n_checked = 0
    max_diff = 0.0
    worst = None

    for module in model.modules():
        if not isinstance(module, QuantizedLinear):
            continue

        mid = getattr(module, "module_id", "")
        in_features = module.in_features
        dtype = module.weight.dtype

        # 3D input [B, T, K] exercises the same path as runtime (bias-aware).
        B, T = 2, 4
        x = torch.randn(B, T, in_features, device=args.device, dtype=dtype)

        # Force raw mode (bypass any quant path).
        module.mode = "raw"
        with torch.no_grad():
            y1 = module(x)
            y2 = F.linear(x, module.weight, module.bias)

        diff = (y1 - y2).abs().max().item()
        n_checked += 1
        if diff > max_diff:
            max_diff = diff
            worst = mid

    print(f"=== G6 Gate 3: Linear raw-equivalence ===")
    print(f"  checked : {n_checked} QuantizedLinear")
    print(f"  max|diff| : {max_diff:.3e}")
    print(f"  worst   : {worst}")

    # raw mode must be bit-exact (weight/bias are direct clones).
    assert n_checked == 224, f"expected 224 Linear, got {n_checked}"
    assert max_diff == 0.0, f"raw mode must be bit-exact, got {max_diff}"

    print("GATE 3: PASS")


if __name__ == "__main__":
    main()
