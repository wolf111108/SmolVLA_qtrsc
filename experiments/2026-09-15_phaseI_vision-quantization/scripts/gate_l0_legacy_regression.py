#!/usr/bin/env python
"""Gate L0 — legacy regression after Phase I vision wrapping code.

Build a canonical G6 config (unchanged) and verify the legacy wrapping is
untouched: QuantizedLinear == 224, QuantizedMatMul == 64, and ZERO
vision.* / connector.* module_ids (vision.enabled / connector.enabled are
absent from the legacy config, so the default-off gate must hold).

Usage:
  python gate_l0_legacy_regression.py --config <path-to-canonical-g6-config>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

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

    n_linear = 0
    n_matmul = 0
    n_vision = 0
    n_connector = 0
    vision_ids = []
    connector_ids = []

    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            mid = getattr(module, "module_id", "")
            n_linear += 1
            if mid.startswith("vision."):
                n_vision += 1
                vision_ids.append(mid)
            elif mid.startswith("connector."):
                n_connector += 1
                connector_ids.append(mid)
        elif isinstance(module, QuantizedMatMul):
            n_matmul += 1

    print("=" * 60)
    print("Gate L0 — legacy regression")
    print("=" * 60)
    print(f"QuantizedLinear : {n_linear}")
    print(f"QuantizedMatMul : {n_matmul}")
    print(f"vision.*        : {n_vision}")
    print(f"connector.*     : {n_connector}")

    ok = True
    if n_linear != 224:
        print(f"FAIL: expected 224 QuantizedLinear, got {n_linear}")
        ok = False
    if n_matmul != 64:
        print(f"FAIL: expected 64 QuantizedMatMul, got {n_matmul}")
        ok = False
    if n_vision != 0:
        print(f"FAIL: expected 0 vision modules, got {n_vision}: {vision_ids}")
        ok = False
    if n_connector != 0:
        print(
            f"FAIL: expected 0 connector modules, got {n_connector}: "
            f"{connector_ids}"
        )
        ok = False

    if ok:
        print("Gate L0: PASS (224 Linear / 64 MatMul / 0 vision / 0 connector)")
        return 0
    print("Gate L0: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
