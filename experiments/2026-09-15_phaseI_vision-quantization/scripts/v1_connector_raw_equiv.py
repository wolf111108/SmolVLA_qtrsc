#!/usr/bin/env python
"""V1-R — Connector raw-wrapper equivalence.

Build two models from the SAME config but toggle connector wrapping off/on,
switch the wrapped connector to raw mode, feed an identical vision-hidden
tensor, and compare connector output. QuantizedLinear(raw) must reproduce
nn.Linear exactly (weight/bias are cloned; raw forward is F.linear).

Usage:
  python v1_connector_raw_equiv.py --config <v1 config yaml>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from vla_tcs2.model_wrapper import ModelWrapper, switch_quantization_mode_all  # noqa: E402
from vla_tcs2.quant_linear import QuantizedLinear  # noqa: E402


def _build(config, connector_enabled: bool):
    cfg = yaml.safe_load(yaml.dump(config))  # deep copy
    cfg["quantization"]["connector"] = {
        "enabled": connector_enabled,
        "calibration_policy": "recalibrate",
    }
    wrapper = ModelWrapper(config=cfg)
    model = wrapper.build()
    if connector_enabled:
        switch_quantization_mode_all(model, "raw")
    return model


def _get_connector(model):
    return model.model.vlm_with_expert.get_vlm_model().connector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(0)
    device = config.get("model", {}).get("device", "cuda")
    # Vision hidden states: [B, 1024, 768] (patch tokens before connector).
    # Match the connector weight dtype (bf16 in the real model).
    x = torch.randn(1, 1024, 768, device=device).to(torch.bfloat16)

    print("building original (connector off)...")
    model_orig = _build(config, connector_enabled=False)

    print("building wrapped (connector on, raw mode)...")
    model_wrapped = _build(config, connector_enabled=True)

    conn_orig = _get_connector(model_orig)
    conn_wrapped = _get_connector(model_wrapped)

    proj_orig = conn_orig.modality_projection.proj
    proj_wrapped = conn_wrapped.modality_projection.proj

    print(f"original connector proj type: {type(proj_orig).__name__}")
    print(f"wrapped  connector proj type: {type(proj_wrapped).__name__}")
    assert isinstance(proj_wrapped, QuantizedLinear), "connector not wrapped"

    with torch.no_grad():
        out_orig = conn_orig(x.clone())
        out_wrapped = conn_wrapped(x.clone())

    diff = (out_orig - out_wrapped).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel = (diff / (out_orig.abs() + 1e-8)).max().item()

    print("=" * 60)
    print("V1-R connector raw equivalence")
    print("=" * 60)
    print(f"max_abs_error  : {max_abs:.3e}")
    print(f"mean_abs_error : {mean_abs:.3e}")
    print(f"max_rel_error  : {rel:.3e}")

    # Raw wrapper should be exact (bit-identical) or dtype-rounding level.
    if max_abs > 1e-4:
        print("V1-R: FAIL (raw wrapper diverged)")
        return 1
    print("V1-R: PASS (raw wrapper exact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
