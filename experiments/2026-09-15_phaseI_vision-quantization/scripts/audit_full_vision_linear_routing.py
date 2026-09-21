#!/usr/bin/env python
"""Gate: audit full Vision Encoder Linear routing.

Builds the real SmolVLA checkpoint and verifies that the full Vision Linear
integration changes ONLY the 72 Vision Transformer Linear sites.

Expected:
  legacy VLM/Expert QuantizedLinear : 224
  Vision QuantizedLinear            : 72
    - attention projections         : 48
    - MLP fc1/fc2                   : 24
  Connector QuantizedLinear         : 0
  total QuantizedLinear             : 296
  VLM/Expert QuantizedMatMul        : 64
  Vision QuantizedMatMul            : 0 (Vision attention is still SDPA)

It also checks module-id uniqueness, per-site scale-group uniqueness, effective
FP8 precision, and the strict calibration action split:
  reuse=288, recalibrate=72
where 288 = 224 legacy Linear + 64 legacy MatMul.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vla_tcs2.calibration import calibration_action_summary  # noqa: E402
from vla_tcs2.model_wrapper import ModelWrapper  # noqa: E402
from vla_tcs2.quant_linear import QuantizedLinear  # noqa: E402
from vla_tcs2.quant_matmul import QuantizedMatMul  # noqa: E402

DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/2026-09-15_phaseI_vision-quantization/configs/"
      "vlin_full_vision_linear_fp8.yaml"
)

ATTN_OPS = {"q_proj", "k_proj", "v_proj", "out_proj"}
MLP_OPS = {"fc1", "fc2"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    return ap.parse_args()


def spec_name(spec) -> str:
    if spec is None:
        return ""
    name = getattr(spec, "name", None)
    return name() if callable(name) else str(name or spec)


def expected_vision_ids() -> set[str]:
    ids: set[str] = set()
    for i in range(12):
        ids.update(
            f"vision.layers.{i}.self_attn.{op}" for op in sorted(ATTN_OPS)
        )
        ids.update(f"vision.layers.{i}.mlp.{op}" for op in sorted(MLP_OPS))
    return ids


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wrapper = ModelWrapper(cfg)
    model = wrapper.build()

    qlinear = [m for m in model.modules() if isinstance(m, QuantizedLinear)]
    qmatmul = [m for m in model.modules() if isinstance(m, QuantizedMatMul)]

    linear_ids = [getattr(m, "module_id", "") or "" for m in qlinear]
    matmul_ids = [getattr(m, "module_id", "") or "" for m in qmatmul]

    vision = [m for m in qlinear if (getattr(m, "module_id", "") or "").startswith("vision.")]
    connector = [m for m in qlinear if (getattr(m, "module_id", "") or "").startswith("connector.")]
    legacy = [
        m for m in qlinear
        if (getattr(m, "module_id", "") or "").startswith(("vlm.", "expert."))
    ]
    other = [
        m for m in qlinear
        if not (getattr(m, "module_id", "") or "").startswith(
            ("vision.", "connector.", "vlm.", "expert.")
        )
    ]

    vision_attn = [
        m for m in vision
        if ".self_attn." in (getattr(m, "module_id", "") or "")
    ]
    vision_mlp = [
        m for m in vision
        if ".mlp." in (getattr(m, "module_id", "") or "")
    ]
    vision_mm = [
        m for m in qmatmul
        if (getattr(m, "module_id", "") or "").startswith("vision.")
    ]

    expected = expected_vision_ids()
    actual = {getattr(m, "module_id", "") or "" for m in vision}

    errors: list[str] = []

    checks = {
        "QuantizedLinear total": (len(qlinear), 296),
        "legacy VLM/Expert Linear": (len(legacy), 224),
        "Vision Linear": (len(vision), 72),
        "Vision attention projection": (len(vision_attn), 48),
        "Vision MLP": (len(vision_mlp), 24),
        "Connector Linear": (len(connector), 0),
        "Other Linear": (len(other), 0),
        "QuantizedMatMul total": (len(qmatmul), 64),
        "Vision QuantizedMatMul": (len(vision_mm), 0),
    }

    print("\n=== ROUTING COUNTS ===")
    for name, (got, want) in checks.items():
        ok = got == want
        print(f"{name:30s}: {got:4d}  expected={want:4d}  {'OK' if ok else 'FAIL'}")
        if not ok:
            errors.append(f"{name}: got {got}, expected {want}")

    if len(linear_ids) != len(set(linear_ids)):
        dup = [k for k, v in Counter(linear_ids).items() if v > 1]
        errors.append(f"duplicate QuantizedLinear module_id: {dup}")
    if len(matmul_ids) != len(set(matmul_ids)):
        dup = [k for k, v in Counter(matmul_ids).items() if v > 1]
        errors.append(f"duplicate QuantizedMatMul module_id: {dup}")

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"missing Vision module_ids: {missing}")
    if extra:
        errors.append(f"unexpected Vision module_ids: {extra}")

    # Every Vision Linear should resolve to FP8 E4M3 A/W/O.
    bad_precision = []
    bad_method = []
    bad_policy = []
    scale_groups = []
    for m in vision:
        mid = getattr(m, "module_id", "")
        kinds = (spec_name(m.a_spec), spec_name(m.w_spec), spec_name(m.o_spec))
        if kinds != ("e4m3", "e4m3", "e4m3"):
            bad_precision.append((mid, kinds))
        if getattr(m, "method", "") != "pot_fp8_outlier":
            bad_method.append((mid, getattr(m, "method", "")))
        if getattr(m, "calibration_policy", "") != "recalibrate":
            bad_policy.append((mid, getattr(m, "calibration_policy", "")))
        scale_groups.append(
            (getattr(m, "scale_group_name", ""), getattr(m, "scale_group_idx", -1))
        )

    if bad_precision:
        errors.append(f"Vision precision mismatch: {bad_precision[:8]}")
    if bad_method:
        errors.append(f"Vision method mismatch: {bad_method[:8]}")
    if bad_policy:
        errors.append(f"Vision calibration policy mismatch: {bad_policy[:8]}")
    if len(scale_groups) != len(set(scale_groups)):
        errors.append("Vision per-site scale groups are not unique (expected 72/72)")

    actions = calibration_action_summary(model)
    print("\n=== CALIBRATION ACTION PLAN ===")
    print(actions)
    if actions != {"reuse": 288, "recalibrate": 72}:
        errors.append(
            f"calibration action split mismatch: {actions}; "
            "expected {'reuse': 288, 'recalibrate': 72}"
        )

    print("\n=== VISION OP BREAKDOWN ===")
    op_counts = Counter((getattr(m, "module_id", "") or "").split(".")[-1] for m in vision)
    for op in ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]:
        print(f"{op:10s}: {op_counts[op]:2d} / 12")
        if op_counts[op] != 12:
            errors.append(f"{op}: got {op_counts[op]}, expected 12")

    if errors:
        print("\nROUTING GATE: FAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(2)

    print("\nROUTING GATE: PASS")
    print("Full Vision Linear integration is exactly 72 sites; connector remains raw.")


if __name__ == "__main__":
    main()
