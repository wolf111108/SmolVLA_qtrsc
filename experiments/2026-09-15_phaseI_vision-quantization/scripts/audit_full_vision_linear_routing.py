#!/usr/bin/env python
"""Gate: audit Vision Encoder Linear routing (VLIN / V2 / V3).

Builds the real SmolVLA checkpoint and verifies that the Vision Linear
integration changes ONLY the expected Vision Transformer Linear sites.

Expected counts are derived from the config's ``vision.linear`` switches
(``attn_proj`` / ``mlp``), each covering 12 layers:

  VLIN (mlp=true,  attn_proj=true)  : Vision=72 (48 attn + 24 MLP),  total 296
  V2   (mlp=true,  attn_proj=false) : Vision=24 (MLP only),         total 248
  V3   (mlp=false, attn_proj=true)  : Vision=48 (attn proj only),   total 272

All variants share: legacy VLM/Expert QuantizedLinear = 224,
VLM/Expert QuantizedMatMul = 64, Vision QuantizedMatMul = 0 (Vision attention
remains SDPA), Connector QuantizedLinear = 0.

It also checks module-id uniqueness, per-site scale-group uniqueness, effective
FP8 precision, and the strict calibration action split:
  reuse=288, recalibrate=<n_vision>
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
VISION_LAYERS = 12
LEGACY_QUANT_SITES = 288  # 224 legacy VLM/Expert Linear + 64 VLM/Expert MatMul


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    return ap.parse_args()


def spec_name(spec) -> str:
    if spec is None:
        return ""
    name = getattr(spec, "name", None)
    return name() if callable(name) else str(name or spec)


def expected_vision_ids(wrap_attn: bool, wrap_mlp: bool) -> set[str]:
    ids: set[str] = set()
    for i in range(VISION_LAYERS):
        if wrap_attn:
            ids.update(
                f"vision.layers.{i}.self_attn.{op}" for op in sorted(ATTN_OPS)
            )
        if wrap_mlp:
            ids.update(f"vision.layers.{i}.mlp.{op}" for op in sorted(MLP_OPS))
    return ids


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wrapper = ModelWrapper(cfg)
    model = wrapper.build()

    vision_linear_cfg = (
        cfg.get("quantization", {}).get("vision", {}).get("linear", {})
    )
    wrap_attn = bool(vision_linear_cfg.get("attn_proj", False))
    wrap_mlp = bool(vision_linear_cfg.get("mlp", False))
    exp_attn = VISION_LAYERS * len(ATTN_OPS) if wrap_attn else 0
    exp_mlp = VISION_LAYERS * len(MLP_OPS) if wrap_mlp else 0
    exp_vision = exp_attn + exp_mlp
    exp_total_linear = 224 + exp_vision
    expected_actions = {"reuse": LEGACY_QUANT_SITES, "recalibrate": exp_vision}
    print(
        f"\nvariant: vision.linear(mlp={wrap_mlp}, attn_proj={wrap_attn}) "
        f"-> expect Vision={exp_vision} ({exp_attn} attn + {exp_mlp} MLP), "
        f"total QuantizedLinear={exp_total_linear}"
    )

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

    expected = expected_vision_ids(wrap_attn, wrap_mlp)
    actual = {getattr(m, "module_id", "") or "" for m in vision}

    errors: list[str] = []

    checks = {
        "QuantizedLinear total": (len(qlinear), exp_total_linear),
        "legacy VLM/Expert Linear": (len(legacy), 224),
        "Vision Linear": (len(vision), exp_vision),
        "Vision attention projection": (len(vision_attn), exp_attn),
        "Vision MLP": (len(vision_mlp), exp_mlp),
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
        errors.append(
            f"Vision per-site scale groups are not unique "
            f"(expected {exp_vision} unique)"
        )

    actions = calibration_action_summary(model)
    print("\n=== CALIBRATION ACTION PLAN ===")
    print(actions)
    if actions != expected_actions:
        errors.append(
            f"calibration action split mismatch: {actions}; "
            f"expected {expected_actions}"
        )

    print("\n=== VISION OP BREAKDOWN ===")
    op_counts = Counter((getattr(m, "module_id", "") or "").split(".")[-1] for m in vision)
    for op in ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]:
        want = VISION_LAYERS if (
            (op in ATTN_OPS and wrap_attn) or (op in MLP_OPS and wrap_mlp)
        ) else 0
        print(f"{op:10s}: {op_counts[op]:2d} / {want}")
        if op_counts[op] != want:
            errors.append(f"{op}: got {op_counts[op]}, expected {want}")

    if errors:
        print("\nROUTING GATE: FAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(2)

    print("\nROUTING GATE: PASS")
    print(
        f"Vision Linear routing is exactly {exp_vision} sites "
        f"({exp_attn} attn_proj + {exp_mlp} MLP); connector remains raw."
    )


if __name__ == "__main__":
    main()
