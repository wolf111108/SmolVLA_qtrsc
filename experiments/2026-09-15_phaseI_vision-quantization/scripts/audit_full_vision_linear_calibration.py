#!/usr/bin/env python
"""Gate: verify calibration coverage for the Vision Encoder Linear sites.

Run this AFTER:
    python main.py --config <cfg> --skip-evaluation

The Vision Linear experiments use per-site scale groups. Each Vision
QuantizedLinear must therefore have exactly three persisted scale files:
    <scale_group>_a_scale_<idx>.p
    <scale_group>_w_scale_<idx>.p
    <scale_group>_o_scale_<idx>.p

Expected counts are derived from the config's ``vision.linear`` switches
(each family covers 12 layers):

  VLIN (mlp=true,  attn_proj=true)  : 72 sites, 72/72 complete, 216 files
  V2   (mlp=true,  attn_proj=false) : 24 sites, 24/24 complete,  72 files
  V3   (mlp=false, attn_proj=true)  : 48 sites, 48/48 complete, 144 files

Every scale must be finite and strictly positive. Legacy VLM/Expert scales
are copied from the canonical G6-A control by the runner and are
intentionally not recalibrated here.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vla_tcs2.model_wrapper import ModelWrapper  # noqa: E402
from vla_tcs2.quant_linear import QuantizedLinear  # noqa: E402

DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/2026-09-15_phaseI_vision-quantization/configs/"
      "vlin_full_vision_linear_fp8.yaml"
)

VISION_LAYERS = 12
ATTN_OPS = ("q_proj", "k_proj", "v_proj", "out_proj")
MLP_OPS = ("fc1", "fc2")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    return ap.parse_args()


def scale_is_valid(value) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(torch.isfinite(value).all() and (value > 0).all())
    try:
        f = float(value)
    except Exception:
        return False
    return math.isfinite(f) and f > 0.0


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scale_dir = Path(cfg["quantization"]["scale_dir"])
    if not scale_dir.is_absolute():
        scale_dir = REPO_ROOT / scale_dir

    if not scale_dir.exists():
        raise SystemExit(
            f"CALIBRATION COVERAGE GATE: FAIL\nscale_dir does not exist: {scale_dir}"
        )

    # Build only to recover the authoritative module_id -> scale_group mapping.
    wrapper = ModelWrapper(cfg)
    model = wrapper.build()

    vision = [
        m for m in model.modules()
        if isinstance(m, QuantizedLinear)
        and (getattr(m, "module_id", "") or "").startswith("vision.")
    ]

    vision_linear_cfg = (
        cfg.get("quantization", {}).get("vision", {}).get("linear", {})
    )
    wrap_attn = bool(vision_linear_cfg.get("attn_proj", False))
    wrap_mlp = bool(vision_linear_cfg.get("mlp", False))
    exp_sites = VISION_LAYERS * (
        len(ATTN_OPS) * wrap_attn + len(MLP_OPS) * wrap_mlp
    )
    exp_files = exp_sites * 3
    print(
        f"variant: vision.linear(mlp={wrap_mlp}, attn_proj={wrap_attn}) "
        f"-> expect {exp_sites} sites / {exp_files} scale files"
    )

    errors: list[str] = []
    rows = []
    present_files = 0

    if len(vision) != exp_sites:
        errors.append(f"Vision Linear count is {len(vision)}, expected {exp_sites}")

    for m in sorted(vision, key=lambda x: getattr(x, "module_id", "")):
        mid = getattr(m, "module_id", "")
        name = getattr(m, "scale_group_name", "") or getattr(m, "layer_name", "")
        idx = getattr(m, "scale_group_idx", getattr(m, "layer_idx", -1))

        files = {
            role: scale_dir / f"{name}_{role}_scale_{idx}.p"
            for role in ("a", "w", "o")
        }

        missing = [role for role, path in files.items() if not path.exists()]
        invalid = []

        for role, path in files.items():
            if not path.exists():
                continue
            present_files += 1
            try:
                with path.open("rb") as f:
                    value = pickle.load(f)
                if not scale_is_valid(value):
                    invalid.append(role)
            except Exception as exc:
                invalid.append(f"{role}({type(exc).__name__})")

        rows.append((mid, name, idx, missing, invalid))
        if missing:
            errors.append(f"{mid}: missing scales {missing}")
        if invalid:
            errors.append(f"{mid}: invalid scales {invalid}")

    print("\n=== VISION CALIBRATION COVERAGE ===")
    print(f"scale_dir              : {scale_dir}")
    print(f"Vision Linear sites    : {len(vision)} / {exp_sites}")
    complete = sum(1 for _, _, _, missing, invalid in rows if not missing and not invalid)
    print(f"complete sites         : {complete} / {exp_sites}")
    print(f"Vision scale files     : {present_files} / {exp_files}")

    # Operator-family summary is useful for spotting one missing family.
    families = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    for op in families:
        op_rows = [r for r in rows if r[0].endswith("." + op)]
        op_ok = sum(1 for r in op_rows if not r[3] and not r[4])
        print(f"{op:10s}             : {op_ok:2d} / {len(op_rows)}")

    if complete != exp_sites:
        errors.append(f"complete Vision sites: {complete}/{exp_sites}")
    if present_files != exp_files:
        errors.append(f"Vision scale files: {present_files}/{exp_files}")

    if errors:
        print("\nCALIBRATION COVERAGE GATE: FAIL")
        for e in errors[:40]:
            print(f"  - {e}")
        if len(errors) > 40:
            print(f"  ... and {len(errors) - 40} more")
        raise SystemExit(2)

    print("\nCALIBRATION COVERAGE GATE: PASS")
    print(
        f"All {exp_sites} Vision Linear sites have finite positive A/W/O scales."
    )


if __name__ == "__main__":
    main()
