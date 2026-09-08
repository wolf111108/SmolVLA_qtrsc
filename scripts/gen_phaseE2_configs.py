#!/usr/bin/env python
"""Generate Phase E2 closed-loop experiment configs from a base template.

Each config isolates ONE physical site (or a small selector set) with a noise
injection method + alpha, reusing the calibrated scales in
scales/phaseE_sensitivity (864 files, per_site granularity).

Run:  python scripts/gen_phaseE2_configs.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "configs/experiments/phaseE2_vlm3_down_a001.yaml"
OUT_DIR = REPO / "configs/experiments"

# (name, test-overrides dict)
# alpha sweep on the two most sensitive VLM sites + controls.
EXPERIMENTS = [
    # --- dose-response: top-1 site vlm.layers.3.mlp.down_proj ---
    ("phaseE2_vlm3_down_a003", {"alpha": 0.03}),
    ("phaseE2_vlm3_down_a010", {"alpha": 0.10}),
    # --- dose-response: top-2 site vlm.layers.0.mlp.down_proj ---
    ("phaseE2_vlm0_down_a001", {"target": {"module_id": "vlm.layers.0.mlp.down_proj"}, "alpha": 0.01}),
    ("phaseE2_vlm0_down_a003", {"target": {"module_id": "vlm.layers.0.mlp.down_proj"}, "alpha": 0.03}),
    ("phaseE2_vlm0_down_a010", {"target": {"module_id": "vlm.layers.0.mlp.down_proj"}, "alpha": 0.10}),
    # --- expert-side top site ---
    ("phaseE2_exp1_up_a003", {"target": {"module_id": "expert.layers.1.mlp.up_proj"}, "alpha": 0.03}),
    ("phaseE2_exp1_up_a010", {"target": {"module_id": "expert.layers.1.mlp.up_proj"}, "alpha": 0.10}),
    # --- controls: mid/low-sensitivity sites ---
    ("phaseE2_exp7qk_a003", {"target": {"module_id": "expert.layer.7.qk"}, "alpha": 0.03}),
    ("phaseE2_exp7qk_a030", {"target": {"module_id": "expert.layer.7.qk"}, "alpha": 0.30}),
    # --- quant_residual (reuses calibrated scales) on a control MatMul ---
    ("phaseE2_exp7qk_qr", {"target": {"module_id": "expert.layer.7.qk"}, "method": "quant_residual_output"}),
    # --- a MatMul on the vlm side (nonzero-cache path) ---
    ("phaseE2_vlm3qk_a003", {"target": {"module_id": "vlm.layer.3.qk"}, "alpha": 0.03}),
    ("phaseE2_vlm3qk_a010", {"target": {"module_id": "vlm.layer.3.qk"}, "alpha": 0.10}),
    # --- baseline: no injection at all (all raw) ---
    ("phaseE2_baseline_raw", {"enabled": False}),
]


def main() -> None:
    base = yaml.safe_load(BASE.read_text())

    for name, overrides in EXPERIMENTS:
        cfg = copy.deepcopy(base)
        cfg["output_dir"] = f"outputs/experiments/{name}"

        test = cfg.get("test", {})
        for k, v in overrides.items():
            test[k] = v
        cfg["test"] = test

        # describe in header comment
        tgt = test.get("target", {}).get("module_id", "-")
        desc = (
            f"{name}: target={tgt} method={test.get('method')} "
            f"alpha={test.get('alpha')} enabled={test.get('enabled')}"
        )

        path = OUT_DIR / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# Phase E2 auto-generated: {desc}\n")
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        print(f"wrote {path.name}: {desc}")


if __name__ == "__main__":
    main()
