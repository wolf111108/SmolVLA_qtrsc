#!/usr/bin/env python
"""Generate Phase E3 configs: component-wise (vlm vs expert) noise tolerance.

Full 5x5 alpha grid on both components simultaneously:
    alpha_vlm    in {0.01, 0.05, 0.10, 0.20, 0.50}
    alpha_expert in {0.01, 0.05, 0.10, 0.20, 0.50}

25 experiments. Each injects gaussian_rms_output noise into ALL vlm.* modules
with alpha_vlm and ALL expert.* modules with alpha_expert, then evaluates
closed-loop SR on libero_object (10 episodes).

Alpha encoding in names: percent *100, e.g. a50 == 0.50.

Run:  python scripts/gen_phaseE3_configs.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "configs/experiments/phaseE2_vlm3_down_a001.yaml"
OUT_DIR = REPO / "configs/experiments"

ALPHAS = [0.01, 0.05, 0.10, 0.20, 0.50]


def a_tag(a: float) -> str:
    return f"{int(round(a * 100)):02d}"


def main() -> None:
    base = yaml.safe_load(BASE.read_text())

    n = 0
    for a_vlm in ALPHAS:
        for a_exp in ALPHAS:
            name = f"phaseE3_v{a_tag(a_vlm)}_e{a_tag(a_exp)}"
            cfg = copy.deepcopy(base)
            cfg["output_dir"] = f"outputs/experiments/{name}"

            cfg["test"] = {
                "enabled": True,
                "method": "gaussian_rms_output",
                "seed": 0,
                "site": "output",
                "use_outlier_protection": False,
                "residual_lambda": 1.0,
                "outlier_ratio": 0.01,
                "targets": [
                    {"target": {"component": "vlm"}, "alpha": a_vlm},
                    {"target": {"component": "expert"}, "alpha": a_exp},
                ],
            }

            desc = (
                f"Phase E3: vlm alpha={a_vlm}, expert alpha={a_exp}, "
                f"method=gaussian_rms_output (all 288 sites injected)"
            )
            path = OUT_DIR / f"{name}.yaml"
            with path.open("w", encoding="utf-8") as f:
                f.write(f"# {desc}\n")
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
            print(f"wrote {path.name}")
            n += 1

    print(f"\ntotal: {n} configs")


if __name__ == "__main__":
    main()
