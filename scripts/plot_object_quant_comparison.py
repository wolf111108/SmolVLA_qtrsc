#!/usr/bin/env python
"""Single bar chart: quantization configs on libero_object (ep10, seed=1000).

Configs compared (SR read from outputs/experiments/*/result.json):
  - FP16 baseline (verify_libero_object ep10, hardcoded 93.8)
  - INT16 (per-tensor, no outlier)
  - INT12 + outlier (ratio=0.01)
  - INT8  + outlier (ratio=0.01)
  - FP8(e4m3) PoT scale + outlier, per_site  (phaseD_ablation_per_site)
  - w4 weights (pot_ao_outlier), per_site    (phaseD_w4_ablation_per_site)

Usage:
    python scripts/plot_object_quant_comparison.py
Output:
    outputs/figures/object_quant_comparison.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "outputs" / "experiments"
FP_OBJ = 93.8  # FP baseline, verify_libero_object ep10

# (label, result.json dir, color, note)
SERIES = [
    ("FP16\nbaseline", None, "gray", "FP"),
    ("INT16", "smolvla_int16_quant_test", "tab:blue", "16/16/16"),
    ("INT12\n+outlier", "smolvla_int12_outlier_quant_test", "tab:blue", "12/12/12"),
    ("INT8\n+outlier", "smolvla_int8_outlier_quant_test", "tab:blue", "8/8/8"),
    ("FP8\nPoT+outlier\nper_site", "phaseD_ablation_per_site", "tab:green", "a/w/o "),
    ("FP8 w4 weights\nper_site", "phaseD_w4_ablation_per_site", "tab:orange", "a/o, w=4bit"),
]


def read_sr(dirname):
    p = EXP / dirname / "result.json"
    if not p.exists():
        print(f"[warn] missing {p}")
        return np.nan
    return json.load(open(p))["overall"]["pc_success"]


def main():
    labels, vals, colors = [], [], []
    tick_labels = []
    for label, d, color, note in SERIES:
        labels.append(label)
        vals.append(FP_OBJ if d is None else read_sr(d))
        colors.append(color)
        tick_labels.append(f"{label}\n{note}")

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.6, color=colors, edgecolor="black", linewidth=0.5)

    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=12, fontweight="bold")

    ax.axhline(FP_OBJ, color="gray", ls="--", lw=1.2)
    ax.text(len(labels) - 0.55, FP_OBJ + 1.2, f"FP baseline {FP_OBJ}",
            fontsize=9, color="gray", ha="right")
    ax.axhspan(FP_OBJ - 5.7, FP_OBJ + 5.7, color="orange", alpha=0.10)
    ax.text(-0.45, FP_OBJ + 6.2, "ep10 noise band (95% CI ±5.7pp)",
            fontsize=8, color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9.5)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Quantization configs on libero_object (10 tasks × 10 ep, seed=1000)\n"
                 "all use calibration HuggingFaceVLA/libero v3.0 (8 ep, stride 4, seed 42); "
                 "outlier_ratio=0.01 where marked")
    ax.grid(axis="y", alpha=0.3)

    out = ROOT / "outputs" / "figures" / "object_quant_comparison.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
