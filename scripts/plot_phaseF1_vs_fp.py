#!/usr/bin/env python
"""Grouped bar chart: FP16 baseline vs F1 (FP8-PoT + outlier, per_site).

Color style matches scripts/plot_object_quant_comparison.py:
  gray      = FP baseline
  tab:green = FP8(e4m3) PoT scale + outlier (ratio=0.01), per_site

Data sources (SR read from disk):
  - FP baseline: outputs/verify_libero/<suite>/eval_info.json (ep50 full run)
  - F1:          outputs/experiments/phaseF1_fp8pot_site_outlier_<suite>/result.json (ep10)

Note: FP baseline is the ep50 verify_libero protocol, F1 is ep10 — the FP
reference is therefore indicative only (on object, ep10 vs ep50 differ by
1.4pp: 93.8 vs 95.2).

Usage:
    python scripts/plot_phaseF1_vs_fp.py
Output:
    outputs/figures/phaseF1_vs_fp.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "outputs" / "experiments"
VERIFY = ROOT / "outputs" / "verify_libero"

C_FP = "gray"        # same style as plot_object_quant_comparison.py
C_F1 = "tab:green"

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABELS = ["Spatial", "Object", "Goal", "Libero-10"]


def read_sr(path):
    p = Path(path)
    if not p.exists():
        print(f"[warn] missing {p}")
        return np.nan
    return json.load(open(p))["overall"]["pc_success"]


def main():
    fp = [read_sr(VERIFY / s / "eval_info.json") for s in SUITES]
    f1 = [read_sr(EXP / f"phaseF1_fp8pot_site_outlier_{s}" / "result.json")
          for s in SUITES]

    deltas = [b - a if not (np.isnan(a) or np.isnan(b)) else np.nan
              for a, b in zip(fp, f1)]

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=150)
    x = np.arange(len(SUITES))
    width = 0.35

    bars_fp = ax.bar(x - width / 2, fp, width=width, label="FP16 baseline (verify_libero, ep50)",
                     color=C_FP, edgecolor="black", linewidth=0.5, hatch="//")
    bars_f1 = ax.bar(x + width / 2, f1, width=width,
                     label="F1: FP8(e4m3) PoT + outlier (per_site, ep10)",
                     color=C_F1, edgecolor="black", linewidth=0.5)

    for bars, vals, color in [(bars_fp, fp, "dimgray"), (bars_f1, f1, "tab:green")]:
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.annotate(f"{v:.1f}", xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=11, fontweight="bold", color=color)

    # delta annotation between the two bars of each suite
    for xi, d in zip(x, deltas):
        if np.isnan(d):
            continue
        sign = "+" if d >= 0 else "−"
        ax.annotate(f"{sign}{abs(d):.1f}", xy=(xi, 3),
                    ha="center", fontsize=9, color="tab:red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(SUITE_LABELS, fontsize=11)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("F1 (FP8 PoT + outlier, per_site) vs FP16 baseline across LIBERO suites\n"
                 "red = Δ (F1 − FP); FP reference is ep50 protocol, F1 is ep10 "
                 "(object ep10 vs ep50: 93.8 vs 95.2)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    out = ROOT / "outputs" / "figures" / "phaseF1_vs_fp.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
