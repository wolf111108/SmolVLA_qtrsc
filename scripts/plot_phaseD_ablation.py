#!/usr/bin/env python
"""Bar chart of Phase D granularity ablation results (D4).

Reads SR directly from outputs/experiments/phaseD_ablation_*/result.json.
Optional --with-w4 adds the w4 counterpart (phaseD_w4_ablation_*) for
comparison (default on if data exists).

Usage:
    python scripts/plot_phaseD_ablation.py            # FP8 + w4 (if present)
    python scripts/plot_phaseD_ablation.py --fp8-only # D4 alone
Output:
    outputs/figures/phaseD_ablation_bar.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "outputs" / "experiments"
FP_OBJ = 93.8  # FP baseline, libero_object ep10

GRAN = ["global", "per_component", "per_layer", "per_site"]
SCALE_FILES = {"global": 27, "per_component": 54,
               "per_layer": 432, "per_site": 864}
EP_TIME = {"global": "~208 s", "per_component": "~271 s",
           "per_layer": "~81 s", "per_site": "~94 s"}


def read_sr(prefix):
    vals = {}
    for g in GRAN:
        p = EXP / f"{prefix}_{g}" / "result.json"
        if p.exists():
            vals[g] = json.load(open(p))["overall"]["pc_success"]
        else:
            print(f"[warn] missing {p}")
            vals[g] = np.nan
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8-only", action="store_true",
                    help="plot only the FP8 (D4) series")
    args = ap.parse_args()

    fp8 = read_sr("phaseD_ablation")
    w4 = None if args.fp8_only else read_sr("phaseD_w4_ablation")

    n_series = 1 if w4 is None else 2
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)

    x = np.arange(len(GRAN))
    width = 0.38 if n_series == 2 else 0.55

    series = [("FP8 (pot_fp8_outlier)", fp8, "tab:blue")]
    if w4 is not None:
        series.append(("w4 weights (pot_ao_outlier)", w4, "tab:orange"))

    for i, (name, vals, color) in enumerate(series):
        offs = x + (i - (n_series - 1) / 2) * width
        v = [vals[g] for g in GRAN]
        bars = ax.bar(offs, v, width=width, label=name, color=color,
                      edgecolor="black", linewidth=0.5)
        for b, val in zip(bars, v):
            if np.isnan(val):
                continue
            ax.annotate(f"{val:.0f}", xy=(b.get_x() + b.get_width() / 2, val),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=11, fontweight="bold")

    # FP baseline reference
    ax.axhline(FP_OBJ, color="gray", ls="--", lw=1.2)
    ax.text(len(GRAN) - 0.5, FP_OBJ + 1, f"FP baseline {FP_OBJ}",
            fontsize=9, color="gray", ha="right")

    # ep10 noise band around FP baseline (+-5.7pp, 95% CI)
    ax.axhspan(FP_OBJ - 5.7, FP_OBJ + 5.7, color="orange", alpha=0.10)
    ax.text(-0.4, FP_OBJ + 6.2, "ep10 noise band (95% CI ±5.7pp)",
            fontsize=8, color="darkorange")

    labels = [f"{g}\n({SCALE_FILES[g]} scale files)\n{EP_TIME[g]}/ep"
              for g in GRAN]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 108)
    title = "Phase D4 — scale granularity ablation (libero_object, 10 tasks × 10 ep, seed=1000)"
    if w4 is not None:
        title += "\nw4 counterpart shown for contrast: per_layer/global collapse to 0% under int4"
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    out = ROOT / "outputs" / "figures" / "phaseD_ablation_bar.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
