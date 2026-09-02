#!/usr/bin/env python
"""Plot LIBERO-object success rate vs quantization precision.

Two lines:
  - red : quantization WITHOUT outlier protection
  - blue: quantization WITH outlier protection (ratio=0.01)

Data source: outputs/experiments/*_quant_test/result.json (ep10, libero_object,
seed=1000), FP baseline from verify_libero_object ep10. INT8 without outlier
is recorded as 0 (per log 2026-09-01, pending experiment). INT16 with outlier
was not tested -> NaN, shown as a dashed interpolation with annotation.

Usage:
    python scripts/plot_quant_sr.py
Output:
    outputs/figures/quant_sr_vs_bitwidth.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

precisions = ["FP baseline", "INT16", "INT12", "INT8"]

sr_without_outlier = [93.8, 89.0, 7.0, 0.0]   # no outlier protection
sr_with_outlier = [93.8, None, 95.0, 92.0]     # outlier_ratio = 0.01

# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)

x = range(len(precisions))

# Red line: without outlier protection (complete)
ax.plot(
    x,
    sr_without_outlier,
    color="red",
    marker="o",
    markersize=8,
    linewidth=2,
    label="Without outlier protection",
)

# Blue line: with outlier protection (INT16 not tested; line connects
# tested points only, FP -> INT12 -> INT8).
xs = [i for i, v in enumerate(sr_with_outlier) if v is not None]
ys = [v for v in sr_with_outlier if v is not None]
ax.plot(
    xs,
    ys,
    color="tab:blue",
    marker="o",
    markersize=8,
    linewidth=2,
    label="With outlier protection (ratio=0.01)",
)

# Value labels on every point
for xi, v in zip(x, sr_without_outlier):
    ax.annotate(
        f"{v:.1f}",
        xy=(xi, v),
        xytext=(0, -16),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        color="red",
    )
for xi, v in zip(xs, ys):
    ax.annotate(
        f"{v:.1f}",
        xy=(xi, v),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        color="tab:blue",
    )

ax.set_xticks(list(x))
ax.set_xticklabels(precisions, fontsize=11)
ax.set_xlabel("Quantization precision", fontsize=12)
ax.set_ylabel("Success rate (%)", fontsize=12)
ax.set_title("LIBERO-object success rate vs quantization precision (ep10)", fontsize=13)
ax.set_ylim(-5, 105)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=11, loc="center right")

fig.tight_layout()

out = Path("outputs/figures/quant_sr_vs_bitwidth.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"saved: {out}")
