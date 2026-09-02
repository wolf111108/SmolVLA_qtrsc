#!/usr/bin/env python
"""Grouped bar chart of the n_action_steps / n_episodes ablation matrix.

Four groups (P0..P3) on the x-axis; each group has 4 suite bars
(Spatial / Object / Goal / Long) plus a black diamond line marking the
4-suite average. All text in English.

Data source: doc/logs/2026-09-01_h100.md §3 (P0 from 08-26 gpupro6000d
log, P1/P2/P3 from ablation runs na10_ep10 / na1_ep50 / mj332 rerun).

Usage:
    python scripts/plot_ablation_matrix.py
Output:
    outputs/figures/ablation_matrix_bar.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Data (success rates, %)
# ----------------------------------------------------------------------------

groups = [
    "P0\n(na=1, ep=10)",
    "P1\n(na=1, ep=50)",
    "P2\n(na=10, ep=10)",
    "P3\n(na=10, ep=50)",
]

suites = ["Spatial", "Object", "Goal", "Long"]
suite_colors = ["tab:green", "tab:orange", "tab:purple", "tab:cyan"]

sr = {
    "Spatial": [81.0, 85.2, 85.0, 86.0],
    "Object":  [78.0, 78.4, 95.0, 93.8],
    "Goal":    [77.0, 78.4, 89.0, 87.8],
    "Long":    [60.0, 55.8, 69.0, 74.2],
}
avg = [74.0, 74.45, 84.5, 85.45]

# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=150)

n_groups = len(groups)
n_bars = len(suites)
x = np.arange(n_groups)
width = 0.19

for i, suite in enumerate(suites):
    offsets = x + (i - (n_bars - 1) / 2) * width
    bars = ax.bar(
        offsets,
        sr[suite],
        width=width,
        color=suite_colors[i],
        label=suite,
        edgecolor="black",
        linewidth=0.4,
    )
    for b, v in zip(bars, sr[suite]):
        ax.annotate(
            f"{v:.0f}" if v == int(v) else f"{v:.1f}",
            xy=(b.get_x() + b.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

# Average as black diamond markers connected by a line
ax.plot(
    x,
    avg,
    color="black",
    marker="D",
    markersize=7,
    linewidth=1.8,
    label="Average",
    zorder=5,
)
for xi, v in zip(x, avg):
    ax.annotate(
        f"{v:.2f}",
        xy=(xi, v),
        xytext=(0, -14),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="black",
    )

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11)
ax.set_xlabel("Ablation condition (na = n_action_steps, ep = n_episodes)", fontsize=12)
ax.set_ylabel("Success rate (%)", fontsize=12)
ax.set_title("Ablation: n_action_steps vs n_episodes (LIBERO, 4 suites)", fontsize=13)
ax.set_ylim(0, 108)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(fontsize=10, loc="upper left", ncol=5, columnspacing=1.2)

fig.tight_layout()

out = Path("outputs/figures/ablation_matrix_bar.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"saved: {out}")
