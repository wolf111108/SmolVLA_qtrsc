#!/usr/bin/env python
"""Phase E dashboard: 4 intuitive panels for sensitivity experiment results.

Data from docs/phaseE_worklog.md + doc/logs/2026-09-08_new_experiment_summary.md
§6 (E1 top-10 / aggregations verified there; E2 full 14-run table).
The full 288-row sensitivity.csv lives on gpupro6000d and is not synced here —
when it is, panel (a) can be regenerated from disk.

Panels:
  (a) E1 top-10 forward sensitivity (log scale) — the 5.7× cliff at rank 1
  (b) E2 dose-response curves — SR vs noise level α per target (flat lines)
  (c) forward rank vs closed-loop impact — no correlation

Usage:
    python scripts/plot_phaseE_dashboard.py
Output:
    outputs/figures/phaseE_dashboard.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Data (from phaseE_worklog.md, E1 sweep-all + E2 closed-loop table)
# ----------------------------------------------------------------------------
E1_TOP10 = [
    ("vlm.layers.3.mlp.down_proj", 1.133e00, 3.85e-01),
    ("vlm.layers.0.mlp.down_proj", 1.973e-01, 4.66e-02),
    ("vlm.layers.1.self_attn.k_proj", 1.094e-01, 2.98e-02),
    ("vlm.layers.1.mlp.down_proj", 1.016e-01, 2.32e-02),
    ("vlm.layers.0.mlp.gate_proj", 9.375e-02, 2.32e-02),
    ("vlm.layers.2.self_attn.o_proj", 9.375e-02, 1.70e-02),
    ("vlm.layers.3.mlp.up_proj", 9.375e-02, 1.98e-02),
    ("vlm.layer.1.qk", 9.375e-02, 2.0e-02),
    ("vlm.layer.2.qk", 9.375e-02, 2.0e-02),
    ("vlm.layer.3.qk", 9.375e-02, 2.0e-02),
]

BASELINE = 94.0  # baseline_raw closed-loop SR

# E2 dose-response: target -> [(alpha label, alpha value, SR)]
DOSE = {
    "vlm3_down (rank 1)": [(0.01, 0.01, 91), (0.03, 0.03, 92), (0.10, 0.10, 89)],
    "vlm0_down (rank 2)": [(0.01, 0.01, 93), (0.03, 0.03, 92), (0.10, 0.10, 93)],
    "exp1_up (expert top)": [(0.03, 0.03, 92), (0.10, 0.10, 92)],
    "exp7qk (rank 156 ctrl)": [(0.03, 0.03, 89), (0.30, 0.30, 92)],
    "vlm3qk (VLM matmul)": [(0.03, 0.03, 92), (0.10, 0.10, 92)],
}
DOSE_COLORS = {
    "vlm3_down (rank 1)": "tab:red",
    "vlm0_down (rank 2)": "tab:blue",
    "exp1_up (expert top)": "tab:orange",
    "exp7qk (rank 156 ctrl)": "tab:green",
    "vlm3qk (VLM matmul)": "tab:purple",
}

# (rank, max closed-loop drop pp, label) for panel (c)
RANK_VS_DROP = [
    (1, 5, "vlm3_down"),
    (2, 1, "vlm0_down"),
    (156, 5, "exp7qk"),
]

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)

# ----------------------------------------------------------------------------
# (a) E1 top-10 forward sensitivity — log-scale horizontal bars
# ----------------------------------------------------------------------------
ax = axes[0]
names = [n for n, _, _ in E1_TOP10][::-1]
vals = [v for _, v, _ in E1_TOP10][::-1]
colors = ["tab:red" if v > 0.5 else "tab:blue" for v in vals]
ax.barh(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7)
ax.set_xscale("log")
ax.set_xlabel("max_abs_diff (log scale)")
ax.set_title("(a) E1 forward sensitivity — top 10 of 288 sites\n"
             "rank 1 is 5.7× rank 2 (red); 7 architectural zero sites at vlm.layer.15",
             fontsize=10)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.3g}", xy=(v, i), xytext=(3, 0),
                textcoords="offset points", va="center", fontsize=7)

# ----------------------------------------------------------------------------
# (b) E2 dose-response curves
# ----------------------------------------------------------------------------
ax = axes[1]
ax.axhline(BASELINE, color="gray", ls="--", lw=1.5)
ax.text(0.009, BASELINE + 0.3, "baseline_raw 94%", fontsize=8, color="gray")
ax.axhspan(BASELINE - 3, BASELINE + 3, color="orange", alpha=0.12)
for name, pts in DOSE.items():
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    ax.plot(xs, ys, "o-", color=DOSE_COLORS[name], lw=1.8, ms=7, label=name)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y}", xy=(x, y), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=DOSE_COLORS[name])
ax.set_xscale("log")
ax.set_xticks([0.01, 0.03, 0.10, 0.30])
ax.set_xticklabels(["0.01", "0.03", "0.10", "0.30"])
ax.set_xlabel("noise level α (log scale)")
ax.set_ylabel("closed-loop SR (%)")
ax.set_ylim(85, 97)
ax.set_title("(b) E2 dose-response — flat: no systematic degradation with α\n"
             "shaded = ±3pp noise band", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left")

# ----------------------------------------------------------------------------
# (c) forward rank vs closed-loop impact
# ----------------------------------------------------------------------------
ax = axes[2]
ranks = [r for r, _, _ in RANK_VS_DROP]
drops = [d for _, d, _ in RANK_VS_DROP]
labels_c = [l for _, _, l in RANK_VS_DROP]
ax.scatter(ranks, drops, s=140, c=["tab:red", "tab:blue", "tab:green"],
           edgecolor="black", linewidth=0.8, zorder=3)
for r, d, l in RANK_VS_DROP:
    ax.annotate(f"{l}\n(rank {r}, −{d}pp)", xy=(r, d), xytext=(8, 6),
                textcoords="offset points", fontsize=8.5)
ax.set_xscale("log")
ax.set_xticks([1, 2, 156])
ax.set_xticklabels(["1", "2", "156"])
ax.set_xlim(0.6, 500)
ax.set_xlabel("E1 forward-sensitivity rank (log scale, 1 = most sensitive)")
ax.set_ylabel("max closed-loop SR drop (pp)")
ax.set_ylim(-1, 7)
ax.set_title("(c) Forward rank does NOT predict closed-loop impact\n"
             "rank 1 and rank 156 both drop only 5pp", fontsize=10)

fig.suptitle("Phase E sensitivity dashboard — single-site noise injection: "
             "highly robust in closed loop\n"
             "(E1: forward sweep on gpupro6000d; E2: libero_object 100 ep per run)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = OUT / "phaseE_dashboard.png"
fig.savefig(out)
print(f"saved {out}")
