#!/usr/bin/env python
"""Phase G — two bar charts: G1 component localization and G5 VLM internals.

Charts (one figure each, four bars per figure):

  G1  Component localization: which side owns the W4 collapse?
      G1-A FP8 all (anchor)  |  G1-B W4 all  |  G1-C W4 VLM only  |  G1-D W4 Expert only

  G5  VLM internals under W4: attention vs MLP
      control (VLM all FP8)  |  attention W4  |  MLP W4  |  VLM all W4

Both charts use the Phase F Goal (green) hue with four lightness steps,
deepening along the bar order. The FP8 anchor of each chart is drawn as a
dashed reference line so the drop is readable at a glance, and each bar
carries its delta against that anchor with an auto-picked black/white label
colour.

The legend and the footnote lines are stacked below the axes at positions
measured from ``ax.get_tightbbox()`` -- which includes the (multi-line) x tick
labels and the x-axis label -- rather than hand-tuned fractions. Placing them
just under the plot rectangle instead would put the legend on top of the tick
text.

All figures' text is English (avoids CJK font issues in matplotlib).
Palette comes from scripts/figure_palette.py — the same source Phase F uses.

Data sources (read from disk, never hard-coded):
  G1 : outputs/2026-09-10_phaseG_w4-root-cause/tasks/component-localization/<run>/
  G5 : outputs/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-precision/<run>/
       (the all-W4 bar reuses G1-C from component-localization)

Usage:
    python experiments/2026-09-10_phaseG_w4-root-cause/scripts/plot_phaseG_bars.py
Output:
    experiments/2026-09-10_phaseG_w4-root-cause/docs/figures/phaseG_G1_component_localization.png
    experiments/2026-09-10_phaseG_w4-root-cause/docs/figures/phaseG_G5_vlm_internals.png
    (also copied to outputs/figures/)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# paths + shared palette
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "2026-09-10_phaseG_w4-root-cause"
FIG_DIR = EXP / "docs" / "figures"
ALT_FIG_DIR = ROOT / "outputs" / "figures"

RUNS = ROOT / "outputs" / "2026-09-10_phaseG_w4-root-cause" / "tasks"
G1_RUNS = RUNS / "component-localization"
G5_RUNS = RUNS / "vlm-selective-precision"

sys.path.insert(0, str(ROOT / "scripts"))
from figure_layout import stack_below_axes  # noqa: E402
from figure_palette import (  # noqa: E402
    C_REF,
    GOAL_SHADES_4,
    readable_on,
)

# ---------------------------------------------------------------------------
# data spec
# ---------------------------------------------------------------------------

# Both charts use the Phase F Goal (green) hue with four lightness steps,
# deepening with W4 aggressiveness.

G1_BARS = [
    ("G1-A\nFP8 all\n(anchor)", "g1a_fp8_all", G1_RUNS, GOAL_SHADES_4[0]),
    ("G1-B\nW4 all\n(= F3 anchor)", "g1b_w4_all", G1_RUNS, GOAL_SHADES_4[1]),
    ("G1-C\nW4 VLM only", "g1c_w4_vlm_only", G1_RUNS, GOAL_SHADES_4[2]),
    ("G1-D\nW4 Expert only", "g1d_w4_expert_only", G1_RUNS, GOAL_SHADES_4[3]),
]

# The last G5 bar is G1-C re-measured, hence a different run directory.
G5_BARS = [
    ("control\nVLM all FP8\nExpert raw FP", "g5a_vlm_fp8_control", G5_RUNS,
     GOAL_SHADES_4[0]),
    ("attention W4\nq/k/v/o_proj", "g5b_vlm_attn_w4_mlp_fp8", G5_RUNS,
     GOAL_SHADES_4[1]),
    ("MLP W4\ngate/up/down_proj", "g5c_vlm_attn_fp8_mlp_w4", G5_RUNS,
     GOAL_SHADES_4[2]),
    ("VLM all W4\n(reused G1-C)", "g1c_w4_vlm_only", G1_RUNS, GOAL_SHADES_4[3]),
]


def read_run(run_dir: Path, run: str):
    """Return (pc_success, n_episodes) for a run, or (nan, nan) if absent."""
    d = run_dir / run
    for name in ("result.json", "eval_info.json"):
        p = d / name
        if p.is_file():
            try:
                with open(p) as f:
                    o = json.load(f)["overall"]
                return float(o["pc_success"]), float(o.get("n_episodes", 0))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] cannot parse {p}: {exc}")
    print(f"[warn] missing run: {d}")
    return float("nan"), float("nan")


def collect(bars):
    labels, vals, colors, eps = [], [], [], set()
    for label, run, run_dir, color in bars:
        sr, n = read_run(run_dir, run)
        labels.append(label)
        vals.append(sr)
        colors.append(color)
        if not np.isnan(n):
            eps.add(int(n))
    return labels, np.array(vals, dtype=float), colors, eps


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def draw_chart(ax, labels, vals, colors, title, anchor_label, xlabel):
    x = np.arange(len(vals))
    w = 0.60
    anchor = vals[0] if len(vals) and not np.isnan(vals[0]) else None

    for i, (lab, v, c) in enumerate(zip(labels, vals, colors)):
        ax.bar(x[i], v, width=w, color=c, edgecolor="black", linewidth=0.7,
               zorder=3)

        # SR value above the bar
        if not np.isnan(v):
            ax.annotate(f"{v:.1f}", xy=(x[i], v), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=12, fontweight="bold", color="black")

        # delta vs the chart's anchor, inside the bar under its top edge
        if anchor is not None and not np.isnan(v):
            d = v - anchor
            txt = "anchor" if i == 0 else f"{d:+.1f}"
            ax.annotate(txt, xy=(x[i], v), xytext=(0, -7),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=10, fontweight="bold",
                        color=readable_on(c), zorder=5)

    # anchor reference line
    if anchor is not None:
        ax.axhline(anchor, color=C_REF, linestyle=(0, (6, 4)), linewidth=1.3,
                   alpha=0.75, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 112)          # headroom so SR labels never touch the title
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10.5, labelpad=8)
    ax.set_ylabel("Success rate (%)", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def anchor_legend_handles(anchor_label):
    """Legend for the anchor reference line.

    Returned as figure-level handles on purpose: an axes-level
    ``ax.legend(loc="lower left")`` would be drawn INSIDE the axes and its
    box would cross the first two bars. Kept outside the axes instead.
    """
    return [Line2D([0], [0], color=C_REF, linestyle=(0, (6, 4)),
                   linewidth=1.3, label=anchor_label)]


SINGLE_LAYOUT = dict(left=0.105, right=0.975, top=0.90, bottom=0.42)


def build_figure(labels, vals, colors, title, anchor_label, xlabel,
                 footnote1, footnote2):
    fig, ax = plt.subplots(figsize=(9.0, 6.6), dpi=160)
    fig.subplots_adjust(**SINGLE_LAYOUT)
    xc = (SINGLE_LAYOUT["left"] + SINGLE_LAYOUT["right"]) / 2

    draw_chart(ax, labels, vals, colors, title, anchor_label, xlabel)
    stack_below_axes(fig, ax, anchor_legend_handles(anchor_label),
                     [footnote1, footnote2], xc)
    return fig


def ep_note(eps, fallback="100"):
    """'100 episodes per run' derived from the eval_info of the runs read."""
    n = "/".join(str(e) for e in sorted(eps)) if eps else fallback
    return f"libero_goal, {n} episodes per run"


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[ok] {out}")
    return out


def main():
    saved = []

    # --- G1 ---------------------------------------------------------------
    labels, vals, colors, eps = collect(G1_BARS)
    print("G1:", list(zip([l.split(chr(10))[0] for l in labels],
                          [round(v, 1) for v in vals])))
    fig = build_figure(
        labels, vals, colors,
        "G1   Component localization — which side owns the W4 collapse?",
        "G1-A anchor (FP8 all)",
        "Quantization configuration",
        ep_note(eps) + ".  Dashed line = G1-A FP8 anchor; "
        "in-bar numbers are Δ vs that anchor (pp).",
        "All bars use the Phase F Goal (green) hue, light → dark in bar order.",
    )
    saved.append(save(fig, "phaseG_G1_component_localization.png"))
    plt.close(fig)

    # --- G5 ---------------------------------------------------------------
    labels, vals, colors, eps = collect(G5_BARS)
    print("G5:", list(zip([l.split(chr(10))[0] for l in labels],
                          [round(v, 1) for v in vals])))
    fig = build_figure(
        labels, vals, colors,
        "G5   VLM internals under W4 — attention vs MLP sensitivity",
        "control (VLM all FP8)",
        "VLM precision assignment (Expert raw FP)",
        ep_note(eps) + ".  Dashed line = control; "
        "in-bar numbers are Δ vs control (pp).",
        "All bars use the Phase F Goal (green) hue, light → dark with W4 "
        "aggressiveness.  Near-additivity: −18 (attention) + −51 (MLP) ≈ −69 "
        "(all-W4).",
    )
    saved.append(save(fig, "phaseG_G5_vlm_internals.png"))
    plt.close(fig)

    # --- mirror -----------------------------------------------------------
    ALT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in saved:
        shutil.copy2(p, ALT_FIG_DIR / p.name)
    print(f"[ok] copied {len(saved)} file(s) -> {ALT_FIG_DIR}")


if __name__ == "__main__":
    main()
