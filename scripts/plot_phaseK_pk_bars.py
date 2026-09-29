#!/usr/bin/env python
"""Phase K — bar charts for the mixed-precision (W8/W8/W4) results.

Two figures, both following the Phase F / Phase G house style:

  Figure 1  phaseK_PK_foursuite_bars.png   (Phase F grouped-bar style)
      x = the four LIBERO suites; one gray hatched bar per suite is the FP
      baseline, one suite-hued bar is the PK quantized result.  Covers both
      PK-full (Spatial / Object / LIBERO-10) and PK-Goal (Goal) in one chart.

  Figure 2  phaseK_PK_goal_ladder.png      (Phase G single-hue ramp style)
      Goal-only precision ladder: raw FP anchor -> W8/W8/W4 (PK-Goal), with
      the anchor drawn as a dashed reference line and in-bar deltas.

Palette comes from scripts/figure_palette.py (the same source Phase F and
Phase G use), so this figure family reads as part of the same set.  Layout
helpers come from scripts/figure_layout.py.

All figure text is English (avoids CJK font issues in matplotlib).

Data sources (read from disk, never hard-coded):
  PK-full quant      : outputs/2026-09-25_phaseK_mixed-full-libero/<suite>/quant/result.json
  PK-Goal quant      : experiments/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/docs/result_quant.json
  PK-Goal raw anchor : experiments/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/docs/result_baseline.json
  FP baseline        : experiments/2026-09-25_phaseK_mixed-full-libero/docs/fp_baseline_na10_ep10_task_success.json

Usage:
    python scripts/plot_phaseK_pk_bars.py
Output:
    outputs/figures/phaseK_PK_foursuite_bars.png
    outputs/figures/phaseK_PK_goal_ladder.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from figure_layout import stack_below_axes  # noqa: E402
from figure_palette import (  # noqa: E402
    C_FP,
    C_NEG,
    C_POS,
    C_REF,
    GOAL_SHADES_4,
    SUITE_LABELS,
    SUITE_SHADES,
    readable_on,
)

PK_FULL = ROOT / "outputs" / "2026-09-25_phaseK_mixed-full-libero"
PK_FULL_DOCS = ROOT / "experiments" / "2026-09-25_phaseK_mixed-full-libero" / "docs"
PK_GOAL_DOCS = (
    ROOT / "experiments" / "2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal" / "docs"
)
FP_JSON = PK_FULL_DOCS / "fp_baseline_na10_ep10_task_success.json"
FIG_DIR_PK = PK_FULL_DOCS / "figures"
FIG_DIR = ROOT / "outputs" / "figures"

# suite label -> (PK-full result.json dir, FP baseline json key)
SUITES = [
    ("Spatial",   "libero_spatial", "libero_spatial"),
    ("Object",    "libero_object",  "libero_object"),
    ("Goal",      None,             "libero_goal"),      # from PK-Goal
    ("LIBERO-10", "libero_10",      "libero_10"),
]

PK_LABEL = "PK  Vision W8 / VLM W8 / Expert W4  (AFP8 PoT)"


def read_sr(path: Path) -> float:
    with path.open() as f:
        return float(json.load(f)["overall"]["pc_success"])


def collect_foursuite():
    fp_json = json.loads(FP_JSON.read_text())
    fp_totals = fp_json["suite_totals"]
    goal_quant = read_sr(PK_GOAL_DOCS / "result_quant.json")

    labels, fp, pk = [], [], []
    for label, run_dir, fp_key in SUITES:
        labels.append(label)
        fp.append(float(fp_totals[fp_key]["sr"]))
        if run_dir is None:
            pk.append(goal_quant)
        else:
            pk.append(read_sr(PK_FULL / run_dir / "quant" / "result.json"))
    return labels, np.array(fp), np.array(pk)


# ---------------------------------------------------------------------------
# Figure 1 — Phase F grouped-bar style
# ---------------------------------------------------------------------------

def draw_grouped(ax, labels, fp, pk, shade_idx=1):
    x = np.arange(len(labels))
    w = 0.38

    for i, lab in enumerate(labels):
        color = SUITE_SHADES[lab][shade_idx]

        ax.bar(x[i] - w / 2, fp[i], width=w, color=C_FP, edgecolor="black",
               linewidth=0.6, hatch="//", zorder=3)
        ax.bar(x[i] + w / 2, pk[i], width=w, color=color, edgecolor="black",
               linewidth=0.6, zorder=3)

        ax.annotate(f"{fp[i]:.1f}", xy=(x[i] - w / 2, fp[i]), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9.5, color="dimgray")
        ax.annotate(f"{pk[i]:.1f}", xy=(x[i] + w / 2, pk[i]), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="black")

        d = pk[i] - fp[i]
        ax.annotate(f"{d:+.1f}", xy=(x[i] + w / 2, pk[i]), xytext=(0, -5),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=9.5, fontweight="bold",
                    color=readable_on(color), zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 116)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Phase K — LIBERO success rate: FP baseline vs mixed-precision PTQ",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("LIBERO suite", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Success rate (%)", fontsize=10.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=9.5)


def build_foursuite():
    labels, fp, pk = collect_foursuite()
    fig, ax = plt.subplots(figsize=(9.4, 7.6), dpi=160)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.915, bottom=0.34)
    xc = (0.105 + 0.975) / 2

    draw_grouped(ax, labels, fp, pk)

    handles = [Patch(facecolor=C_FP, edgecolor="black", hatch="//",
                     label="FP baseline (mj332_A_na10_ep10, cross-round)")]
    handles += [Patch(facecolor=SUITE_SHADES[l][1], edgecolor="black",
                      label=l) for l in labels]

    # --- stack legend -> mean rows -> footnotes, every position MEASURED ----
    # Hard-coded fractions here would silently collide with the legend, which
    # is placed relative to the axes' tight bbox (tick labels + xlabel).
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    gap_legend, gap_row, gap_note = 0.026, 0.015, 0.013

    y = ax.get_tightbbox(r).transformed(inv).y0 - gap_legend
    legend = fig.legend(handles=handles, loc="upper center", ncol=5,
                        bbox_to_anchor=(xc, y), fontsize=9.0, frameon=True,
                        framealpha=0.95, borderpad=0.6, handlelength=1.5,
                        columnspacing=1.2)
    fig.canvas.draw()
    y = legend.get_window_extent(r).transformed(inv).y0 - gap_row

    def mean_row(y, tag, a, b):
        """One 'tag   FP x%  ->  PK y%   (Δ pp)' row; returns the next free y."""
        d = b - a
        t1 = fig.text(xc, y, f"{tag}     FP {a:.1f}%    →    PK {b:.1f}%   ",
                      ha="right", va="top", fontsize=9.4, color="dimgray")
        t2 = fig.text(xc, y, f"({d:+.1f} pp)", ha="left", va="top",
                      fontsize=10.4, fontweight="bold",
                      color=C_POS if d >= 0 else C_NEG)
        fig.canvas.draw()
        bottoms = [t.get_window_extent(r).transformed(inv).y0 for t in (t1, t2)]
        return min(bottoms) - gap_row

    m_fp_all, m_pk_all = fp.mean(), pk.mean()
    same = [0, 1, 3]  # Spatial / Object / LIBERO-10 -> the same-round three
    y = mean_row(y, "All four suites", m_fp_all, m_pk_all)
    y = mean_row(y, "Same-round 3 suites", fp[same].mean(), pk[same].mean())

    notes = [
        "Spatial / Object / LIBERO-10 from PK-full (10 episodes/task); Goal from "
        "PK-Goal (10 episodes/task, same config, different run).",
        "Bar hue identifies the suite; the hatched gray bar is the FP baseline. "
        "In-bar numbers are Δ vs that baseline (pp).",
        "Baseline caveat: mj332_A_na10_ep10 has pretrained_revision=None "
        "(checkpoint version not pinned), so deltas carry cross-round drift.",
    ]
    y -= gap_note
    for note in notes:
        t = fig.text(xc, y, note, ha="center", va="top", fontsize=8.2,
                     color="gray")
        fig.canvas.draw()
        y = t.get_window_extent(r).transformed(inv).y0 - gap_note
    if y < 0.01:
        print(f"[warn] footnotes run off the figure (last y={y:.3f})")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Phase G single-hue ramp style (Goal ladder)
# ---------------------------------------------------------------------------

def read_run(path: Path):
    with path.open() as f:
        o = json.load(f)["overall"]
    return float(o["pc_success"]), int(o.get("n_episodes", 0))


def build_goal_ladder():
    raw, _ = read_run(PK_GOAL_DOCS / "result_baseline.json")
    w8, ep = read_run(PK_GOAL_DOCS / "result_quant.json")

    bars = [
        ("raw FP eager\n(in-house anchor)", raw, GOAL_SHADES_4[0]),
        ("Vision W8 / VLM W8\nExpert W4  (AFP8 PoT)", w8, GOAL_SHADES_4[2]),
    ]
    labels = [b[0] for b in bars]
    vals = np.array([b[1] for b in bars], dtype=float)
    colors = [b[2] for b in bars]

    fig, ax = plt.subplots(figsize=(7.6, 7.0), dpi=160)
    fig.subplots_adjust(left=0.13, right=0.965, top=0.91, bottom=0.40)
    xc = (0.13 + 0.965) / 2

    x = np.arange(len(vals))
    anchor = vals[0]
    for i, (v, c) in enumerate(zip(vals, colors)):
        ax.bar(x[i], v, width=0.5, color=c, edgecolor="black", linewidth=0.7,
               zorder=3)
        ax.annotate(f"{v:.1f}", xy=(x[i], v), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="black")
        d = v - anchor
        txt = "anchor" if i == 0 else f"{d:+.1f}"
        ax.annotate(txt, xy=(x[i], v), xytext=(0, -8), textcoords="offset points",
                    ha="center", va="top", fontsize=10, fontweight="bold",
                    color=readable_on(c), zorder=5)

    ax.axhline(anchor, color=C_REF, linestyle=(0, (6, 4)), linewidth=1.3,
               alpha=0.75, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Phase K — Goal precision ladder (Vision/VLM/Expert weights)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Weight precision assignment", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Success rate (%)", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [Line2D([0], [0], color=C_REF, linestyle=(0, (6, 4)),
                      linewidth=1.3, label="raw FP anchor (in-house)")]
    notes = [
        "libero_goal, 10 tasks x 10 episodes per run (seed 1000); "
        "dashed line = raw FP anchor.",
        "In-bar numbers are Δ vs that anchor (pp).  Both bars use the Phase F "
        "Goal (green) hue, light → dark.",
        "Not shown: the W4/W8/W4 variant (Vision W4) is still running "
        "(no final SR yet).",
    ]
    stack_below_axes(fig, ax, handles, notes, xc)
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR_PK.mkdir(parents=True, exist_ok=True)
    saved = []

    labels, fp, pk = collect_foursuite()
    print("four-suite FP:", fp.tolist())
    print("four-suite PK:", pk.tolist())
    print("  deltas      :", (pk - fp).tolist())
    print(f"  mean FP {fp.mean():.1f} -> PK {pk.mean():.1f} "
          f"({pk.mean() - fp.mean():+.1f} pp)")
    same = [0, 1, 3]
    print(f"  same-round 3: FP {fp[same].mean():.1f} -> PK {pk[same].mean():.1f} "
          f"({pk[same].mean() - fp[same].mean():+.1f} pp)")

    fig = build_foursuite()
    fig.savefig(FIG_DIR / "phaseK_PK_foursuite_bars.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    saved.append(FIG_DIR / "phaseK_PK_foursuite_bars.png")
    plt.close(fig)

    raw, _ = read_run(PK_GOAL_DOCS / "result_baseline.json")
    w8, _ = read_run(PK_GOAL_DOCS / "result_quant.json")
    print(f"goal ladder : raw FP {raw:.1f} -> W8/W8/W4 {w8:.1f} ({w8 - raw:+.1f} pp)")

    fig = build_goal_ladder()
    fig.savefig(FIG_DIR / "phaseK_PK_goal_ladder.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    saved.append(FIG_DIR / "phaseK_PK_goal_ladder.png")
    plt.close(fig)

    for p in saved:
        print(f"[ok] {p}")

    import shutil
    for p in saved:
        shutil.copy2(p, FIG_DIR_PK / p.name)
    print(f"[ok] copied {len(saved)} file(s) -> {FIG_DIR_PK}")


if __name__ == "__main__":
    main()
