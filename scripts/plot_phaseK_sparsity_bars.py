#!/usr/bin/env python
"""Phase K — sparsity bar charts (activation vs static weight).

Two figures, both in the Phase F grouped-bar house style:

  Figure 1  phaseK_sparsity_activation_bars.png
      x = the four LIBERO suites; three bars per suite = Encoder / VLM /
      Expert.  Bar height = runtime S|MMM **bit** sparsity of the Linear
      *activation* role; a second small label under it gives element sparsity,
      because the two metrics differ by two orders of magnitude for VLM/Expert
      and a single scale cannot show both.

  Figure 2  phaseK_sparsity_weight_bars.png
      Same layout for the **static weight** sparsity (sign-aware, INT8 for
      Encoder/VLM and INT4 for Expert).  Identical across suites by
      construction, which the figure makes visible at a glance.

Palette comes from scripts/figure_palette.py -- the same source Phase F and
Phase G use, so this figure family reads as part of the same set.  Here the
hue is fixed PER COMPONENT (the transpose of Phase F's per-suite hue) because
the component must be identifiable across the whole chart.
Layout helpers come from scripts/figure_layout.py.

All figure text is English (avoids CJK font issues in matplotlib).

Metric definitions (do not mix them):
  * runtime native counting  -> excludes zeros manufactured by the FP outlier
    side path (``*_native`` columns).
  * S|MMM v1 bit metric      -> E4M3 raw ``S EEEE MMM`` counts only ``S`` and
    ``MMM``; exponent and the hidden leading 1 are excluded.
  * static weight            -> sign-aware integer metric; INT8 and INT4 are
    reported separately, never pooled into one number.

Data sources (read from disk, never hard-coded).  Spatial / Object /
LIBERO-10 are this round's PK-full runs; Goal reuses the predecessor
single-suite experiment because the Goal suite was skipped this round, so its
row is a marked cross-round reference:
  outputs/2026-09-25_phaseK_mixed-full-libero/<suite>/quant/sparsity/
      module_sparsity.csv          (activation, all four suites except Goal)
      weight_sparsity_static.csv   (weight,    all four suites except Goal)
  experiments/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/docs/
      module_sparsity_quant.csv / weight_sparsity_static_quant.csv   (Goal)

Usage:
    python scripts/plot_phaseK_sparsity_bars.py
Output:
    outputs/figures/phaseK_sparsity_activation_bars.png
    outputs/figures/phaseK_sparsity_weight_bars.png
    (also copied to experiments/2026-09-25_phaseK_mixed-full-libero/docs/figures/)
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from figure_palette import COMPONENT_SHADES, readable_on  # noqa: E402

PK = ROOT / "outputs" / "2026-09-25_phaseK_mixed-full-libero"
PK_DOCS = ROOT / "experiments" / "2026-09-25_phaseK_mixed-full-libero" / "docs"
GOAL_DOCS = (
    ROOT / "experiments" / "2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal" / "docs"
)
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR_PK = PK_DOCS / "figures"

# (chart label, module_sparsity.csv, weight_sparsity_static.csv, cross_round)
SUITES = [
    ("Spatial",   PK / "libero_spatial/quant/sparsity/module_sparsity.csv",
                  PK / "libero_spatial/quant/sparsity/weight_sparsity_static.csv", False),
    ("Object",    PK / "libero_object/quant/sparsity/module_sparsity.csv",
                  PK / "libero_object/quant/sparsity/weight_sparsity_static.csv", False),
    ("Goal",      GOAL_DOCS / "module_sparsity_quant.csv",
                  GOAL_DOCS / "weight_sparsity_static_quant.csv", True),
    ("LIBERO-10", PK / "libero_10/quant/sparsity/module_sparsity.csv",
                  PK / "libero_10/quant/sparsity/weight_sparsity_static.csv", False),
]

COMPONENTS = ["Encoder", "VLM", "Expert"]
# component label -> the ``component`` value written into the CSVs
CSV_KEY = {"Encoder": "vision", "VLM": "vlm", "Expert": "expert"}


def read(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def aggregate(rows):
    """-> (element %, bit %) summed over rows (integer numerators first)."""
    if not rows:
        return float("nan"), float("nan")
    native = "zero_elements_native" in rows[0]
    zk, tk = (("zero_elements_native", "total_elements_native") if native
              else ("zero_elements", "total_elements"))
    sk, bk = (("sparse_bits_native", "total_bits_native") if native
              else ("sparse_bits", "total_bits"))
    z = sum(int(r[zk]) for r in rows)
    t = sum(int(r[tk]) for r in rows)
    s = sum(int(r[sk]) for r in rows)
    b = sum(int(r[bk]) for r in rows)
    return (100.0 * z / t if t else float("nan"),
            100.0 * s / b if b else float("nan"))


def collect_activation():
    """Linear *activation* role only: element% and bit% per suite/component."""
    out = {}
    for label, mod_csv, _w, _x in SUITES:
        rows = [r for r in read(mod_csv) if r["tensor_role"] == "activation"]
        out[label] = {c: aggregate([r for r in rows if r["component"] == CSV_KEY[c]])
                      for c in COMPONENTS}
    return out


def collect_weight():
    """Static weight: element% and bit% per suite/component."""
    out = {}
    for label, _m, w_csv, _x in SUITES:
        rows = read(w_csv)
        out[label] = {c: aggregate([r for r in rows if r["component"] == CSV_KEY[c]])
                      for c in COMPONENTS}
    return out


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def draw(ax, data, title, ylabel):
    labels = [s[0] for s in SUITES]
    x = np.arange(len(labels))
    w = 0.26
    offs = (-w, 0.0, w)

    for i, lab in enumerate(labels):
        for j, comp in enumerate(COMPONENTS):
            elem, bit = data[lab][comp]
            color = COMPONENT_SHADES[comp]
            xc = x[i] + offs[j]
            ax.bar(xc, bit, width=w, color=color, edgecolor="black",
                   linewidth=0.6, zorder=3)
            # bit% on top (the metric the bar encodes), element% right under it
            ax.annotate(f"{bit:.1f}", xy=(xc, bit), xytext=(0, 12),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold", color="black",
                        zorder=5)
            ax.annotate(f"el {elem:.2f}" if elem >= 0.005 else "el 0.00",
                        xy=(xc, bit), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.2,
                        color="dimgray", zorder=5)
            # delta gap between the two metrics, inside the bar
            ax.annotate(f"{bit - elem:+.1f}", xy=(xc, bit), xytext=(0, -6),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=6.4, fontweight="bold",
                        color=readable_on(color), zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 92)
    ax.set_yticks(np.arange(0, 81, 20))
    ax.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("LIBERO suite", fontsize=10.5, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.tick_params(axis="y", labelsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def build(data, title, ylabel, notes, legend_extra=None):
    fig, ax = plt.subplots(figsize=(9.6, 7.8), dpi=160)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.36)
    xc = (0.10 + 0.98) / 2

    draw(ax, data, title, ylabel)

    handles = [Patch(facecolor=COMPONENT_SHADES[c], edgecolor="black", label=c)
               for c in COMPONENTS]
    if legend_extra:
        handles += legend_extra

    # stack legend -> cross-suite means -> footnotes, every y MEASURED
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    gap_legend, gap_row, gap_note = 0.026, 0.015, 0.013

    y = ax.get_tightbbox(r).transformed(inv).y0 - gap_legend
    legend = fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                        bbox_to_anchor=(xc, y), fontsize=8.6, frameon=True,
                        framealpha=0.95, borderpad=0.6, handlelength=1.5,
                        columnspacing=1.2)
    fig.canvas.draw()
    y = legend.get_window_extent(r).transformed(inv).y0 - gap_row

    # one compact line with the cross-suite mean of each component, then the
    # footnotes -- every y MEASURED, never a hand-tuned fraction
    labels = [s[0] for s in SUITES]
    parts, worst = [], 0.0
    for c in COMPONENTS:
        bits = [data[l][c][1] for l in labels]
        if np.isnan(bits).any():
            continue
        worst = max(worst, max(bits) - min(bits))
        parts.append(f"{c} {np.mean(bits):.2f}%")
    t = fig.text(xc, y, "cross-suite mean    " + "   ·   ".join(parts)
                           + f"    (range ≤ {worst:.2f} pp)",
                 ha="center", va="top", fontsize=9.4, color="dimgray")
    fig.canvas.draw()
    y = t.get_window_extent(r).transformed(inv).y0 - gap_note * 2

    y -= gap_note
    for note in notes:
        t = fig.text(xc, y, note, ha="center", va="top", fontsize=8.2,
                     color="gray")
        fig.canvas.draw()
        y = t.get_window_extent(r).transformed(inv).y0 - gap_note
    if y < 0.01:
        print(f"[warn] footnotes run off the figure (last y={y:.3f})")
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR_PK.mkdir(parents=True, exist_ok=True)

    act = collect_activation()
    wgt = collect_weight()
    for name, data in [("ACTIVATION", act), ("WEIGHT", wgt)]:
        print(f"\n=== {name} (bit% / element%) ===")
        for lab, _m, _w, _x in SUITES:
            row = "  ".join(
                f"{c}={data[lab][c][1]:6.2f}/{data[lab][c][0]:6.3f}" for c in COMPONENTS)
            print(f"  {lab:<10s} {row}")

    saved = []

    fig = build(
        act,
        "Phase K — activation sparsity by component (Vision W8 / VLM W8 / Expert W4)",
        "Activation S|MMM bit sparsity (%)",
        [
            "Runtime native counting (FP outlier side-path zeros excluded). "
            "Metric = S|MMM v1: E4M3 S EEEE MMM, only S and MMM counted.",
            "Bar height = bit sparsity; the small grey label under each value is "
            "the element sparsity ('el x.xx' = element %).",
            "Linear activation role only (inputs of the quantized Linear). "
            "MatMul A/B/O and Linear output roles are not included.",
            "Spatial/Object/LIBERO-10 are this round's runs; Goal reuses the "
            "predecessor single-suite run (Goal was skipped this round).",
        ],
    )
    p = FIG_DIR / "phaseK_sparsity_activation_bars.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    saved.append(p)
    plt.close(fig)

    fig = build(
        wgt,
        "Phase K — static weight sparsity by component (Vision/VLM INT8, Expert INT4)",
        "Static weight S|MMM bit sparsity (%)",
        [
            "Sign-aware integer metric; INT8 and INT4 are reported separately and "
            "never pooled into one number.",
            "Bar height = bit sparsity; the small grey label under each value is "
            "the element sparsity ('el x.xx' = element %).",
            "Weights and their scales are shared by all suites, so these bars are "
            "identical across suites by construction -- the chart makes that "
            "verifiable at a glance.",
            "Spatial/Object/LIBERO-10 are this round's runs; Goal reuses the "
            "predecessor single-suite run (Goal was skipped this round).",
        ],
    )
    p = FIG_DIR / "phaseK_sparsity_weight_bars.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    saved.append(p)
    plt.close(fig)

    for p in saved:
        shutil.copy2(p, FIG_DIR_PK / p.name)
        print(f"[ok] {p}")
    print(f"[ok] copied {len(saved)} file(s) -> {FIG_DIR_PK}")


if __name__ == "__main__":
    main()
