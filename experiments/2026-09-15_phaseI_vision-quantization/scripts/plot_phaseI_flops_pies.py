#!/usr/bin/env python
"""Phase I — FLOPs share pie charts.

Two pies, at two levels of the same hierarchy:

  Pie A  Vision encoder + connector internals (MEASURED, V0 workload audit)
         MLP / Attention projection / QK / PV / Connector
         Source: outputs/2026-09-15_phaseI_vision-quantization/
                 v0_workload_audit/vision_flops.csv

  Pie B  Whole inference, one `sample_actions()` (manual §0 estimate)
         Vision encoder + pixel shuffle/connector  |  VLM 16-layer prefix
         prefill  |  Action Expert 10-step denoise
         Source: 2026-09-15_phaseI_vision_quantization_experiment_manual.md §0

Pie B's numbers are PARSED from the manual rather than hard-coded, so the
figure cannot silently disagree with the document it illustrates; parsing
failing loudly is preferred over a stale constant. (Cross-check: the manual
estimates Vision+Connector at 430.6 G while V0 measured 428.2 G -- agreement
within 0.6%.)

Colour scheme: pie B gives each top-level branch its own hue (Vision = blue,
VLM = orange, Expert = green, from scripts/figure_palette.py). Pie A is a
zoom-in on the blue branch, so it uses five lightness steps of that same blue
(scripts/figure_palette.py BLUE_SHADES_5), ordered darkest = largest share.

Layout: percentage labels go INSIDE a wedge when it is big enough to hold them
and OUTSIDE with a leader line otherwise, so a 0.7% slice stays readable. The
legend and footnotes are stacked below the axes at renderer-measured positions
(scripts/figure_layout.py).

Usage:
    python experiments/2026-09-15_phaseI_vision-quantization/scripts/plot_phaseI_flops_pies.py
Output:
    experiments/2026-09-15_phaseI_vision-quantization/docs/figures/
        phaseI_flops_pie_vision.png
        phaseI_flops_pie_inference.png
        phaseI_flops_pie_2panel.png
    (also copied to outputs/figures/)
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# paths + shared palette / layout
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "2026-09-15_phaseI_vision-quantization"
FIG_DIR = EXP / "docs" / "figures"
ALT_FIG_DIR = ROOT / "outputs" / "figures"

MANUAL = ROOT / "2026-09-15_phaseI_vision_quantization_experiment_manual.md"
V0_DIR = (ROOT / "outputs" / "2026-09-15_phaseI_vision-quantization"
          / "v0_workload_audit")

sys.path.insert(0, str(ROOT / "scripts"))
from figure_layout import stack_below_axes  # noqa: E402
from figure_palette import (  # noqa: E402
    BLUE_SHADES_5,
    SUITE_SHADES,
    readable_on,
)

# top-level branch hues (pie B) -- mid lightness step of each family
C_VISION = SUITE_SHADES["Spatial"][1]     # blue
C_VLM = SUITE_SHADES["Object"][1]         # orange
C_EXPERT = SUITE_SHADES["Goal"][1]        # green

# operator key -> human label, for the V0 CSV
V0_LABELS = {
    "mlp": "Vision MLP (fc1/fc2)",
    "attention_projection": "Attention Q/K/V/out proj",
    "qk": "Attention QK^T",
    "pv": "Attention P x V",
    "connector_proj": "Connector projection",
}

TOTAL_FLOP_LABEL = "GFLOPs / sample_actions()"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def read_v0_flops():
    """Return (labels, gflops, colors) for the Vision+Connector breakdown.

    Colours are five lightness steps of the blue branch, darkest = largest
    share, so the reading order matches the visual weight.
    """
    path = V0_DIR / "vision_flops.csv"
    if not path.is_file():
        raise SystemExit(f"missing V0 audit: {path}")

    rows = list(csv.DictReader(open(path, newline="")))
    rows.sort(key=lambda r: float(r["flops"]), reverse=True)

    labels, gflops = [], []
    for r in rows:
        op = r["operator"]
        labels.append(V0_LABELS.get(op, r["component"] + "." + op))
        gflops.append(float(r["flops"]) / 1e9)

    n = len(rows)
    ramp = BLUE_SHADES_5
    if n > len(ramp):
        raise SystemExit(f"{n} slices but only {len(ramp)} blue steps available")
    # darkest for the biggest slice
    colors = [ramp[len(ramp) - 1 - i] for i in range(n)]
    return labels, gflops, colors


def parse_manual_inference_flops():
    """Parse the §0 whole-inference FLOPs table out of the Phase I manual.

    Returns [(label, gflops)]. Raises if the table cannot be found, so a
    reformatted manual surfaces as an error instead of a wrong figure.
    """
    if not MANUAL.is_file():
        raise SystemExit(f"missing manual: {MANUAL}")

    rows = []
    for line in MANUAL.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        m = re.search(r"~?([\d.]+)\s*GFLOPs", cells[1])
        if not m:
            continue
        label = cells[0].strip("`").strip()
        if "总计" in label or "total" in label.lower():
            continue
        rows.append((label, float(m.group(1))))

    if len(rows) != 3:
        raise SystemExit(
            f"expected 3 top-level rows in the manual §0 table, got {len(rows)}: "
            f"{rows}")
    return rows


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def draw_pie(ax, values, colors, labels, inside_threshold=5.0):
    """Pie with in-wedge pct labels, falling back to leader lines when small."""
    total = float(sum(values))
    wedges, _ = ax.pie(
        values, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=1.6),
    )

    artists = []
    for w, lab, v, c in zip(wedges, labels, values, colors):
        pct = 100.0 * v / total
        ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        if pct >= inside_threshold:
            x, y = 0.60 * np.cos(ang), 0.60 * np.sin(ang)
            artists.append(ax.text(
                x, y, f"{pct:.2f}%", ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=readable_on(c), zorder=5))
        else:
            # leader line out to the side, text pushed clear of the wedge
            x, y = 1.16 * np.cos(ang), 1.16 * np.sin(ang)
            ha = "left" if x >= 0 else "right"
            artists.append(ax.annotate(
                f"{lab}\n{pct:.2f}%",
                xy=(0.97 * np.cos(ang), 0.97 * np.sin(ang)),
                xytext=(x, y), ha=ha, va="center", fontsize=9.5,
                color=readable_on(c) if False else "black",
                arrowprops=dict(arrowstyle="-", color="dimgray", lw=1.0,
                                shrinkA=0, shrinkB=2),
                zorder=6))

    ax.set_aspect("equal")
    return wedges, artists


def legend_handles(labels, values, colors):
    total = float(sum(values))
    return [Patch(facecolor=c, edgecolor="white",
                  label=f"{lab} — {v:.1f} G ({100.0 * v / total:.2f}%)")
            for lab, v, c in zip(labels, values, colors)]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def build_vision_pie():
    labels, gflops, colors = read_v0_flops()
    total = sum(gflops)

    fig, ax = plt.subplots(figsize=(9.2, 7.4), dpi=160)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.855, bottom=0.30)
    fig.suptitle("Phase I (V0, measured) — Vision encoder + connector FLOPs",
                 fontsize=13.5, fontweight="bold", y=0.975)
    ax.set_title(f"{total:.1f} GFLOPs per sample_actions()   "
                 f"(2 cameras x 1024 patch tokens, 12-layer ViT)",
                 fontsize=10, color="dimgray", pad=14)

    draw_pie(ax, gflops, colors, labels)
    stack_below_axes(
        fig, ax, legend_handles(labels, gflops, colors),
        ["Measured by hooking the real wrap-time modules "
         "(outputs/.../v0_workload_audit/vision_flops.csv); "
         "1 MAC = 2 FLOPs.",
         "Five lightness steps of one hue: this pie zooms into the blue "
         "(Vision) branch of the top-level breakdown below. "
         "Patch-embedding Conv2d is not instrumented in V0 "
         "(manual §0 estimates it at 0.56% of Vision)."],
        xc=0.5)

    out = FIG_DIR / "phaseI_flops_pie_vision.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {out}")
    return out


def build_inference_pie():
    rows = parse_manual_inference_flops()
    # manual order is Vision / VLM / Expert; keep it
    labels = [r[0] for r in rows]
    gflops = [r[1] for r in rows]
    colors = [C_VISION, C_VLM, C_EXPERT]
    total = sum(gflops)

    fig, ax = plt.subplots(figsize=(9.2, 7.4), dpi=160)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.855, bottom=0.30)
    fig.suptitle("Phase I — whole-inference FLOPs per sample_actions()",
                 fontsize=13.5, fontweight="bold", y=0.975)
    ax.set_title(f"{total:.1f} GFLOPs  ≈  {total / 1000:.2f} TFLOPs "
                 f"per sample_actions()",
                 fontsize=10, color="dimgray", pad=14)

    draw_pie(ax, gflops, colors, labels)
    stack_below_axes(
        fig, ax, legend_handles(labels, gflops, colors),
        ["Manual §0 estimate for lerobot/smolvla_libero with LIBERO "
         "dual-camera 512x512 input.",
         "Cross-check: V0 measured Vision+Connector at 428.2 G, agreeing with "
         "the 430.6 G estimate to within 0.6%.  One hue per top-level branch "
         "(Vision blue, VLM orange, Expert green)."],
        xc=0.5)

    out = FIG_DIR / "phaseI_flops_pie_inference.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {out}")
    return out


def build_panel():
    """Side-by-side view: whole inference next to the Vision zoom-in."""
    rows = parse_manual_inference_flops()
    b_labels = [r[0] for r in rows]
    b_vals = [r[1] for r in rows]
    b_colors = [C_VISION, C_VLM, C_EXPERT]
    a_labels, a_vals, a_colors = read_v0_flops()

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 8.6), dpi=160)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.845, bottom=0.27)

    for ax, (labels, vals, colors, title) in zip(axes, [
        (b_labels, b_vals, b_colors, "Whole inference (manual §0 estimate)"),
        (a_labels, a_vals, a_colors, "Zoom-in: Vision encoder + connector "
                                     "(V0 measured)"),
    ]):
        ax.set_title(f"{title}\n{sum(vals):.1f} GFLOPs per "
                     f"sample_actions()", fontsize=11.5, fontweight="bold",
                     pad=14)
        draw_pie(ax, vals, colors, labels)

    fig.suptitle("Phase I — where the inference FLOPs go",
                 fontsize=15, fontweight="bold", y=0.975)

    handles = legend_handles(b_labels, b_vals, b_colors)
    handles += [Patch(facecolor=c, edgecolor="white", label=f"[Vision] {lab}")
                for lab, c in zip(a_labels, a_colors)]
    y = axes[0].get_position().y0 - 0.055
    fig.canvas.draw()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=2, fontsize=9, frameon=True, framealpha=0.95,
               borderpad=0.8, handlelength=1.6)
    fig.canvas.draw()
    ry = fig.legends[0].get_window_extent(
        fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0
    fig.text(0.5, ry - 0.03,
             "Left: one hue per top-level branch.  Right: five lightness steps "
             "of the Vision (blue) hue.  Percentages are FLOP shares; the "
             "small Connector slice uses a leader line.",
             ha="center", va="top", fontsize=8.2, color="gray")

    out = FIG_DIR / "phaseI_flops_pie_2panel.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {out}")
    return out


def main():
    saved = [build_vision_pie(), build_inference_pie(), build_panel()]

    ALT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in saved:
        shutil.copy2(p, ALT_FIG_DIR / p.name)
    print(f"[ok] copied {len(saved)} file(s) -> {ALT_FIG_DIR}")

    # machine-readable echo of the numbers that went into the figures
    print("\nVision+Connector breakdown (V0 measured):")
    for lab, v in zip(*read_v0_flops()[:2]):
        print(f"  {lab:32s} {v:8.2f} G")
    print("\nWhole inference (manual §0):")
    rows = parse_manual_inference_flops()
    tot = sum(r[1] for r in rows)
    for lab, v in rows:
        print(f"  {lab:32s} {v:8.2f} G  {100 * v / tot:6.2f}%")
    print(f"  {'TOTAL':32s} {tot:8.2f} G  100.00%")


if __name__ == "__main__":
    main()
