#!/usr/bin/env python
"""Phase F — three suite-level grouped bar charts (one chart per protocol).

Design (per request):
  * three charts: one per protocol (F1 / F2 / F3)
  * within a chart the four LIBERO suites use four distinct hues
  * across charts each suite keeps its hue and only the lightness changes,
    deepening with quantization aggressiveness (F1 light -> F3 dark)
  * the FP16 baseline is drawn as an extra light-gray hatched bar next to
    every suite, so each chart is self-contained
  * bar labels show SR; the delta vs the FP16 baseline is printed inside the
    protocol bar (text colour auto-picked for contrast)

Layout is deliberately collision-proof: every text block except the bar
labels lives OUTSIDE the axes, so nothing can overlap a tall bar.

All figure text is English (avoids CJK font issues in matplotlib).

Data sources:
  FP16 reference : outputs/verify_libero/<suite>/                 (ep500)
  F1 / F2 / F3   : outputs/experiments/phaseF{1,2,3}_*_<suite>/   (ep10)

Usage:
    python experiments/2026-09-08_phaseF_fp8pot-foursuite/scripts/plot_phaseF_bars.py
Output:
    experiments/2026-09-08_phaseF_fp8pot-foursuite/docs/figures/phaseF_bars_{F1,F2,F3}.png
    experiments/2026-09-08_phaseF_fp8pot-foursuite/docs/figures/phaseF_bars_3panel.png
    (also copied to outputs/figures/)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
EXP_OUT = ROOT / "outputs" / "experiments"
VERIFY = ROOT / "outputs" / "verify_libero"
FIG_DIR = (
    ROOT / "experiments" / "2026-09-08_phaseF_fp8pot-foursuite" / "docs" / "figures"
)
ALT_FIG_DIR = ROOT / "outputs" / "figures"

# ---------------------------------------------------------------------------
# data spec
# ---------------------------------------------------------------------------

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABELS = ["Spatial", "Object", "Goal", "LIBERO-10"]

# (key, output-dir prefix, chart title)
PROTOCOLS = [
    ("F1", "phaseF1_fp8pot_site_outlier",
     "F1   FP8(e4m3) + PoT  ·  outlier protection 0.01"),
    ("F2", "phaseF2_fp8pot_site_nooutlier",
     "F2   FP8(e4m3) + PoT  ·  no outlier protection"),
    ("F3", "phaseF3_fp8pot_w4_site_outlier",
     "F3   Linear W4 + FP8 a/out  ·  outlier protection 0.01"),
]

# Four hues (one per suite) x three lightness steps (one per protocol).
# Same hue inside a row, light -> dark down the row.
SUITE_SHADES = {
    "Spatial":   ("#9ecae1", "#4292c6", "#08519c"),  # blue
    "Object":    ("#fdae6b", "#e6550d", "#a63603"),  # orange
    "Goal":      ("#a1d99b", "#31a354", "#006d2c"),  # green
    "LIBERO-10": ("#bcbddc", "#807dba", "#54278f"),  # purple
}
C_FP = "#d9d9d9"      # FP16 reference bar (constant across charts)
C_POS = "#1a7f37"
C_NEG = "#b3261e"


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def readable_on(hex_color: str) -> str:
    """Pick black or white text for maximum contrast on a fill colour."""
    return "white" if _luminance(hex_color) < 0.55 else "black"


def read_sr(d: Path) -> float:
    """Read pc_success from result.json or eval_info.json under directory d."""
    for name in ("result.json", "eval_info.json"):
        p = d / name
        if p.is_file():
            try:
                with open(p) as f:
                    return float(json.load(f)["overall"]["pc_success"])
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] cannot parse {p}: {exc}")
    print(f"[warn] no result found in {d}")
    return float("nan")


def collect():
    fp = np.array([read_sr(VERIFY / s) for s in SUITES], dtype=float)
    protos = []
    for key, prefix, title in PROTOCOLS:
        vals = np.array(
            [read_sr(EXP_OUT / f"{prefix}_{s}") for s in SUITES], dtype=float
        )
        protos.append((key, title, vals))
    return fp, protos


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def draw_bars(ax, fp, vals, shade_idx, title, show_ylabel=True):
    x = np.arange(len(SUITES))
    w = 0.38

    for i, lab in enumerate(SUITE_LABELS):
        color = SUITE_SHADES[lab][shade_idx]

        ax.bar(x[i] - w / 2, fp[i], width=w, color=C_FP, edgecolor="black",
               linewidth=0.6, hatch="//", zorder=3)
        ax.bar(x[i] + w / 2, vals[i], width=w, color=color, edgecolor="black",
               linewidth=0.6, zorder=3)

        # SR value above each bar
        if not np.isnan(fp[i]):
            ax.annotate(f"{fp[i]:.1f}", xy=(x[i] - w / 2, fp[i]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5, color="dimgray")
        if not np.isnan(vals[i]):
            ax.annotate(f"{vals[i]:.1f}", xy=(x[i] + w / 2, vals[i]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11,
                        fontweight="bold", color="black")

        # delta vs FP inside the protocol bar, just under its top edge
        if not (np.isnan(fp[i]) or np.isnan(vals[i])):
            d = vals[i] - fp[i]
            ax.annotate(f"{d:+.1f}", xy=(x[i] + w / 2, vals[i]),
                        xytext=(0, -5), textcoords="offset points",
                        ha="center", va="top", fontsize=9.5,
                        fontweight="bold", color=readable_on(color), zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(SUITE_LABELS, fontsize=10.5)
    ax.set_ylim(0, 116)          # headroom so SR labels never hit the title
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=12)
    ax.set_xlabel("LIBERO suite", fontsize=10.5, labelpad=8)
    if show_ylabel:
        ax.set_ylabel("Success rate (%)", fontsize=10.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=9.5)


def legend_handles(shade_idx):
    """FP16 patch + one patch per suite, sampled at this chart's shade."""
    handles = [Patch(facecolor=C_FP, edgecolor="black", hatch="//",
                     label="FP16 baseline (verify_libero)")]
    handles += [Patch(facecolor=SUITE_SHADES[lab][shade_idx], edgecolor="black",
                      label=lab)
                for lab in SUITE_LABELS]
    return handles


def mean_line(fig, x, y, fp, vals, key, fontsize=10.5, compact=False):
    """'Suite mean  FP16 x%  ->  <key> y%  (Δ)' with a colour-coded delta.

    ``x`` is a FIGURE-fraction x used as the split point: the descriptive part
    is right-aligned to it and the delta left-aligned from it, so the whole
    string stays centred on ``x`` whatever its text length. Callers must pass
    the x of the axes they belong to (not a hard-coded 0.5) or side-by-side
    panels will draw their mean lines on top of each other.
    """
    if np.isnan(fp).any() or np.isnan(vals).any():
        return
    m_fp, m_v = fp.mean(), vals.mean()
    d = m_v - m_fp
    prefix = "mean   " if compact else "Suite mean      "
    fig.text(x, y, f"{prefix}FP16 {m_fp:.1f}%    →    {key} {m_v:.1f}%   ",
             ha="right", va="center", fontsize=fontsize, color="dimgray")
    fig.text(x, y, f"({d:+.1f} pp)",
             ha="left", va="center", fontsize=fontsize + 1.1,
             fontweight="bold", color=C_POS if d >= 0 else C_NEG)


FOOTNOTE_1 = ("FP16 reference is the verify_libero protocol (50 episodes per task, "
              "500 total) while F1–F3 are 10 episodes per task, so the FP16 bars "
              "are indicative only.")
FOOTNOTE_2 = ("Suite mean is an equal-weight average over the four suites.   "
              "Bar hue identifies the suite; bar shade deepens from F1 to F3 with "
              "quantization aggressiveness.")


# reserved-margin layouts, shared by the builders and the layout checker
SINGLE_LAYOUT = dict(left=0.10, right=0.97, top=0.88, bottom=0.33)
PANEL_LAYOUT = dict(left=0.045, right=0.985, top=0.83, bottom=0.30)


def build_single(fp, key, title, vals, idx):
    """One chart for one protocol (legend + mean line + footnotes)."""
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=160)
    fig.subplots_adjust(**SINGLE_LAYOUT)
    xc = (SINGLE_LAYOUT["left"] + SINGLE_LAYOUT["right"]) / 2

    draw_bars(ax, fp, vals, idx, title)
    fig.legend(handles=legend_handles(idx), loc="upper center",
               bbox_to_anchor=(0.5, 0.255), ncol=5, fontsize=9.5,
               frameon=True, framealpha=0.95, columnspacing=1.4,
               handlelength=1.6, borderpad=0.7)
    mean_line(fig, xc, 0.115, fp, vals, key)
    fig.text(xc, 0.052, FOOTNOTE_1, ha="center", va="center",
             fontsize=8, color="gray")
    fig.text(xc, 0.016, FOOTNOTE_2, ha="center", va="center",
             fontsize=8, color="gray")
    return fig


def build_panel(fp, protos):
    """Combined 1x3 figure, one panel per protocol."""
    fig, axes = plt.subplots(1, 3, figsize=(20.5, 6.0), dpi=160, sharey=True)
    fig.subplots_adjust(**PANEL_LAYOUT)

    for idx, ((key, title, vals), ax) in enumerate(zip(protos, axes)):
        draw_bars(ax, fp, vals, idx, title, show_ylabel=(idx == 0))
        # mean line centred on THIS panel, in the band below its axes
        pos = ax.get_position()
        mean_line(fig, pos.x0 + pos.width / 2, pos.y0 - 0.075,
                  fp, vals, key, fontsize=9.5, compact=True)

    fig.suptitle("Phase F — LIBERO four-suite success rate by quantization protocol",
                 fontsize=15, fontweight="bold", y=0.985)
    fig.legend(handles=legend_handles(1), loc="upper center",
               bbox_to_anchor=(0.5, 0.175), ncol=5, fontsize=10.5,
               frameon=True, framealpha=0.95, columnspacing=1.6,
               handlelength=1.8, borderpad=0.8)
    fig.text(0.5, 0.045, FOOTNOTE_1 + "   " + FOOTNOTE_2,
             ha="center", va="center", fontsize=8.5, color="gray")
    return fig


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[ok] {out}")
    return out


def main():
    fp, protos = collect()
    print("FP16 :", np.round(fp, 1).tolist())
    for key, _t, vals in protos:
        print(f"{key}   :", np.round(vals, 1).tolist())

    saved = []
    for idx, (key, title, vals) in enumerate(protos):
        fig = build_single(fp, key, title, vals, idx)
        saved.append(save(fig, f"phaseF_bars_{key}.png"))
        plt.close(fig)

    fig = build_panel(fp, protos)
    saved.append(save(fig, "phaseF_bars_3panel.png"))
    plt.close(fig)

    ALT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in saved:
        shutil.copy2(p, ALT_FIG_DIR / p.name)
    print(f"[ok] copied {len(saved)} file(s) -> {ALT_FIG_DIR}")


if __name__ == "__main__":
    main()
