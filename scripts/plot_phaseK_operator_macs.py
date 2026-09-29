#!/usr/bin/env python
"""Phase K — per-operator MAC / activation-sparsity charts (one per component).

Three figures, one per model component:

  phaseK_operator_macs_encoder.png   vision  (SigLIP encoder, 12 layers)
  phaseK_operator_macs_vlm.png       vlm     (SmolLM2 text model, 16 layers)
  phaseK_operator_macs_expert.png    expert  (action expert, 16 layers)

Chart design (as requested):
  * x axis      = the operators actually present in that component, clustered by
                  name: q / k / v / o / qk / pv / gate / up / down
                  (the encoder MLP is fc1 / fc2 rather than gate / up / down).
  * bar height  = the operator's TOTAL dense-equivalent MACs over the rollout.
  * the bottom ``sparsity x height`` of every bar is drawn in a second colour
    and means "this fraction of the operator's MACs sits behind S|MMM zero
    bits".  Example: an 81.9 TMAC bar at 53.1% renders a 43.5 TMAC coloured
    region.

Only the **activation** side is measured:
  * Linear   -> ``tensor_role == "activation"`` (the input tensor).  The
                ``output`` role and the static ``weight`` are NOT included.
  * MatMul   -> ``tensor_role in {"A", "B"}`` -- for QK^T these are Q and K, for
                P@V they are the softmax probabilities and V.  Both operands are
                runtime activations; README: "MatMul B is the K^T/V activation,
                not a static weight".  The ``O`` role (the result) is excluded.

Data sources (read from disk, never hard-coded):
  MACs      : outputs/2026-09-25_phaseK_mixed-full-libero/<suite>/quant/compute.csv
              (shape-based hook; the authoritative MAC source -- workload.csv's
               per-role rows repeat the same MACs and must NOT be summed)
  Linear    : .../<suite>/quant/sparsity/module_sparsity.csv  (*_native columns)
  MatMul    : .../<suite>/quant/sparsity/workload.csv

CAVEATS -- read before quoting any number
  1. These MACs are **dense-equivalent algorithmic** counts, not executed GPU
     instructions and not sparse-hardware work.
  2. The coloured fraction is an **encoding-level** S|MMM zero-bit ratio.  Per
     the project manual a bit-zero ratio is NOT a compression ratio and NOT a
     speedup, and must not be multiplied by FLOPs to claim a saving.  The
     element-zero ratio, which is what zero-skipping hardware can actually act
     on, is printed as a secondary grey label; for most Linear ops it is ~0.
  3. Patch embedding (Conv2d) carries no S|MMM statistics, so it is excluded
     from the bars; its MACs are stated in the footnote instead.

Usage:
    python scripts/plot_phaseK_operator_macs.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from figure_palette import C_FP, COMPONENT_SHADES, readable_on  # noqa: E402

PK = ROOT / "outputs" / "2026-09-25_phaseK_mixed-full-libero"
PK_DOCS = ROOT / "experiments" / "2026-09-25_phaseK_mixed-full-libero" / "docs"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR_PK = PK_DOCS / "figures"

# MACs / sparsity are taken from this run so every bar is one consistent rollout.
SUITE = "libero_spatial"
GENERATIONS = None  # filled from compute_summary.json

# chart order follows the request: projections -> attention matmuls -> MLP
OP_ORDER = ["q", "k", "v", "o", "qk", "pv", "gate", "up", "down", "fc1", "fc2"]
# operator tail in the CSVs -> display label
LABEL = {
    "q_proj": "q", "k_proj": "k", "v_proj": "v",
    "o_proj": "o", "out_proj": "o",
    "qk": "qk", "pv": "pv",
    "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
    "fc1": "fc1", "fc2": "fc2",
}
COMPONENTS = [
    ("encoder", "vision", "Encoder (SigLIP, 12 layers)", "Encoder"),
    ("vlm", "vlm", "VLM (SmolLM2 text model, 16 layers)", "VLM"),
    ("expert", "expert", "Expert (action expert, 16 layers)", "Expert"),
]


def read(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def load(suite: str):
    """-> (per-component macs dict, per-(component, label) activation counts).

    The component is taken from compute.csv's own ``component`` column rather
    than from the module_id prefix: the unquantized patch embedding is logged
    under the full path
    ``model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding``
    with ``component=vision``, so a prefix test would silently drop its MACs.
    """
    q = PK / suite / "quant"
    macs: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in read(q / "compute.csv"):
        macs[r["component"]][r["module_id"]] += float(r["macs"])

    # counts are carried as (zero, total, sparse_bits, total_bits) per key
    act: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0] * 4)

    for r in read(q / "sparsity" / "module_sparsity.csv"):
        if r["tensor_role"] != "activation":
            continue
        key = (r["component"], LABEL[r["operator"]])
        act[key][0] += int(r["zero_elements_native"])
        act[key][1] += int(r["total_elements_native"])
        act[key][2] += int(r["sparse_bits_native"])
        act[key][3] += int(r["total_bits_native"])

    for r in read(q / "sparsity" / "workload.csv"):
        if r["op_type"] != "matmul" or r["tensor_role"] not in ("A", "B"):
            continue
        mid = r["module_id"]
        comp = mid.split(".")[0]
        label = LABEL[mid.split(".")[-1]]
        bits = float(r["bits"])
        sparse = float(r["bit_sparsity"]) * bits
        # workload.csv exposes the bit rate only; 4 bits per element is the
        # native S|MMM width, so the element counts are recovered from it.
        act[(comp, label)][0] += sparse / 4.0
        act[(comp, label)][1] += bits / 4.0
        act[(comp, label)][2] += sparse
        act[(comp, label)][3] += bits

    return macs, act


def series(macs, act, comp):
    """Ordered [(label, TMAC, element %, bit %)] and the unplotted MAC remainder."""
    comp_macs = macs[comp]
    plotted = []
    for key in OP_ORDER:
        mids = [m for m in comp_macs if LABEL.get(m.split(".")[-1]) == key]
        if not mids:
            continue
        counts = act.get((comp, key))
        if counts is None or counts[3] == 0:
            continue
        mac = sum(comp_macs[m] for m in mids) / 1e12
        plotted.append((key, mac,
                        100.0 * counts[0] / counts[1] if counts[1] else 0.0,
                        100.0 * counts[2] / counts[3] if counts[3] else 0.0))
    total = sum(comp_macs.values()) / 1e12
    shown = sum(p[1] for p in plotted)
    # clamp: the two sums are floats and can differ by ~1e-13 (a signed -0.00
    # would then be printed for a component that has nothing left over)
    return plotted, max(0.0, total - shown)


def draw(ax, plotted, hue, title, subtitle):
    x = np.arange(len(plotted))
    w = 0.62
    ymax = max(p[1] for p in plotted) * 1.22
    ax.set_ylim(0, ymax)          # fixed BEFORE annotating: the 'thin region'
                                  # test below needs a settled y scale

    # pass 1 -- bars only
    for i, (_key, mac, _elem, bit) in enumerate(plotted):
        # full bar = total dense-equivalent MACs (neutral, the "reference")
        ax.bar(x[i], mac, width=w, color=C_FP, edgecolor="black",
               linewidth=0.7, zorder=3)
        # bottom portion of the bar = the S|MMM zero-bit share
        ax.bar(x[i], mac * bit / 100.0, width=w, color=hue, edgecolor="black",
               linewidth=0.7, zorder=4)

    # pass 2 -- labels
    for i, (_key, mac, elem, bit) in enumerate(plotted):
        skip = mac * bit / 100.0
        ax.annotate(f"{mac:.1f}", xy=(x[i], mac), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9.6, fontweight="bold", color="black", zorder=6)
        # The percentage sits inside the coloured region; the element ratio sits
        # just ABOVE the boundary (in the neutral part).  Putting both inside
        # collides on short regions, where the two text boxes would overlap.
        el_txt = f"el {elem:.2f}" if elem >= 0.005 else "el 0.00"
        if skip < 0.035 * ymax:      # too short for two stacked lines
            ax.annotate(f"{bit:.1f}%  {el_txt}", xy=(x[i], skip), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.0, fontweight="bold", color="black", zorder=7)
        else:
            ax.annotate(f"{bit:.1f}%", xy=(x[i], skip / 2), ha="center",
                        va="center", fontsize=9.4, fontweight="bold",
                        color=readable_on(hue), zorder=7)
            ax.annotate(el_txt, xy=(x[i], skip), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.4, color="dimgray", zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in plotted], fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=22)
    ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=9.5, color="dimgray")
    ax.set_xlabel("Operator", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Total dense-equivalent MACs (TMAC)", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def build(key, comp, title, hue_name, macs, act, generations):
    plotted, remainder = series(macs, act, comp)
    hue = COMPONENT_SHADES[hue_name]
    shown = sum(p[1] for p in plotted)

    fig, ax = plt.subplots(figsize=(9.8, 9.0), dpi=160)
    fig.subplots_adjust(left=0.095, right=0.98, top=0.87, bottom=0.38)
    xc = (0.095 + 0.98) / 2

    draw(ax, plotted, hue, title,
         f"{SUITE}, {generations} generations · activation side only")

    handles = [
        Patch(facecolor=C_FP, edgecolor="black",
              label="total dense-equivalent MACs"),
        Patch(facecolor=hue, edgecolor="black",
              label="S|MMM zero-bit share (illustrative)"),
    ]

    # weighted aggregate over the plotted operators
    agg_num = sum(p[1] * p[3] for p in plotted)
    agg_den = sum(p[1] for p in plotted)
    agg_bit = agg_num / agg_den if agg_den else float("nan")

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    gap_legend, gap_row, gap_note = 0.026, 0.016, 0.013

    y = ax.get_tightbbox(r).transformed(inv).y0 - gap_legend
    legend = fig.legend(handles=handles, loc="upper center", ncol=2,
                        bbox_to_anchor=(xc, y), fontsize=9.2, frameon=True,
                        framealpha=0.95, borderpad=0.6, handlelength=1.8,
                        columnspacing=1.6)
    fig.canvas.draw()
    y = legend.get_window_extent(r).transformed(inv).y0 - gap_row

    rows = [
        f"plotted operators   {shown:.1f} TMAC"
        + (f"   ({100 * shown / (shown + remainder):.1f}% of the component)"
           if shown + remainder else ""),
        f"MAC-weighted S|MMM zero-bit share over those operators   "
        f"{agg_bit:.2f}%",
    ]
    if remainder > 1e-9:
        rows.append(f"not plotted   {remainder:.2f} TMAC"
                    f"  (patch embedding Conv2d / ops with no S|MMM statistics)")
    for row in rows:
        t = fig.text(xc, y, row, ha="center", va="top", fontsize=9.3,
                     color="dimgray")
        fig.canvas.draw()
        y = t.get_window_extent(r).transformed(inv).y0 - gap_row

    notes = [
        "Bar height = total MACs over the rollout.  The coloured bottom region is "
        "that operator's S|MMM zero-bit share of those MACs.",
        "Activation side only: Linear input tensor; MatMul operands A and B.  "
        "Linear 'output' role, MatMul 'O' role and all static weights are excluded.",
        "'el x.xx' under each percentage is the element-zero ratio -- what "
        "zero-skipping hardware could actually act on.  For most Linear ops it is ~0.",
        "CAUTION: MACs are dense-equivalent algorithmic counts, not executed GPU "
        "instructions.",
        "A bit-zero ratio is neither a compression ratio nor a speedup, and must "
        "not be multiplied by FLOPs to claim a saving.",
    ]
    y -= gap_note
    for note in notes:
        t = fig.text(xc, y, note, ha="center", va="top", fontsize=8.1,
                     color="gray")
        fig.canvas.draw()
        y = t.get_window_extent(r).transformed(inv).y0 - gap_note
    if y < 0.008:
        print(f"[warn] {key}: footnotes reach y={y:.3f}")
    return fig, plotted, shown, remainder, agg_bit

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR_PK.mkdir(parents=True, exist_ok=True)

    import json
    global GENERATIONS
    GENERATIONS = json.loads(
        (PK / SUITE / "quant" / "compute_summary.json").read_text())["generations"]

    macs, act = load(SUITE)
    saved = []
    for key, comp, title, hue_name in COMPONENTS:
        fig, plotted, shown, rem, agg = build(
            key, comp, f"Phase K — {title}: MACs and activation sparsity",
            hue_name, macs, act, GENERATIONS)
        print(f"\n=== {key} ===  plotted {shown:.2f} TMAC, "
              f"unplotted {rem:.2f} TMAC, weighted zero-bit {agg:.2f}%")
        for lab, mac, elem, bit in plotted:
            print(f"   {lab:6s} MAC={mac:8.3f} TMAC   skip={mac * bit / 100:7.3f}"
                  f"   bit={bit:6.2f}%   elem={elem:7.3f}%")
        p = FIG_DIR / f"phaseK_operator_macs_{key}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        saved.append(p)
        plt.close(fig)

    for p in saved:
        shutil.copy2(p, FIG_DIR_PK / p.name)
        print(f"[ok] {p}")
    print(f"[ok] copied {len(saved)} file(s) -> {FIG_DIR_PK}")


if __name__ == "__main__":
    main()
