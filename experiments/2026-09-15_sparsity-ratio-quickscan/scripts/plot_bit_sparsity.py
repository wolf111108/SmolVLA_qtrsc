#!/usr/bin/env python
"""Sparsity quick scan — BIT sparsity, all four configs, runtime vs weights split.

Four figures (2 tensor kinds x 2 stages):

  runtime  VLM prefill      Linear activation/output + QK A/B/O + PV A/B/O   (8)
  runtime  Expert denoise   same roles                                      (8)
  weights  VLM prefill      q/k/v/o/gate/up/down_proj                       (7)
  weights  Expert denoise   same roles                                      (7)

Four configs per role, ordered mild -> aggressive by nominal weight bit budget
(this also keeps the integer family and the float family each adjacent):

  INT16        16-bit integer weight        (lightest)
  INT8          8-bit integer weight
  FP8           8-bit float (e4m3) weight
  FP8-W4        4-bit integer weight        (most aggressive)   <- deployment pair

Why runtime and static weights are SEPARATE figures (not one chart with a
divider):
  They are different kinds of measurement. Runtime tensors are per-forward
  quantized activations / MatMul operands observed during rollout; static
  weights are the calibrated weight tensors, attributed to a stage by component
  (vlm -> prefill, expert -> denoise), which is a deployment / workload
  attribution and NOT a claim that weights vary over time. A shared axis invites
  reading them as one comparable family, which they are not. The earlier
  single-chart version also mislabelled its divider (off-by-one, line drawn at
  6.5 instead of 7.5) so the "static weights" caption landed on the PV columns;
  splitting removes that whole class of error.

Why the two stages stay in separate figures (README_EXPERIMENT.md §1/§14):
  One SmolVLA action generation runs VLM prefill ONCE but Expert denoise over
  10 flow steps, so merging them into one headline ratio is meaningless.

IMPORTANT — QK/PV have NO weights here: they are MatMul *sites*, so they appear
only among the runtime roles. `weight_sparsity_static.csv` contains exactly 224
rows = 7 Linear projections x 16 layers x 2 components, with no qk/pv row
(verified against the CSV). MatMul operand quantisation is captured by the
runtime A/B/O roles instead.

Palette: `scripts/figure_palette.py`, the single source of truth Phase F
introduced and Phase G reuses. The Phase F Goal (green) hue carries the protocol
family with F's convention "light = milder":
  GOAL_SHADES_4 = ("#c7e9c0", "#a1d99b", "#31a354", "#006d2c")
A dashed step marks each role's FP8 value (the accuracy-preserving anchor per
Phase F/H); the in-bar number on the darkest bar is Δ(FP8-W4 − FP8) in pp.

Metric semantics (README §3): INT uses a sign-aware sparse-bit metric; FP8 E4M3
uses the 4-bit significand zero-bit ratio. They are NOT the same encoding
definition — compare shapes, never raw pp.

All figure text is English (avoids CJK font issues in matplotlib).

Data source (read from disk, never hard-coded):
  outputs/2026-09-15_sparsity-ratio-quickscan/quick_sparsity_by_role.csv

Usage:
    python experiments/2026-09-15_sparsity-ratio-quickscan/scripts/plot_bit_sparsity.py
Output:
    experiments/2026-09-15_sparsity-ratio-quickscan/docs/figures/
        quickscan_bit_sparsity_runtime_vlm_prefill.png
        quickscan_bit_sparsity_runtime_expert_denoise.png
        quickscan_bit_sparsity_weights_vlm_prefill.png
        quickscan_bit_sparsity_weights_expert_denoise.png
    (also copied to outputs/figures/)
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# paths + shared palette / layout
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "2026-09-15_sparsity-ratio-quickscan"
FIG_DIR = EXP / "docs" / "figures"
ALT_FIG_DIR = ROOT / "outputs" / "figures"
OUT_ROOT = ROOT / "outputs" / "2026-09-15_sparsity-ratio-quickscan"
BY_ROLE = OUT_ROOT / "quick_sparsity_by_role.csv"

sys.path.insert(0, str(ROOT / "scripts"))
from figure_layout import stack_below_axes  # noqa: E402
from figure_palette import (  # noqa: E402
    C_REF,
    GOAL_SHADES_4,
    readable_on,
)

# mild -> aggressive by nominal weight bit budget (see module docstring)
CFG_STYLE = [
    ("INT16 (W = int16)", "INT16", GOAL_SHADES_4[0]),
    ("INT8 (W = int8)", "INT8", GOAL_SHADES_4[1]),
    ("FP8 (W = e4m3)", "FP8 PoT", GOAL_SHADES_4[2]),
    ("FP8-W4 (W = int4)", "FP8W4 PoT", GOAL_SHADES_4[3]),
]
# the accuracy-preserving anchor used for the dashed reference + Δ annotation
ANCHOR_IDX = 2                      # FP8
DELTA_ON_IDX = 3                    # FP8-W4

# ---------------------------------------------------------------------------
# role groups — runtime and static weights are never mixed in one chart
# ---------------------------------------------------------------------------

# Runtime: 8 roles. QK/PV live here ONLY (MatMul sites, no weights of their own).
RUNTIME_ROLES = [
    ("Linear:activation", "Linear\nactivation"),
    ("Linear:output", "Linear\noutput"),
    ("QK:A", "QK\nA"),
    ("QK:B", "QK\nB"),
    ("QK:O", "QK\nO"),
    ("PV:A", "PV\nA"),
    ("PV:B", "PV\nB"),
    ("PV:O", "PV\nO"),
]

# Static weights: 7 roles, one per Linear projection.
WEIGHT_ROLES = [
    ("weight:q_proj", "q_proj"),
    ("weight:k_proj", "k_proj"),
    ("weight:v_proj", "v_proj"),
    ("weight:o_proj", "o_proj"),
    ("weight:gate_proj", "gate_proj"),
    ("weight:up_proj", "up_proj"),
    ("weight:down_proj", "down_proj"),
]

KINDS = [
    ("runtime", RUNTIME_ROLES,
     "runtime tensors  (quantized activations / MatMul operands)"),
    ("weights", WEIGHT_ROLES,
     "static weights  (calibrated Linear weight tensors)"),
]
STAGES = ["VLM prefill", "Expert denoise"]


def _norm_role(role: str) -> str:
    """Strip the weight_spec suffix so int16/int8/e4m3/int4 share a key."""
    return role.split("[", 1)[0] if role.startswith("weight:") else role


def load_by_role():
    """{(config_label, stage): {role_key: (elem_pct, bit_pct)}} from Secondary CSV."""
    if not BY_ROLE.is_file():
        raise SystemExit(
            f"missing {BY_ROLE} — run summarize_quick_sparsity.py first"
        )
    data = {}
    with open(BY_ROLE, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["config"], r["stage"])
            data.setdefault(key, {})[_norm_role(r["role"])] = (
                float(r["element_sparsity"]) * 100.0,
                float(r["bit_sparsity"]) * 100.0,
            )
    return data


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _series(by_role, cfg_key, stage, roles):
    """Bit-sparsity values (in %) for one config across the given roles."""
    vals = np.full(len(roles), np.nan)
    for i, (rk, _) in enumerate(roles):
        rec = by_role.get((cfg_key, stage), {}).get(rk)
        if rec is not None:
            vals[i] = rec[1]
    return vals


def draw_chart(ax, by_role, stage, roles, title, xlabel):
    n = len(roles)
    nbar = len(CFG_STYLE)
    x = np.arange(n, dtype=float)
    w = 0.20
    span = w * 1.1
    offsets = (np.arange(nbar) - (nbar - 1) / 2.0) * span      # ~[-.33 .. .33]

    series = {}
    for ci, (label, cfg_key, color) in enumerate(CFG_STYLE):
        vals = _series(by_role, cfg_key, stage, roles)
        series[cfg_key] = vals
        ax.bar(x + offsets[ci], np.nan_to_num(vals), width=w, color=color,
               edgecolor="black", linewidth=0.5, zorder=3, label=label)
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            ax.annotate(f"{v:.0f}", xy=(x[i] + offsets[ci], v), xytext=(0, 2.5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.2, fontweight="bold", color="black",
                        zorder=6)

    # dashed reference at the FP8 value + Δ(FP8-W4 − FP8) inside the dark bar
    anchor = series[CFG_STYLE[ANCHOR_IDX][1]]
    dark = series[CFG_STYLE[DELTA_ON_IDX][1]]
    dark_color = CFG_STYLE[DELTA_ON_IDX][2]
    for i in range(n):
        if not np.isnan(anchor[i]):
            ax.plot([x[i] - 0.40, x[i] + 0.40], [anchor[i], anchor[i]],
                    color=C_REF, linestyle=(0, (4, 3)), linewidth=1.2,
                    alpha=0.85, zorder=4)
        if not np.isnan(dark[i]) and not np.isnan(anchor[i]):
            d = dark[i] - anchor[i]
            ax.annotate(f"{d:+.0f}", xy=(x[i] + offsets[DELTA_ON_IDX], dark[i]),
                        xytext=(0, -8.5), textcoords="offset points",
                        ha="center", va="top", fontsize=7.0, fontweight="bold",
                        color=readable_on(dark_color), zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in roles], fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10.5, labelpad=8)
    ax.set_ylabel("Bit sparsity (%)", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


FOOT1 = (
    "libero_goal task0 x 1 episode, n_action_steps=10, num_steps=10.   "
    "Value above each bar = bit sparsity (%);  in-bar number on the darkest "
    "bar = Δ(FP8-W4 − FP8) in pp.\n"
    "Bar order is mild → aggressive by nominal weight bit budget "
    "(INT16 > INT8 ≈ FP8 > FP8-W4);  dashed step = that role's FP8 value "
    "(accuracy-preserving anchor)."
)

FOOT2 = (
    "Bit metric is NOT uniform: INT uses a sign-aware sparse-bit metric, FP8 "
    "E4M3 uses the 4-bit significand zero-bit ratio — compare shapes, not raw pp.\n"
    "QK/PV are MatMul sites with no weights of their own, so they appear only "
    "in the runtime figures.\n"
    "Scope: quantized VLM text Transformer + Action Expert only "
    "(no vision encoder / connector)."
)


def build_figure(by_role, stage, kind_name, roles, kind_caption):
    n = len(roles)
    width = 7.2 + 0.80 * n
    fig, ax = plt.subplots(figsize=(width, 8.4), dpi=160)
    layout = dict(left=0.085, right=0.99, top=0.90, bottom=0.38)
    fig.subplots_adjust(**layout)
    xc = (layout["left"] + layout["right"]) / 2

    title = f"Q0–Q3 bit sparsity — {stage}  ·  {kind_name}"
    draw_chart(ax, by_role, stage, roles, title, kind_caption)

    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=lab)
               for lab, _, c in CFG_STYLE]
    handles.append(Line2D([0], [0], color=C_REF, linestyle=(0, (4, 3)),
                          linewidth=1.2, label="FP8 value (per role)"))
    stack_below_axes(fig, ax, handles, [FOOT1, FOOT2], xc,
                     legend_fontsize=8.8, fontsize=7.6)
    return fig


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[ok] {out}")
    return out


def main():
    by_role = load_by_role()
    saved = []

    for kind_name, roles, kind_caption in KINDS:
        for stage in STAGES:
            tag = "vlm_prefill" if stage == "VLM prefill" else "expert_denoise"
            fname = f"quickscan_bit_sparsity_{kind_name}_{tag}.png"
            fig = build_figure(by_role, stage, kind_name, roles, kind_caption)
            saved.append(save(fig, fname))
            plt.close(fig)

    # console recap
    print()
    for kind_name, roles, _ in KINDS:
        print(f"=== {kind_name} ({len(roles)} roles) ===")
        for stage in STAGES:
            parts = []
            for lab, cfg_key, _ in CFG_STYLE:
                v = _series(by_role, cfg_key, stage, roles)
                ok = ~np.isnan(v)
                parts.append(f"{lab.splitlines()[0]:7s} {np.nanmean(v[ok]):5.1f}")
            print(f"  {stage:16s} mean ← " + " | ".join(parts))

    ALT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in saved:
        shutil.copy2(p, ALT_FIG_DIR / p.name)
    print(f"\n[ok] copied {len(saved)} file(s) -> {ALT_FIG_DIR}")


if __name__ == "__main__":
    main()
