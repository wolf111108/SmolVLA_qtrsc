#!/usr/bin/env python
"""
Aggregate and analyze raw tensor dumps produced during calibration.

The calibration path (QuantizedLinear._maybe_dump_tensors) dumps per-layer
activation / weight / output tensors to VLA_TENSOR_DUMP_DIR (default
outputs/tensor_dump). This script:

1. Scans the dump directory and groups tensors by (layer_name, layer_idx, kind).
2. Computes distribution statistics aimed at choosing an outlier_ratio:
   - abs percentiles (p50/p90/p99/p99.9/p99.99/max)
   - tail decay ratios (p99/max, p99.9/max, p99.99/max)
   - top-k L2 energy share for k in {0.01%, 0.1%, 1%, 5%}  (maps to outlier_ratio)
   - mean / std
3. Writes a JSON summary + CSV table.
4. Plots (matplotlib, Agg backend — no display needed):
   - log-scale abs-value histograms per kind (aggregated across layers)
   - top-k energy share curve per layer (weight only, since it's the SQNR culprit)

Usage:
    python scripts/analyze_tensor_dump.py [--dir outputs/tensor_dump] \
        [--out outputs/tensor_dump_analysis]

Environment-independent: reads whatever is already dumped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch

# Matplotlib with headless Agg backend (no display on remote box).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# =============================================================================
# Filename parsing
# =============================================================================

# weight_q_proj_0.pt  /  activation_q_proj_0_step2.pt  /  output_down_proj_5_step0.pt
_FILENAME_RE = re.compile(
    r"^(weight|activation|output)_(.+?)_(\d+)(?:_step(\d+))?\.pt$"
)


def parse_filename(fname: str) -> tuple[str, str, int, int | None] | None:
    m = _FILENAME_RE.match(fname)
    if not m:
        return None
    kind, layer_name, layer_idx, step = m.groups()
    return kind, layer_name, int(layer_idx), (int(step) if step is not None else None)


# =============================================================================
# Statistics
# =============================================================================


def compute_stats(t: torch.Tensor) -> dict:
    """Compute distribution statistics for a single tensor."""
    t = t.detach().float().flatten()
    n = t.numel()
    if n == 0:
        return {}

    abs_t = t.abs()

    def topk_energy_share(fraction: float) -> float:
        k = max(1, int(n * fraction))
        # Energy of the top-k largest (by abs) elements / total energy.
        topk = abs_t.topk(k).values
        total_energy = (t ** 2).sum().item()
        if total_energy == 0:
            return 0.0
        return (topk ** 2).sum().item() / total_energy

    max_abs = abs_t.max().item()

    def pct(p: float) -> float:
        return torch.quantile(abs_t, p).item()

    return {
        "numel": n,
        "max_abs": max_abs,
        "mean": t.mean().item(),
        "std": t.std().item(),
        "abs_p50": pct(0.50),
        "abs_p90": pct(0.90),
        "abs_p99": pct(0.99),
        "abs_p99_9": pct(0.999),
        "abs_p99_99": pct(0.9999),
        # tail decay: how close the tail quantiles are to the max (≈1 = fat tail)
        "p99_over_max": (pct(0.99) / max_abs) if max_abs > 0 else 0.0,
        "p99_9_over_max": (pct(0.999) / max_abs) if max_abs > 0 else 0.0,
        "p99_99_over_max": (pct(0.9999) / max_abs) if max_abs > 0 else 0.0,
        # top-k L2 energy share — directly maps to outlier_ratio
        "top0.01pct_energy": topk_energy_share(0.0001),
        "top0.1pct_energy": topk_energy_share(0.001),
        "top1pct_energy": topk_energy_share(0.01),
        "top5pct_energy": topk_energy_share(0.05),
    }


# =============================================================================
# Aggregation
# =============================================================================


def load_and_group(dump_dir: str) -> dict:
    """Group dumped tensors by (layer_name, layer_idx, kind).

    Returns:
        {(layer_name, layer_idx): {kind: [tensor, ...]}}
        Weight appears once; activation/output have one entry per step.
    """
    groups: dict = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(dump_dir)):
        if not fname.endswith(".pt"):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            print(f"  [skip] unrecognized filename: {fname}")
            continue
        kind, layer_name, layer_idx, step = parsed
        t = torch.load(os.path.join(dump_dir, fname), map_location="cpu")
        groups[(layer_name, layer_idx)][kind].append(t)
    return groups


# =============================================================================
# Plotting
# =============================================================================


def plot_histograms(groups: dict, out_dir: str) -> None:
    """Log-scale abs-value histograms per kind, aggregated across all layers."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, kind in zip(axes, ["activation", "weight", "output"]):
        all_abs = []
        for (layer_name, layer_idx), kinds in groups.items():
            for t in kinds.get(kind, []):
                all_abs.append(t.detach().float().flatten().abs())
        if not all_abs:
            ax.set_title(f"{kind} (no data)")
            continue
        cat = torch.cat(all_abs)
        # log-scale histogram of abs values
        vals = cat[cat > 0].log10().numpy()
        ax.hist(vals, bins=100, log=True)
        ax.set_title(f"{kind}: log10(abs) distribution (all layers)")
        ax.set_xlabel("log10(abs value)")
        ax.set_ylabel("count (log)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_log_abs.png"), dpi=120)
    plt.close(fig)


def plot_weight_topk_energy(groups: dict, out_dir: str) -> None:
    """Top-k energy share per weight layer (weight is the SQNR culprit)."""
    rows = []
    for (layer_name, layer_idx), kinds in sorted(groups.items()):
        if "weight" not in kinds:
            continue
        s = compute_stats(kinds["weight"][0])
        rows.append(
            (
                f"{layer_name}_{layer_idx}",
                s["top0.01pct_energy"],
                s["top0.1pct_energy"],
                s["top1pct_energy"],
                s["top5pct_energy"],
            )
        )
    if not rows:
        print("  [plot] no weight data for top-k energy plot")
        return

    labels = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.35), 5))
    x = range(len(rows))
    ax.plot(x, [r[1] for r in rows], label="top 0.01%", marker="o")
    ax.plot(x, [r[2] for r in rows], label="top 0.1%", marker="o")
    ax.plot(x, [r[3] for r in rows], label="top 1%", marker="o")
    ax.plot(x, [r[4] for r in rows], label="top 5%", marker="o")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("share of L2 energy")
    ax.set_title("Weight: top-k energy share (higher = fatter tail / more outliers)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "weight_topk_energy.png"), dpi=120)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--dir", default=os.path.join(_repo_root, "outputs", "tensor_dump"))
    ap.add_argument("--out", default=os.path.join(_repo_root, "outputs", "tensor_dump_analysis"))
    args = ap.parse_args()

    dump_dir = args.dir
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning {dump_dir} ...")
    groups = load_and_group(dump_dir)
    print(f"  {len(groups)} layers found")

    # Compute per-layer, per-kind stats.
    summary = {}
    csv_lines = [
        "layer_name,layer_idx,kind,max_abs,std,p99_over_max,p99_9_over_max,"
        "top0.1pct_energy,top1pct_energy,top5pct_energy"
    ]
    for (layer_name, layer_idx), kinds in sorted(groups.items()):
        key = f"{layer_name}_{layer_idx}"
        summary[key] = {}
        for kind, tensors in kinds.items():
            # Aggregate across steps by concatenation for activation/output.
            cat = torch.cat([t.detach().float().flatten() for t in tensors])
            s = compute_stats(cat)
            summary[key][kind] = s
            csv_lines.append(
                f"{layer_name},{layer_idx},{kind},{s['max_abs']:.4g},{s['std']:.4g},"
                f"{s['p99_over_max']:.3f},{s['p99_9_over_max']:.3f},"
                f"{s['top0.1pct_energy']:.3f},{s['top1pct_energy']:.3f},"
                f"{s['top5pct_energy']:.3f}"
            )

    # Write JSON + CSV.
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "stats.csv"), "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    print(f"  stats written to {out_dir}/stats.json and stats.csv")

    # Plots.
    print("Plotting ...")
    plot_histograms(groups, out_dir)
    plot_weight_topk_energy(groups, out_dir)
    print(f"  plots written to {out_dir}/*.png")


if __name__ == "__main__":
    main()
