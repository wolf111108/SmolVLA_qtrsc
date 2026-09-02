#!/usr/bin/env python
"""
Per-channel analysis of raw tensor dumps (companion to analyze_tensor_dump.py).

Motivation: the outlier protection in QuantizedLinear
(get_outlier_mask_channel) protects CHANNELS (top-k along the hidden dim),
so element-wise statistics (analyze_tensor_dump.py) do not directly map to
the protection granularity. This script aggregates along the channel
dimension to answer: "which channels hold the energy, and how many
channels should outlier_ratio protect?"

For each dumped tensor we reshape to [N, C] where C is the last dim
(the hidden dim used by get_outlier_mask_channel) and compute:

Per (layer, kind):
  - channel_absmax: max |x| over N for each channel
  - channel_energy: sum x^2 over N for each channel, normalized to share
  - top-k channel energy share for k in {0.1%, 0.5%, 1%, 5%} channels
    (this is the channel-level analogue of outlier_ratio)
  - energy of the channels selected by absmax-top-k (mirrors
    get_outlier_mask_channel, which ranks channels by absmax)

Outputs (to --out dir):
  channel_stats.json    full per-layer per-kind stats
  channel_stats.csv     flat table sorted by weight / activation
  plot_channel_energy_share.png   per-layer top-k channel energy curves

Usage:
    python scripts/analyze_tensor_dump_channel.py \
        [--dir outputs/tensor_dump] [--out outputs/tensor_dump_channel_analysis]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


_FILENAME_RE = re.compile(
    r"^(weight|activation|output)_(.+?)_(\d+)(?:_step(\d+))?\.pt$"
)


def parse_filename(fname: str):
    m = _FILENAME_RE.match(fname)
    if not m:
        return None
    kind, layer_name, layer_idx, step = m.groups()
    return kind, layer_name, int(layer_idx), (int(step) if step is not None else None)


def compute_channel_stats(t: torch.Tensor) -> dict:
    """
    Channel-level statistics for one tensor.

    Channels are the LAST dimension (hidden dim), matching
    get_outlier_mask_channel's reshape(-1, shape[-1]).
    """
    t = t.detach().float()
    t2d = t.reshape(-1, t.shape[-1])          # [N, C]
    n, c = t2d.shape
    if n == 0 or c == 0:
        return {}

    abs_t = t2d.abs()

    # Per-channel absmax (the score get_outlier_mask_channel ranks by).
    channel_absmax = abs_t.amax(dim=0)        # [C]

    # Per-channel energy and its share of total.
    channel_energy = (t2d ** 2).sum(dim=0)    # [C]
    total_energy = channel_energy.sum().item()
    if total_energy <= 0:
        return {"numel": n * c, "channels": c, "total_energy": 0.0}

    energy_share_sorted, _ = torch.sort(channel_energy, descending=True)

    def top_channels_energy_share(fraction: float) -> float:
        """Energy share of the top-k channels BY ENERGY (k = fraction * C)."""
        k = max(1, int(c * fraction))
        return energy_share_sorted[:k].sum().item() / total_energy

    def absmax_top_channels_energy_share(fraction: float) -> float:
        """
        Energy share of the top-k channels BY ABSMAX — this mirrors the
        actual protection policy (get_outlier_mask_channel ranks by absmax).
        """
        k = max(1, int(c * fraction))
        idx = channel_absmax.topk(k).indices
        return channel_energy[idx].sum().item() / total_energy

    sorted_absmax, _ = torch.sort(channel_absmax, descending=True)

    return {
        "n_rows": n,
        "channels": c,
        "total_energy": total_energy,
        # channel absmax decay: how flat is the channel-absmax profile
        "ch_absmax_p50_over_max": (
            (torch.quantile(channel_absmax, 0.50) / channel_absmax.max()).item()
            if channel_absmax.max() > 0 else 0.0
        ),
        "ch_absmax_p90_over_max": (
            (torch.quantile(channel_absmax, 0.90) / channel_absmax.max()).item()
            if channel_absmax.max() > 0 else 0.0
        ),
        # energy share of the top-k highest-energy channels
        "top0.1pct_ch_energy": top_channels_energy_share(0.001),
        "top0.5pct_ch_energy": top_channels_energy_share(0.005),
        "top1pct_ch_energy": top_channels_energy_share(0.01),
        "top5pct_ch_energy": top_channels_energy_share(0.05),
        # energy share of channels picked by ABSMAX ranking (policy-faithful)
        "absmax_top0.1pct_ch_energy": absmax_top_channels_energy_share(0.001),
        "absmax_top0.5pct_ch_energy": absmax_top_channels_energy_share(0.005),
        "absmax_top1pct_ch_energy": absmax_top_channels_energy_share(0.01),
        "absmax_top5pct_ch_energy": absmax_top_channels_energy_share(0.05),
        # raw channel profiles kept out of JSON by default (large); the
        # top-k channel ids can be rederived offline if needed.
    }


def plot_channel_energy_curves(stats: dict, out_dir: str) -> None:
    """Per-layer cumulative top-k channel energy curves, per kind."""
    kinds = ["weight", "activation", "output"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, kind in zip(axes, kinds):
        rows = []
        for key, kind_stats in stats.items():
            s = kind_stats.get(kind)
            if s is None:
                continue
            rows.append((
                key,
                s["top1pct_ch_energy"],
                s["top5pct_ch_energy"],
            ))
        if not rows:
            ax.set_title(f"{kind} (no data)")
            continue

        rows.sort(key=lambda r: r[1])  # sort by top1% share
        labels = [r[0] for r in rows]
        x = range(len(rows))

        ax.plot(x, [r[1] for r in rows], label="top 1% channels", marker=".")
        ax.plot(x, [r[2] for r in rows], label="top 5% channels", marker=".")

        # only label a subset of xticks to stay readable
        step = max(1, len(labels) // 25)
        ax.set_xticks(list(x)[::step])
        ax.set_xticklabels(labels[::step], rotation=90, fontsize=6)
        ax.set_ylabel("share of total L2 energy")
        ax.set_title(f"{kind}: energy in top-k channels (sorted by top-1%)")
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_channel_energy_share.png"), dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="/home/zyzhao/VLA_tcs2/outputs/tensor_dump")
    ap.add_argument("--out", default="/home/zyzhao/VLA_tcs2/outputs/tensor_dump_channel_analysis")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Group tensors per (layer, kind); activation/output have one tensor
    # per calibration step and are concatenated along the row dim later.
    print(f"Scanning {args.dir} ...")
    grouped: dict = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(args.dir)):
        if not fname.endswith(".pt"):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        kind, layer_name, layer_idx, _ = parsed
        grouped[(layer_name, layer_idx)][kind].append(
            torch.load(os.path.join(args.dir, fname), map_location="cpu")
        )

    stats: dict = {}
    csv_lines = [
        "layer,kind,channels,ch_absmax_p50_over_max,"
        "top0.1pct_ch_energy,top1pct_ch_energy,top5pct_ch_energy,"
        "absmax_top1pct_ch_energy,absmax_top5pct_ch_energy"
    ]

    for (layer_name, layer_idx), kinds in sorted(grouped.items()):
        key = f"{layer_name}_{layer_idx}"
        stats[key] = {}
        for kind, tensors in kinds.items():
            # Concatenate along rows (steps share the channel dim).
            cat = torch.cat(
                [t.reshape(-1, tensors[0].shape[-1]) for t in tensors], dim=0
            )
            s = compute_channel_stats(cat)
            stats[key][kind] = s
            csv_lines.append(
                f"{key},{kind},{s.get('channels', 0)},"
                f"{s.get('ch_absmax_p50_over_max', 0):.3f},"
                f"{s.get('top0.1pct_ch_energy', 0):.3f},"
                f"{s.get('top1pct_ch_energy', 0):.3f},"
                f"{s.get('top5pct_ch_energy', 0):.3f},"
                f"{s.get('absmax_top1pct_ch_energy', 0):.3f},"
                f"{s.get('absmax_top5pct_ch_energy', 0):.3f}"
            )

    with open(os.path.join(args.out, "channel_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out, "channel_stats.csv"), "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"  stats -> {args.out}/channel_stats.{{json,csv}}")

    print("Plotting ...")
    plot_channel_energy_curves(stats, args.out)
    print(f"  plots -> {args.out}/plot_channel_energy_share.png")


if __name__ == "__main__":
    main()
