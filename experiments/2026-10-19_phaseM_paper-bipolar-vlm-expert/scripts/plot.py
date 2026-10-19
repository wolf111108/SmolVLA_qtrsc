#!/usr/bin/env python3
"""Phase latency / EMA plots and exact-overlap resource occupancy bins."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--bins", type=int, default=1000)
    a = p.parse_args()
    replay = a.run_dir / "replays/D3"
    s = json.loads((replay / "summary.json").read_text())
    if s["status"] != "complete":
        raise ValueError("Refusing to plot an incomplete replay")
    figs = replay / "timeline/figures"
    figs.mkdir(parents=True, exist_ok=True)
    phases = ["vlm_prefill", "expert_denoise_total"]
    colors = plt.cm.tab20.colors
    for field, filename, scale, label in [
        (
            "critical",
            "latency_sources",
            1000 / s["config"]["clock_hz"],
            "Critical-path latency (ms)",
        ),
        ("ema", "ema_reads", 1 / 2**20, "DRAM bus reads (MiB)"),
    ]:
        keys = sorted(
            set(
                k
                for phase in phases
                for k in s["groups"][phase][field]
                if field == "critical"
                or k.startswith("read.")
                and k.endswith(".bus_bytes")
            )
        )
        fig, ax = plt.subplots(figsize=(8, 3.4))
        bottom = np.zeros(2)
        for i, key in enumerate(keys):
            vals = np.array(
                [s["groups"][phase][field].get(key, 0) * scale for phase in phases]
            )
            ax.bar(
                np.arange(2),
                vals,
                bottom=bottom,
                width=0.45,
                color=colors[i % 20],
                label=key.replace("read.", "").replace(".bus_bytes", ""),
            )
            bottom += vals
        ax.set_xticks([0, 1], ["VLM prefill", "Expert denoise"])
        ax.set_ylabel(label)
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(figs / f"{filename}.png", dpi=180)
        plt.close(fig)
    if a.bins < 1:
        raise ValueError("--bins must be positive")
    resources = sorted(s["resource_busy_cycles"])
    edges = np.linspace(0, max(1, s["total_cycles"]), a.bins + 1)
    fig, ax = plt.subplots(figsize=(9, max(4, len(resources) * 0.24)))
    y = np.arange(len(resources))
    den = max(s["total_cycles"], 1)
    ax.barh(
        y + 0.18,
        [s["resource_busy_cycles"][r] / den for r in resources],
        height=0.34,
        label="Reserved time / total",
    )
    ax.barh(
        y - 0.18,
        [s["resource_service_cycles"].get(r, 0) / den for r in resources],
        height=0.34,
        label="Requested service / total",
    )
    ax.set_yticks(y, resources, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of target hardware time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "resource_utilization.png", dpi=180)
    plt.close(fig)
    occupancy = np.zeros((len(resources), a.bins))
    indices = {r: i for i, r in enumerate(resources)}
    events = replay / "timeline/events.csv.gz"
    details = {}
    if events.exists():
        with gzip.open(events, "rt") as f:
            for row in csv.DictReader(f):
                start, end = int(row["start"]), int(row["end"])
                meta = json.loads(row["metadata"])
                for r in json.loads(row["resources"]):
                    l = max(0, np.searchsorted(edges, start, side="right") - 1)
                    h = min(a.bins, np.searchsorted(edges, end, side="left") + 1)
                    if h > l:
                        occupancy[indices[r], l:h] += np.maximum(
                            0,
                            np.minimum(edges[l + 1 : h + 1], end)
                            - np.maximum(edges[l:h], start),
                        )
                phase = meta.get("component")
                if phase in ("vlm", "expert"):
                    # First GEMM-related window per phase, bounded for legibility.
                    if row["kind"] == "bsc.C1" and phase not in details:
                        details[phase] = (start, [])
                    if phase in details and start < details[phase][0] + 500:
                        details[phase][1].append(row)
        occupancy /= np.diff(edges)
        if np.max(occupancy) > 1 + 1e-9:
            raise AssertionError("Resource occupancy exceeds 100%")
        np.savez_compressed(
            replay / "timeline/occupancy_bins.npz",
            edges_cycles=edges,
            resources=np.array(resources),
            occupancy=occupancy,
        )
        fig, ax = plt.subplots(figsize=(11, max(4, len(resources) * 0.22)))
        im = ax.imshow(
            occupancy,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[0, s["latency_seconds"] * 1000, -0.5, len(resources) - 0.5],
            vmin=0,
            vmax=1,
            cmap="Blues",
        )
        fig.colorbar(im, ax=ax, label="Occupied fraction", fraction=0.025, pad=0.02)
        ax.axvline(
            s["groups"]["vlm_prefill"]["latency_seconds"] * 1000,
            color="tab:red",
            linewidth=0.8,
            linestyle="--",
            label="Prefill end",
        )
        ax.legend(loc="upper right", fontsize=7)
        ax.set_yticks(range(len(resources)), resources, fontsize=7)
        ax.set_xlabel("Target hardware time (ms)")
        fig.tight_layout()
        fig.savefig(figs / "pipeline_occupancy.png", dpi=180)
        plt.close(fig)
        for phase, (origin, rows) in details.items():
            rr = sorted({r for row in rows for r in json.loads(row["resources"])})
            fig, ax = plt.subplots(figsize=(11, 5))
            for row in rows:
                for r in json.loads(row["resources"]):
                    ax.broken_barh(
                        [
                            (
                                int(row["start"]) - origin,
                                int(row["end"]) - int(row["start"]),
                            )
                        ],
                        (rr.index(r) - 0.35, 0.7),
                        facecolors=colors[rr.index(r) % 20],
                    )
            ax.set_yticks(range(len(rr)), rr, fontsize=7)
            ax.set_xlim(0, 500)
            ax.set_xlabel("Cycles from first BSC tile")
            fig.tight_layout()
            fig.savefig(figs / f"detail_{phase}.png", dpi=180)
            plt.close(fig)
    comparison = a.run_dir / "comparison.json"
    if comparison.exists():
        data = json.loads(comparison.read_text())
        fig, ax = plt.subplots(figsize=(10, 3.5))
        x = np.arange(len(data))
        for i, phase in enumerate(phases):
            vals = [d["groups"][phase]["latency_seconds"] * 1000 for d in data]
            ax.bar(x + (i - 0.5) * 0.32, vals, width=0.3, label=phase)
        ax.set_xticks(x, [d["name"] for d in data], rotation=30, ha="right")
        ax.set_ylabel("Latency (ms)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs / "ablation_and_sensitivity.png", dpi=180)
        plt.close(fig)
    print(figs)


if __name__ == "__main__":
    main()
