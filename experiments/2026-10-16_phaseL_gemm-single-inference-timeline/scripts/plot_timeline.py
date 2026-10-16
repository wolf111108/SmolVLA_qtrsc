#!/usr/bin/env python3
"""Plot exact resource occupancy binned over inference, utilization and local GEMM Gantt windows."""
import argparse
import csv
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=1000)
    parser.add_argument("--detail-cycles", type=int, default=10000)
    args = parser.parse_args()
    if args.bins < 1 or args.detail_cycles < 1:
        parser.error("bins/detail-cycles must be positive")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    import numpy as np
    directory = args.timeline_dir
    data = json.loads((directory / "summary.json").read_text())
    total, clock, resources = data["total_cycles"], data["clock_hz"], data["resources"]
    if total <= 0:
        raise ValueError("Cannot plot an empty timeline")
    args.bins = min(args.bins, total)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    ids = {resource: i for i, resource in enumerate(resources)}
    width = total / args.bins
    work = np.zeros((len(resources), args.bins))
    diff = np.zeros((len(resources), args.bins + 1))
    with gzip.open(directory / "events.csv.gz", "rt") as stream:
        for event in csv.DictReader(stream):
            start, end = int(event["start_cycle"]), int(event["end_cycle"])
            if end <= start:
                continue
            first = min(args.bins - 1, start * args.bins // total)
            last = min(args.bins - 1, (end * args.bins - 1) // total)
            for resource in event["resources"].split("|"):
                i = ids[resource]
                if first == last:
                    work[i, first] += end - start
                else:
                    work[i, first] += (first + 1) * width - start
                    work[i, last] += end - last * width
                    diff[i, first + 1] += 1
                    diff[i, last] -= 1
    occupancy = (work + np.cumsum(diff[:, :-1], axis=1) * width) / width
    if occupancy.max() > 1 + 1e-8 or occupancy.min() < -1e-8:
        raise ValueError("Overlapping reservations on an exclusive physical resource")
    occupancy = np.clip(occupancy, 0, 1)
    np.savez_compressed(directory / "occupancy_bins.npz", reserved_fraction=occupancy,
                        bin_start_cycles=np.arange(args.bins) * width, resources=np.array(resources), clock_hz=clock)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        fig.savefig(figures / (name + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(figures / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, max(4, len(resources) * .28)))
    image = ax.imshow(occupancy, interpolation="nearest", aspect="auto", cmap="Blues", norm=Normalize(0, 1),
                      extent=(0, total / clock * 1e3, len(resources) - .5, -.5))
    ax.set_yticks(range(len(resources)), resources)
    ax.set_xlabel("Simulated GEMM-only time (ms); operators concatenated in capture order")
    ax.set_title("Resource reservation occupancy over one action chunk")
    # Exact stage boundaries, not host or non-GEMM time.
    boundary = 0
    for key, stage in data["stages"].items():
        if boundary:
            ax.axvline(boundary / clock * 1e3, color="#ef9b48", linewidth=.7, alpha=.7)
        boundary += stage["cycles"]
    fig.colorbar(image, ax=ax, label="Fraction of bin reserved")
    save(fig, "pipeline_occupancy")
    with (directory / "resources.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(9, max(4, len(rows) * .35)))
    ys = np.arange(len(rows))
    ax.barh(ys - .17, [float(r["reserved_fraction"]) * 100 for r in rows], height=.3, color="#487caa", label="Reserved / locked")
    ax.barh(ys + .17, [float(r["service_fraction"]) * 100 for r in rows], height=.3, color="#e7a86b", label="Service demand / total time")
    ax.set_yticks(ys, [r["resource"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Fraction of complete GEMM timeline (%)")
    ax.set_title(f"Resource occupancy and service demand; effective PE utilization {data['effective_pe_utilization']:.1%}")
    ax.legend(loc="lower right")
    save(fig, "resource_utilization")
    colors = {"load": "#487caa", "feed": "#65aaa0", "compute": "#e7a86b",
              "drain": "#9a7eaf", "store": "#bd7581", "spill": "#9a7eaf", "restore": "#487caa"}
    with gzip.open(directory / "representatives.json.gz", "rt") as stream:
        representatives = json.load(stream)
    for component, record in representatives.items():
        window = min(args.detail_cycles, record["total_cycles"])
        fig, ax = plt.subplots(figsize=(12, max(4, len(resources) * .28)))
        for timing in record["timeline"]:
            start, end = timing["start_cycle"], min(window, timing["end_cycle"])
            if start >= window or end <= start:
                continue
            operation = timing["event_id"].rsplit("/", 1)[-1].split("_")[0]
            for resource in timing["resources"]:
                ax.broken_barh([(start / clock * 1e6, (end - start) / clock * 1e6)],
                               (ids[resource] - .35, .7), facecolors=colors.get(operation, "#aaaaaa"))
        ax.set_yticks(range(len(resources)), resources)
        ax.invert_yaxis()
        ax.set_xlim(0, window / clock * 1e6)
        ax.set_xlabel("Local GEMM time (µs); colored intervals are resource reservations")
        ax.set_title(f"{component}: {record['module_id']} (operator {record['operator_id']}); first {window} cycles")
        ax.legend(handles=[Patch(color=colors[k], label=k) for k in ("load", "feed", "compute", "drain", "store")],
                  ncol=5, loc="upper center", bbox_to_anchor=(.5, -.12), frameon=False)
        save(fig, f"detail_{component}")
    print(figures)


if __name__ == "__main__":
    main()
