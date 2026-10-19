#!/usr/bin/env python3
"""Replay without torch/checkpoint/GPU; multiple groups reuse ONE captured workload."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from vla_tcs2.hardware.paper.architecture import PaperConfig
from vla_tcs2.hardware.paper.replay import replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--suite",
        action="store_true",
        help="D0-D3, bandwidth and auxiliary sensitivity replays",
    )
    p.add_argument(
        "--no-timeline",
        action="store_true",
        help="Counters only; default saves full timeline for main D3",
    )
    a = p.parse_args()
    c = yaml.safe_load((a.run_dir / "resolved_config.yaml").read_text())
    base = PaperConfig(**c["paper_hardware"])
    runs = [("D3", base)]
    if a.suite:
        runs += [(g, replace(base, group=g)) for g in ("D0", "D1", "D2")]
        runs += [
            (f"D3_bw{bw}", replace(base, dram_bytes_per_second=bw * 1e9))
            for bw in (6, 24)
        ]
        runs += [
            ("D3_ideal_memory", replace(base, ideal_memory=True)),
            ("D3_ideal_bsc", replace(base, ideal_bsc=True)),
        ]
        runs += [(f"D3_aux{x}", replace(base, auxiliary_scale=x)) for x in (0.5, 2)]
    comparison = []
    for name, config in runs:
        out = a.run_dir / "replays" / name
        if out.exists():
            s = json.loads((out / "summary.json").read_text())
            if s.get("status") != "complete" or s.get("config") != __import__(
                "dataclasses"
            ).asdict(config):
                raise ValueError(
                    f"Existing failed/mismatched replay: {out}; use a fresh directory"
                )
        else:
            s = replay(
                a.run_dir / "workload",
                out,
                config,
                timeline=not a.no_timeline and name == "D3",
            )
        comparison.append(
            {"name": name, "groups": s["groups"], "ema": s["ema"], "work": s["work"]}
        )
    (a.run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main()
