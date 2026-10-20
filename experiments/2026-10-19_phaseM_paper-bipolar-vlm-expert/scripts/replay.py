#!/usr/bin/env python3
"""Replay without torch/checkpoint/GPU; multiple groups reuse ONE captured workload.

Groups are independent processes when --jobs > 1: each writes its own output
directory, and comparison.json is assembled at the end from per-group summaries.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def group_names(suite):
    names = ["D3"]
    if suite:
        names += ["D0", "D1", "D2", "D3_bw6", "D3_bw24",
                  "D3_ideal_memory", "D3_ideal_bsc", "D3_aux0.5", "D3_aux2"]
    return names


def group_configs(base, suite):
    runs = [("D3", base)]
    if suite:
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
    return runs


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
    p.add_argument(
        "--timeline-window",
        type=int,
        nargs=2,
        metavar=("START_CYCLE", "END_CYCLE"),
        help="Export detailed transactions only inside this cycle window",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print per-operator progress every N operators",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Run independent groups as parallel processes",
    )
    p.add_argument("--only", help="Comma-separated subset of group names")
    a = p.parse_args()

    def child(name):
        cmd = [sys.executable, __file__, "--run-dir", str(a.run_dir), "--suite"]
        if a.no_timeline:
            cmd.append("--no-timeline")
        if a.timeline_window:
            cmd += ["--timeline-window", *map(str, a.timeline_window)]
        cmd += ["--progress-every", str(a.progress_every), "--only", name]
        log = a.run_dir / "replays" / f"{name}.child.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        return subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT)

    if a.jobs > 1 and not a.only:
        pending = group_names(a.suite)
        if a.only:
            pending = [n for n in pending if n in a.only.split(",")]
        procs = {name: child(name) for name in pending}
        failed = []
        for name, proc in procs.items():
            if proc.wait() != 0:
                failed.append(name)
        if failed:
            raise SystemExit(f"Parallel groups failed: {failed}")
        return

    from vla_tcs2.hardware.paper.architecture import PaperConfig
    from vla_tcs2.hardware.paper.replay import replay

    c = yaml.safe_load((a.run_dir / "resolved_config.yaml").read_text())
    base = PaperConfig(**c["paper_hardware"])
    runs = group_configs(base, a.suite)
    if a.only:
        wanted = a.only.split(",")
        runs = [(n, cfg) for n, cfg in runs if n in wanted]
    comparison = []
    for name, config in runs:
        out = a.run_dir / "replays" / name
        if out.exists():
            s = json.loads((out / "summary.json").read_text())
            if s.get("status") != "complete" or s.get("config") != asdict(config):
                raise ValueError(
                    f"Existing failed/mismatched replay: {out}; use a fresh directory"
                )
        else:
            s = replay(
                a.run_dir / "workload",
                out,
                config,
                timeline=not a.no_timeline and name == "D3",
                timeline_window=a.timeline_window,
                progress_every=a.progress_every,
            )
        comparison.append(
            {"name": name, "groups": s["groups"], "ema": s["ema"], "work": s["work"]}
        )
    if a.only:
        return
    # Assemble full comparison from every completed group directory.
    from vla_tcs2.hardware.paper.architecture import PaperConfig

    c = yaml.safe_load((a.run_dir / "resolved_config.yaml").read_text())
    base = PaperConfig(**c["paper_hardware"])
    comparison = []
    for name, _ in group_configs(base, True):
        s = json.loads((a.run_dir / "replays" / name / "summary.json").read_text())
        comparison.append(
            {"name": name, "groups": s["groups"], "ema": s["ema"], "work": s["work"]}
        )
    (a.run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main()
