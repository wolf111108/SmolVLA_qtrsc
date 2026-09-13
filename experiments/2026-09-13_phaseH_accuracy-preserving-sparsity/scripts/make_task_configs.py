#!/usr/bin/env python
"""Phase H task-config generator (experiment_setup.md §11.3).

Generates one config per LIBERO task so each task's sparsity CSV and SR are
recorded separately (avoiding hand-maintained 20+ YAML copies). Only
output_dir / evaluation.env.task_ids / evaluation.n_episodes are overridden.

Usage:
    python make_task_configs.py --base s0_fp8_all_base.yaml --stage h0_smoke
    python make_task_configs.py --base s1_expert_w4_base.yaml --stage h1_10ep
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))

# stage -> (episodes_per_task default, task id range)
STAGES = {
    "h0_smoke": (1, range(1)),       # task 0 only
    "h1_10ep": (1, range(10)),
    "h2_30ep": (3, range(10)),
    "h3_100ep": (10, range(10)),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base config filename in configs/")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--episodes-per-task", type=int, default=None)
    p.add_argument(
        "--config-name",
        default=None,
        help="output config label (default: base filename without _base.yaml)",
    )
    p.add_argument("--tasks", default=None, help="comma list e.g. 0,1,2")
    return p.parse_args()


def main():
    args = parse_args()

    base_path = os.path.join(EXP_DIR, "configs", args.base)
    if not os.path.isfile(base_path):
        sys.exit(f"base config not found: {base_path}")

    config_name = args.config_name or args.base.replace("_base.yaml", "")
    ep_default, task_range = STAGES[args.stage]
    ep_per_task = args.episodes_per_task if args.episodes_per_task is not None else ep_default

    if args.tasks:
        task_ids = [int(t) for t in args.tasks.split(",")]
    else:
        task_ids = list(task_range)

    with open(base_path) as f:
        base = yaml.safe_load(f)

    exp_name = "2026-09-13_phaseH_accuracy-preserving-sparsity"
    out_root = f"outputs/{exp_name}/{args.stage}/{config_name}"

    for tid in task_ids:
        cfg = dict(base)
        cfg["output_dir"] = f"{out_root}/task{tid:02d}"
        cfg.setdefault("evaluation", {}).setdefault("env", {})["task_ids"] = [tid]
        cfg["evaluation"]["n_episodes"] = ep_per_task

        out_rel = f"generated/{args.stage}/{config_name}/task{tid:02d}.yaml"
        out_abs = os.path.join(EXP_DIR, out_rel)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        with open(out_abs, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"wrote {out_rel}")

    print(
        f"done: {len(task_ids)} configs -> "
        f"generated/{args.stage}/{config_name}/ (ep/task={ep_per_task})"
    )


if __name__ == "__main__":
    main()
