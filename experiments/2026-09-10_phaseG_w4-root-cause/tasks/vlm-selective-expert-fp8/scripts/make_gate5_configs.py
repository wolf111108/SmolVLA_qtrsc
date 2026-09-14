#!/usr/bin/env python
"""生成 G6 Gate 5 的 task0×1 临时 config（复用正式 config，仅改 evaluation）。"""

from __future__ import annotations

import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)

CONFIGS = [
    "g6a_all_fp8_control",
    "g6b_vlm_attn_w4_expert_fp8",
    "g6c_vlm_mlp_w4_expert_fp8",
    "g6d_vlm_all_w4_expert_fp8",
]


def main():
    gen_dir = os.path.join(TASK, "generated", "gate5_task0x1")
    os.makedirs(gen_dir, exist_ok=True)

    for name in CONFIGS:
        src = os.path.join(TASK, "configs", f"{name}.yaml")
        cfg = yaml.safe_load(open(src))
        cfg["output_dir"] = (
            "outputs/2026-09-10_phaseG_w4-root-cause/tasks/"
            f"vlm-selective-expert-fp8/gate5_task0x1/{name}"
        )
        cfg.setdefault("evaluation", {}).setdefault("env", {})["task_ids"] = [0]
        cfg["evaluation"]["n_episodes"] = 1

        dst = os.path.join(gen_dir, f"{name}.yaml")
        with open(dst, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
