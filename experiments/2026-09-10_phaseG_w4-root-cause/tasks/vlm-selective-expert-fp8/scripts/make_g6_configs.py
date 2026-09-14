#!/usr/bin/env python
"""Generate G6 configs (Expert-FP8 background VLM selective precision).

Four configs share the same base (VLM + Expert all Linear FP8, MatMul FP8);
only the `linear.overrides` and output/scale dirs differ:

  g6a_all_fp8_control     : no override
  g6b_vlm_attn_w4         : vlm.layers.*.self_attn.*_proj -> W4
  g6c_vlm_mlp_w4          : vlm.layers.*.mlp.*_proj       -> W4
  g6d_vlm_all_w4          : vlm.layers.*.*.*_proj         -> W4
"""

from __future__ import annotations

import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)

EXPERIMENT = "2026-09-10_phaseG_w4-root-cause"
TASK_NAME = "vlm-selective-expert-fp8"


def _operator(name, w_bit="e4m3"):
    return {
        "a_bit": "e4m3",
        "w_bit": w_bit,
        "o_bit": "e4m3",
        "d_bit": 4,
        "outlier_ratio": 0.01,
        "p": 4,
    }


def _base(name, overrides):
    return {
        "output_dir": (
            f"outputs/{EXPERIMENT}/tasks/{TASK_NAME}/{name}"
        ),
        "model": {
            "type": "smolvla",
            "path": "lerobot/smolvla_libero",
            "device": "cuda",
            "revision": None,
            "overrides": {"n_action_steps": 10, "num_steps": 10},
        },
        "quantization": {
            "enabled": True,
            "mode": "quant_forward",
            "method": "pot_fp8_outlier",
            "scale_dir": f"scales/{EXPERIMENT}/{TASK_NAME}/{name}",
            "quantize_matmul": True,
            "matmul_scale_granularity": "per_site",
            "linear_scale_granularity": "per_site",
            "outlier_ratio": 0.01,
            "calibration_policy": {
                "default": "auto",
                "layer_policy": {
                    "q_proj": "recalibrate",
                    "k_proj": "recalibrate",
                    "v_proj": "recalibrate",
                    "o_proj": "recalibrate",
                    "gate_proj": "recalibrate",
                    "up_proj": "recalibrate",
                    "down_proj": "recalibrate",
                    "qk_matmul": "recalibrate",
                    "pv_matmul": "recalibrate",
                },
                "per_layer_policy": {},
            },
            "q_proj": _operator("q_proj"),
            "k_proj": _operator("k_proj"),
            "v_proj": _operator("v_proj"),
            "o_proj": _operator("o_proj"),
            "gate_proj": _operator("gate_proj"),
            "up_proj": _operator("up_proj"),
            "down_proj": _operator("down_proj"),
            "qk_matmul": {
                "A_bit": "e4m3", "B_bit": "e4m3", "O_bit": "e4m3",
                "d_bit": 4, "outlier_ratio": 0.01, "p": 4,
            },
            "pv_matmul": {
                "A_bit": "e4m3", "B_bit": "e4m3", "O_bit": "e4m3",
                "d_bit": 4, "outlier_ratio": 0.01, "p": 4,
            },
            "linear": {
                "enabled": True,
                "include": ["vlm.*", "expert.*"],
                "exclude": [],
                "overrides": overrides,
            },
            "weight_quant_granularity": "per_tensor",
            "weight_group_size": None,
        },
        "calibration": {
            "dataset_repo_id": "HuggingFaceVLA/libero",
            "dataset_revision": "v3.0",
            "episodes": 8,
            "batch_size": 8,
            "frame_stride": 4,
            "seed": 42,
        },
        "evaluation": {
            "env": {
                "type": "libero",
                "task": "libero_goal",
                "max_parallel_tasks": 1,
            },
            "n_episodes": 10,
            "batch_size": 1,
            "use_async_envs": False,
            "seed": 1000,
            "max_episodes_rendered": 0,
            "rename_map": {
                "observation.images.image": "observation.images.camera1",
                "observation.images.image2": "observation.images.camera2",
            },
        },
    }


def _w4_override(name, module_id_glob):
    return {
        "name": name,
        "target": {"module_id": module_id_glob},
        "config": {"w_bit": 4, "method": "pot_ao_outlier"},
    }


CONFIGS = {
    "g6a_all_fp8_control": [],
    "g6b_vlm_attn_w4_expert_fp8": [
        _w4_override("vlm_attention_w4", "vlm.layers.*.self_attn.*_proj"),
    ],
    "g6c_vlm_mlp_w4_expert_fp8": [
        _w4_override("vlm_mlp_w4", "vlm.layers.*.mlp.*_proj"),
    ],
    "g6d_vlm_all_w4_expert_fp8": [
        _w4_override("vlm_all_w4", "vlm.layers.*.*.*_proj"),
    ],
}


def main():
    cfg_dir = os.path.join(TASK, "configs")
    os.makedirs(cfg_dir, exist_ok=True)

    for name, overrides in CONFIGS.items():
        cfg = _base(name, overrides)
        path = os.path.join(cfg_dir, f"{name}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"wrote {name}.yaml")


if __name__ == "__main__":
    main()
