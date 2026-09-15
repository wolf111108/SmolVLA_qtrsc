#!/usr/bin/env python
"""Phase I V0 — Vision workload / structure / FLOPs audit.

Does NOT quantize. Loads the original SmolVLA policy, audits the vision
encoder + connector structure (static), then drives ONE real LIBERO inference
to record actual camera-call count and tensor shapes (runtime), and finally
computes per-operator dense theoretical FLOPs from those runtime shapes.

Outputs (into --out-dir):
    vision_structure.json
    vision_runtime_shapes.json
    vision_flops.csv
    vision_flops_summary.md

Usage:
  python audit_vision_structure.py --config <config.yaml> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

from vla_tcs2.eval import evaluate  # noqa: E402


# ---------------------------------------------------------------------------
# Static structure audit
# ---------------------------------------------------------------------------

def _dump_structure(policy: SmolVLAPolicy) -> dict:
    vlm_expert = policy.model.vlm_with_expert
    vlm_model = vlm_expert.get_vlm_model()
    vision_model = vlm_model.vision_model
    connector = vlm_model.connector

    vcfg = vision_model.config
    layers = vision_model.encoder.layers

    structure = {
        "vision_model_type": type(vision_model).__name__,
        "num_vision_layers": len(layers),
        "vision_config": {
            "hidden_size": getattr(vcfg, "hidden_size", None),
            "intermediate_size": getattr(vcfg, "intermediate_size", None),
            "num_attention_heads": getattr(vcfg, "num_attention_heads", None),
            "num_hidden_layers": getattr(vcfg, "num_hidden_layers", None),
            "image_size": getattr(vcfg, "image_size", None),
            "patch_size": getattr(vcfg, "patch_size", None),
            "num_channels": getattr(vcfg, "num_channels", None),
            "attn_implementation": getattr(
                vcfg, "_attn_implementation", None
            ),
        },
        "connector": {
            "scale_factor": getattr(connector, "scale_factor", None),
            "proj_in": getattr(
                connector.modality_projection.proj, "in_features", None
            ),
            "proj_out": getattr(
                connector.modality_projection.proj, "out_features", None
            ),
        },
        "per_layer": [],
    }

    for i, layer in enumerate(layers):
        attn = layer.self_attn
        mlp = layer.mlp
        entry = {
            "layer_idx": i,
            "self_attn": {
                name: {
                    "type": type(getattr(attn, name)).__name__,
                    "in": getattr(attn, name).in_features,
                    "out": getattr(attn, name).out_features,
                }
                for name in ("q_proj", "k_proj", "v_proj", "out_proj")
            },
            "mlp": {
                name: {
                    "type": type(getattr(mlp, name)).__name__,
                    "in": getattr(mlp, name).in_features,
                    "out": getattr(mlp, name).out_features,
                }
                for name in ("fc1", "fc2")
            },
        }
        structure["per_layer"].append(entry)

    # Verify every layer has the expected Linear projection / MLP.
    for i, layer in enumerate(layers):
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert isinstance(getattr(layer.self_attn, name), nn.Linear), (
                f"vision layer {i} self_attn.{name} is not nn.Linear"
            )
        for name in ("fc1", "fc2"):
            assert isinstance(getattr(layer.mlp, name), nn.Linear), (
                f"vision layer {i} mlp.{name} is not nn.Linear"
            )

    return structure


# ---------------------------------------------------------------------------
# Runtime hooks (camera calls + shapes)
# ---------------------------------------------------------------------------

class RuntimeRecorder:
    def __init__(self):
        self.vision_calls = 0
        self.vision_input_shapes = []
        self.vision_output_shapes = []
        self.connector_calls = 0
        self.connector_input_shapes = []
        self.connector_output_shapes = []
        self.sample_actions_calls = 0
        self.handles = []

    def _vision_pre(self, module, args):
        self.vision_calls += 1
        pv = args[0] if args else None
        if pv is not None:
            self.vision_input_shapes.append(list(pv.shape))

    def _vision_post(self, module, args, output):
        hs = getattr(output, "last_hidden_state", None)
        if hs is not None:
            self.vision_output_shapes.append(list(hs.shape))

    def _connector_pre(self, module, args):
        self.connector_calls += 1
        x = args[0] if args else None
        if x is not None:
            self.connector_input_shapes.append(list(x.shape))

    def _connector_post(self, module, args, output):
        self.connector_output_shapes.append(list(output.shape))

    def attach(self, vision_model, connector, policy):
        self.handles.append(
            vision_model.register_forward_pre_hook(self._vision_pre)
        )
        self.handles.append(
            vision_model.register_forward_hook(self._vision_post)
        )
        self.handles.append(
            connector.register_forward_pre_hook(self._connector_pre)
        )
        self.handles.append(
            connector.register_forward_hook(self._connector_post)
        )
        # Monkey-patch _get_action_chunk (not a forward; can't use forward
        # hooks) to count sample_actions invocations and derive cameras
        # per sample_actions.
        self._orig_get_action_chunk = policy._get_action_chunk

        def _wrapped(batch, noise=None, **kwargs):
            self.sample_actions_calls += 1
            return self._orig_get_action_chunk(batch, noise, **kwargs)

        policy._get_action_chunk = _wrapped

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        if hasattr(self, "_orig_get_action_chunk"):
            # policy reference stored on attach; restore via closure is
            # handled by caller simply dropping the recorder. Best-effort.
            pass

    def summary(self) -> dict:
        n_cam = (
            self.vision_calls / self.sample_actions_calls
            if self.sample_actions_calls else 0.0
        )
        return {
            "vision_forward_calls": self.vision_calls,
            "sample_actions_calls": self.sample_actions_calls,
            "cameras_per_sample_actions": n_cam,
            "vision_input_shapes": self.vision_input_shapes,
            "vision_output_shapes": self.vision_output_shapes,
            "connector_calls": self.connector_calls,
            "connector_input_shapes": self.connector_input_shapes,
            "connector_output_shapes": self.connector_output_shapes,
        }


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------

def _compute_flops(structure: dict, runtime: dict) -> list[dict]:
    """Dense theoretical FLOPs per operator from runtime shapes (1 MAC = 2 FLOPs)."""
    vcfg = structure["vision_config"]
    H = vcfg["hidden_size"]
    I = vcfg["intermediate_size"]
    n_layers = structure["num_vision_layers"]

    # cameras per sample_actions (NOT total calls): FLOPs are per
    # sample_actions(), matching the manual §0 table.
    n_cameras = runtime.get("cameras_per_sample_actions", 2.0)
    if not n_cameras:
        n_cameras = 2.0

    # Tokens per camera from the vision output shape [B, T, H].
    T = None
    if runtime["vision_output_shapes"]:
        T = runtime["vision_output_shapes"][0][1]  # [B, T, H]
    if T is None:
        # Fallback: derive from image_size / patch_size.
        img = vcfg["image_size"]
        patch = vcfg["patch_size"]
        T = (img // patch) ** 2

    B = 1  # batch = 1 for eval
    rows = []

    def add(op_type, operator, flops, note=""):
        rows.append({
            "component": "vision",
            "op_type": op_type,
            "operator": operator,
            "flops": flops,
            "percent_of_vision": 0.0,  # filled later
            "note": note,
        })

    # Attention projection q/k/v/out: each 2 * B * T * H * H.
    attn_proj_flops = 4 * (2 * B * T * H * H)
    add("linear", "qkvout_proj", attn_proj_flops)

    # QK^T + P*V: self-attention both ~ 2 * B * T * T * H (heads cancel).
    qk_flops = 2 * B * T * T * H
    pv_flops = 2 * B * T * T * H
    add("matmul", "qk", qk_flops)
    add("matmul", "pv", pv_flops)

    # MLP fc1 + fc2: fc1 2*B*T*H*I, fc2 2*B*T*I*H.
    mlp_flops = 4 * B * T * H * I
    add("linear", "fc1_fc2", mlp_flops)

    # Per-layer totals (× n_layers) then scale by camera count.
    per_layer = {
        "attn_proj": attn_proj_flops,
        "qk": qk_flops,
        "pv": pv_flops,
        "mlp": mlp_flops,
    }

    # Connector: proj_in -> proj_out on T_conn tokens.
    conn = structure["connector"]
    c_in = conn["proj_in"]
    c_out = conn["proj_out"]
    T_conn = None
    if runtime["connector_output_shapes"]:
        T_conn = runtime["connector_output_shapes"][0][1]
    if T_conn is None:
        T_conn = T // (conn["scale_factor"] ** 2)
    connector_flops = 2 * B * T_conn * c_in * c_out

    # Build final per-component breakdown (already per-camera; multiply below).
    total_vision_per_camera = (
        attn_proj_flops + qk_flops + pv_flops + mlp_flops
    ) * n_layers + connector_flops

    grand = total_vision_per_camera * n_cameras

    out_rows = [
        {
            "component": "vision",
            "operator": "attention_projection",
            "flops": attn_proj_flops * n_layers * n_cameras,
            "percent_of_vision": 0.0,
        },
        {
            "component": "vision",
            "operator": "qk",
            "flops": qk_flops * n_layers * n_cameras,
            "percent_of_vision": 0.0,
        },
        {
            "component": "vision",
            "operator": "pv",
            "flops": pv_flops * n_layers * n_cameras,
            "percent_of_vision": 0.0,
        },
        {
            "component": "vision",
            "operator": "mlp",
            "flops": mlp_flops * n_layers * n_cameras,
            "percent_of_vision": 0.0,
        },
        {
            "component": "connector",
            "operator": "connector_proj",
            "flops": connector_flops * n_cameras,
            "percent_of_vision": 0.0,
        },
    ]

    total = sum(r["flops"] for r in out_rows)
    for r in out_rows:
        r["percent_of_vision"] = r["flops"] / total if total else 0.0

    # Attach meta.
    meta = {
        "n_cameras": n_cameras,
        "tokens_per_camera": T,
        "connector_tokens": T_conn,
        "total_flops": total,
    }
    return out_rows, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    model_path = model_cfg.get("path", "lerobot/smolvla_libero")
    revision = model_cfg.get("revision", None)

    policy_cfg = SmolVLAConfig.from_pretrained(model_path, revision=revision)
    policy_cfg.pretrained_path = Path(model_path)
    if "device" in model_cfg:
        policy_cfg.device = model_cfg["device"]
    for name, value in model_cfg.get("overrides", {}).items():
        setattr(policy_cfg, name, value)

    print(f"Loading SmolVLA from: {model_path}")
    policy = SmolVLAPolicy.from_pretrained(
        model_path, config=policy_cfg, revision=revision
    )
    policy.eval()

    # ---- Static structure ----
    structure = _dump_structure(policy)
    (out_dir / "vision_structure.json").write_text(
        json.dumps(structure, indent=2)
    )
    print(
        f"[V0] vision layers={structure['num_vision_layers']} "
        f"hidden={structure['vision_config']['hidden_size']} "
        f"attn_impl={structure['vision_config']['attn_implementation']}"
    )

    # ---- Runtime: hooks + one real inference ----
    vlm_model = policy.model.vlm_with_expert.get_vlm_model()
    vision_model = vlm_model.vision_model
    connector = vlm_model.connector

    recorder = RuntimeRecorder()
    recorder.attach(vision_model, connector, policy)

    print(f"[V0] running {args.episodes} real inference episode(s)...")
    evaluate(
        model=policy,
        config=config,
        output_dir=out_dir / "v0_eval",
    )

    runtime = recorder.summary()
    recorder.detach()
    (out_dir / "vision_runtime_shapes.json").write_text(
        json.dumps(runtime, indent=2)
    )
    print(
        f"[V0] vision_forward_calls={runtime['vision_forward_calls']} "
        f"sample_actions_calls={runtime['sample_actions_calls']} "
        f"cameras_per_sample={runtime['cameras_per_sample_actions']}"
    )

    # ---- FLOPs ----
    flops_rows, meta = _compute_flops(structure, runtime)
    with open(out_dir / "vision_flops.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=[
                "component", "operator", "flops", "percent_of_vision",
            ]
        )
        w.writeheader()
        for r in flops_rows:
            w.writerow(r)

    # summary markdown
    lines = [
        "# Vision workload audit (V0)",
        "",
        f"- vision layers: {structure['num_vision_layers']}",
        f"- hidden: {structure['vision_config']['hidden_size']}",
        f"- intermediate: {structure['vision_config']['intermediate_size']}",
        f"- attention impl: {structure['vision_config']['attn_implementation']}",
        f"- runtime camera calls / sample_actions: {meta['n_cameras']}",
        f"- tokens per camera: {meta['tokens_per_camera']}",
        f"- connector tokens: {meta['connector_tokens']}",
        "",
        "| operator | FLOPs | % of vision |",
        "|---|---:|---:|",
    ]
    for r in flops_rows:
        lines.append(
            f"| {r['component']}.{r['operator']} | {r['flops']:.3e} | "
            f"{r['percent_of_vision']*100:.2f}% |"
        )
    lines.append("")
    lines.append(f"Total vision+connector FLOPs / sample_actions: {meta['total_flops']:.3e}")
    (out_dir / "vision_flops_summary.md").write_text("\n".join(lines) + "\n")

    print(f"[V0] outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
