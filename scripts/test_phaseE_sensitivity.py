"""Phase E — Sensitivity sweep (plan §19, §29).

Measures per-site output sensitivity by injecting deterministic Gaussian RMS
noise into exactly ONE physical module at a time (all others stay raw), then
forwarding a fixed synthetic input (prefix prefill + one denoise step) and
measuring how much the suffix hidden states shift vs. the raw baseline.

This is a *fast, calibration-free* first-pass sensitivity map (gaussian methods
need no scales). It ranks modules by output perturbation; the top candidates
can later be confirmed with a closed-loop LIBERO SR run.

Outputs:
    --out-dir/sensitivity.csv   module_id, component, layer, operator, site,
                                max_abs_diff, mean_abs_diff, relative_norm

Usage:
    # full sweep (64 MatMul + 224 Linear)
    python scripts/test_phaseE_sensitivity.py checkpoints/smolvla_base --sweep all

    # single target
    python scripts/test_phaseE_sensitivity.py checkpoints/smolvla_base \
        --module-id expert.layer.7.qk

    # grouped (component x operator), fast overview
    python scripts/test_phaseE_sensitivity.py checkpoints/smolvla_base --sweep group

    # tune noise
    python scripts/test_phaseE_sensitivity.py checkpoints/smolvla_base \
        --sweep group --method gaussian_rms_output --alpha 0.03 --site output
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from vla_tcs2.model_wrapper import (
    ModelWrapper,
    apply_sensitivity_target,
    switch_quantization_mode_all,
)
from vla_tcs2.quant_linear import QuantizedLinear
from vla_tcs2.quant_matmul import QuantizedMatMul


def build(path: str) -> ModelWrapper:
    cfg = {
        "model": {
            "type": "smolvla",
            "path": path,
            "device": "cuda",
            "overrides": {"n_action_steps": 1, "num_steps": 10},
        },
        "quantization": {
            "enabled": True,
            "method": "per_tensor",
            "quantize_matmul": True,
            "matmul_scale_granularity": "per_site",
            "linear_scale_granularity": "per_site",
            "scale_dir": "/tmp/_phaseE_sensitivity",
            "linear": {"enabled": True, "include": ["*"], "exclude": []},
        },
    }
    w = ModelWrapper(cfg)
    w.build(mode="raw")
    return w


def make_input(am):
    """Return a fixed synthetic (prefix, denoise) forward input."""
    vlm_h = am.get_vlm_model().text_model.config.hidden_size
    exp_h = am.lm_expert.config.hidden_size
    torch.manual_seed(0)
    B, S_p, S_s = 1, 4, 2
    prefix = torch.randn(B, S_p, vlm_h, device="cuda", dtype=torch.float32)
    mask = torch.ones(B, S_p, S_p, dtype=torch.bool, device="cuda")
    pos = torch.arange(S_p, device="cuda").unsqueeze(0)
    suffix = torch.randn(B, S_s, exp_h, device="cuda", dtype=torch.float32)
    full = torch.ones(B, S_s, S_p + S_s, dtype=torch.bool, device="cuda")
    pos2 = torch.arange(S_p, S_p + S_s, device="cuda").unsqueeze(0)
    return prefix, mask, pos, suffix, full, pos2


def run_forward(am, inp):
    prefix, mask, pos, suffix, full, pos2 = inp
    with torch.no_grad():
        _, cache = am.forward(mask, pos, None, [prefix, None], use_cache=True)
        out, _ = am.forward(full, pos2, cache, [None, suffix], use_cache=True)
    return out[1]  # suffix hidden states


def parse_module(module_id: str):
    """Split module_id into (component, layer, operator)."""
    parts = module_id.split(".")
    component = parts[0]
    layer = None
    operator = parts[-1]
    for i, p in enumerate(parts):
        if p in ("layers", "layer"):
            layer = int(parts[i + 1])
            break
    return component, layer, operator


def all_module_ids(model):
    ids = []
    for m in model.modules():
        if isinstance(m, (QuantizedLinear, QuantizedMatMul)):
            ids.append(m.module_id)
    return ids


def grouped_module_ids(model):
    """One representative module_id per (component, operator) group."""
    reps = {}
    for mid in all_module_ids(model):
        comp, layer, op = parse_module(mid)
        reps.setdefault((comp, op), mid)
    return list(reps.values())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default="checkpoints/smolvla_base")
    p.add_argument("--sweep", choices=["all", "group"], default="all")
    p.add_argument("--module-id", default=None, help="single target module_id")
    p.add_argument("--method", default="gaussian_rms_output")
    p.add_argument("--alpha", type=float, default=0.03)
    p.add_argument("--site", default="output",
                   help="Linear: input|weight|output; MatMul: A|B|output")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-outlier-protection", action="store_true")
    p.add_argument("--out-dir", default="outputs/sensitivity")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w = build(args.checkpoint)
    am = w.model.model.vlm_with_expert
    inp = make_input(am)

    # Baseline (all raw).
    switch_quantization_mode_all(w.model, "raw")
    base = run_forward(am, inp)

    if args.module_id:
        targets = [args.module_id]
    elif args.sweep == "group":
        targets = grouped_module_ids(w.model)
    else:
        targets = all_module_ids(w.model)

    print(f"targets: {len(targets)}")
    rows = []
    for i, mid in enumerate(targets):
        comp, layer, op = parse_module(mid)
        cfg = {
            "test": {
                "enabled": True,
                "method": args.method,
                "alpha": args.alpha,
                "seed": args.seed,
                "site": args.site,
                "use_outlier_protection": args.use_outlier_protection,
                "target": {"module_id": mid},
            }
        }
        n = apply_sensitivity_target(w.model, cfg)
        if n != 1:
            print(f"  WARNING: {mid} matched {n} modules (expected 1)")
        out = run_forward(am, inp)
        d = (out - base).abs()
        max_abs = d.max().item()
        mean_abs = d.mean().item()
        rel = (d.norm() / (base.norm() + 1e-12)).item()
        rows.append({
            "module_id": mid,
            "component": comp,
            "layer": layer,
            "operator": op,
            "max_abs_diff": f"{max_abs:.6e}",
            "mean_abs_diff": f"{mean_abs:.6e}",
            "relative_norm": f"{rel:.6e}",
        })
        if (i + 1) % 32 == 0 or i == len(targets) - 1:
            print(f"  [{i + 1}/{len(targets)}] {mid} max|diff|={max_abs:.3e}")

    # Restore raw before exiting.
    switch_quantization_mode_all(w.model, "raw")

    # Sort by max_abs_diff descending (most sensitive first).
    rows.sort(key=lambda r: -float(r["max_abs_diff"]))

    csv_path = out_dir / "sensitivity.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {csv_path} ({len(rows)} rows)")
    print("\nTop 10 most sensitive:")
    for r in rows[:10]:
        print(f"  {r['module_id']:36s} {r['max_abs_diff']}")


if __name__ == "__main__":
    main()
