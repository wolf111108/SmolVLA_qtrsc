#!/usr/bin/env python
"""G0 weight-error-audit — offline numerical audit (no rollout).

Compare Linear weight quantization error under two protocols (FP8 vs W4,
both = Phase F F1/F3 protocols) for all wrapped Linear sites, using
calibration batches as forward data. For every physical Linear layer:

  weight-level: sqnr_db / nmse / cosine / unique_codes / saturation / protected_ratio
  output-level: sqnr_db / nmse / cosine (simulated quant vs FP reference)

Outputs (into --out-dir):
  linear_error_stats.csv     # all layers, both configs, all metrics
  linear_error_ranked.csv    # ranked by w4 output_nmse desc
  component_summary.csv      # vlm vs expert aggregation
  operator_summary.csv       # q/k/v/o/gate/up/down aggregation
  layer_summary.csv          # layer 0..15 aggregation

Usage (from repo root):
  MUJOCO_GL=egl python experiments/2026-09-10_phaseG_w4-root-cause/tasks/weight-error-audit/scripts/audit_weight_error.py \
      --config-fp8 experiments/.../configs/g0_fp8.yaml \
      --config-w4  experiments/.../configs/g0_w4.yaml \
      --out-dir outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-error-audit

Each config is expected to have been run with `main.py --skip-evaluation`
first (calibration writes scales into its scale_dir).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vla_tcs2.model_wrapper import ModelWrapper  # noqa: E402
from vla_tcs2.quant.quant_spec import quant_awo  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sqnr_db(ref: torch.Tensor, test: torch.Tensor, eps: float = 1e-12) -> float:
    signal = torch.sum(ref.float() ** 2).item()
    noise = torch.sum((ref.float() - test.float()) ** 2).item()
    if noise <= eps:
        return float("inf")
    return 10.0 * math.log10((signal + eps) / (noise + eps))


def _nmse(ref: torch.Tensor, test: torch.Tensor, eps: float = 1e-12) -> float:
    noise = torch.sum((ref.float() - test.float()) ** 2).item()
    signal = torch.sum(ref.float() ** 2).item()
    return noise / (signal + eps)


def _cosine(ref: torch.Tensor, test: torch.Tensor, eps: float = 1e-12) -> float:
    a = ref.float().flatten()
    b = test.float().flatten()
    return F.cosine_similarity(a, b, dim=0).clamp(-1, 1).item() if a.numel() else 1.0


# ---------------------------------------------------------------------------
# Per-layer audit
# ---------------------------------------------------------------------------

@torch.no_grad()
def audit_layer(ql, samples: list[torch.Tensor]) -> dict:
    """Audit one QuantizedLinear with cached calibration activations.

    Follows scales_with_outlier semantics: outlier channels (x) / elements (w)
    are protected in FP; metrics are computed over the *normal* part only
    (protected_ratio reported separately), per G0 setup §8.
    """
    from vla_tcs2.quant.scale_methods import (
        get_outlier_mask_1d,
        get_outlier_mask_channel,
    )

    ratio = getattr(ql, "outlier_ratio", 0.01)
    w = ql.weight.detach().float()
    x = torch.cat(samples, dim=0) if samples else torch.zeros(1, 1, w.size(1))

    channel_mask = get_outlier_mask_channel(x, ratio)          # [H]
    x_ch_mask = channel_mask.view(1, 1, -1)
    w_ch_mask = channel_mask.view(1, -1)
    w_mask = w_ch_mask | get_outlier_mask_1d(ql.weight, ratio)

    protected_ratio = w_mask.float().mean().item()
    x_normal = x * (~x_ch_mask).to(x.dtype)
    w_normal = w * (~w_mask).to(torch.float32)

    # --- weight metrics (normal part only) ---
    w_spec = ql.w_spec
    w_scale = ql.w_interval
    n_bits = int(getattr(w_spec, "bits", 0) or getattr(w_spec, "bit", 0))
    if w_scale is None:
        w_scale = w_normal.abs().max().item() / (2 ** (n_bits - 1))
    # NOTE: quant_awo returns the integer *code* ([-qmax, qmax-1]) for int
    # specs, not the dequantized value. Dequantize by scaling back.
    w_code = quant_awo(w_normal, w_scale, w_spec, out_dtype=torch.float32)
    w_sim = w_code * w_scale

    unique_codes = int(torch.unique(w_code.round().to(torch.int64)).numel())
    sat_mask = (w_normal.abs() / w_scale) > (2 ** (n_bits - 1)) + 0.5
    saturation_ratio = sat_mask.float().mean().item()

    # --- output metrics (normal part; FP activations input) ---
    out_ref = F.linear(x_normal.to(torch.float32), w_normal, None)
    out_sim = F.linear(x_normal.to(torch.float32), w_sim, None)

    return {
        "weight_scale": float(w_scale) if w_scale else 0.0,
        "weight_sqnr_db": _sqnr_db(w_normal, w_sim),
        "weight_nmse": _nmse(w_normal, w_sim),
        "weight_cosine": _cosine(w_normal, w_sim),
        "quant_zero_ratio": (w_sim == 0).float().mean().item(),
        "saturation_ratio": saturation_ratio,
        "unique_quant_codes": int(unique_codes),
        "protected_ratio": protected_ratio,
        "output_sqnr_db": _sqnr_db(out_ref, out_sim),
        "output_nmse": _nmse(out_ref, out_sim),
        "output_cosine": _cosine(out_ref, out_sim),
    }


def collect_activations(model, batches, max_samples: int = 8) -> dict[str, list]:
    """Run calibration forwards; cache per-module_id input activations."""
    store: dict[str, list] = defaultdict(list)

    def hook(mod, args, kwargs, out):
        mid = getattr(mod, "module_id", None)
        if mid is None or len(store[mid]) >= max_samples:
            return
        x = args[0] if args else kwargs.get("x")
        if x is not None and x.ndim == 3:  # [B, S, H]
            store[mid].append(x.detach().float().cpu())

    handles = []
    from vla_tcs2.quant_linear import QuantizedLinear

    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            handles.append(m.register_forward_hook(hook, with_kwargs=True))

    model.eval()
    with torch.no_grad():
        for b in batches[:max_samples]:
            model.predict_action_chunk(b)

    for h in handles:
        h.remove()
    return store


def parse_module_id(mid: str) -> tuple[str, int, str]:
    """`vlm.layers.3.mlp.down_proj` -> (vlm, 3, down_proj)."""
    parts = mid.split(".")
    comp = parts[0]
    idx = int(parts[2])
    op = parts[-1]
    return comp, idx, op


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_protocol(tag: str, config_path: Path, act_store: dict | None,
                     collect_acts: bool, max_samples: int) -> tuple[list[dict], dict | None]:
    config = yaml.safe_load(config_path.read_text())
    w = ModelWrapper(config)
    w.build(mode="scale_inspection")
    model = w.model

    # Reuse saved scales from the config's scale_dir (calibration already run).
    from vla_tcs2.quant_linear import QuantizedLinear

    n_loaded = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            try:
                m._load_scales()
                n_loaded += 1
            except FileNotFoundError:
                pass
    print(f"[{tag}] scales loaded: {n_loaded}")
    if n_loaded == 0:
        raise SystemExit(
            f"[{tag}] no scales found — run `main.py --config {config_path} "
            f"--skip-evaluation` first to calibrate."
        )

    acts = act_store
    if collect_acts:
        from vla_tcs2.calibration import prepare_calibration_batches

        calib_cfg = config.get("calibration", {})
        calib_cfg.setdefault(
            "rename_map",
            config.get("evaluation", {}).get("rename_map", {}) or {},
        )
        batches = prepare_calibration_batches(
            model=model,
            calib_cfg=calib_cfg,
            device=torch.device(config["model"].get("device", "cuda")),
            fallback_rename_map=calib_cfg["rename_map"],
        )
        acts = collect_activations(model, batches, max_samples=max_samples)

    rows = []
    for m in model.modules():
        if not isinstance(m, QuantizedLinear):
            continue
        mid = getattr(m, "module_id", "") or ""
        if not mid.startswith(("vlm.", "expert.")):
            continue  # 224 vlm/expert sites only
        comp, lidx, op = parse_module_id(mid)
        samples = [s.to(next(m.parameters()).device) for s in (acts.get(mid) or [])]
        row = {
            "config": tag,
            "module_name": mid,
            "component": comp,
            "layer_idx": lidx,
            "op_type": op,
            "weight_shape": "x".join(map(str, m.weight.shape)),
        }
        row.update(audit_layer(m, samples))
        rows.append(row)
        print(f"[{tag}] {mid}: w_sqnr={row['weight_sqnr_db']:.1f}dB "
              f"out_nmse={row['output_nmse']:.3e}")

    # Free GPU memory before the next protocol.
    del model, w
    torch.cuda.empty_cache()
    return rows, acts


def write_csvs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())

    stats_p = out_dir / "linear_error_stats.csv"
    with stats_p.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    w4 = [r for r in rows if r["config"] == "w4"]
    ranked = sorted(w4, key=lambda r: r["output_nmse"], reverse=True)
    with (out_dir / "linear_error_ranked.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(ranked)

    def agg(key_fn):
        g: dict[tuple, list] = defaultdict(list)
        for r in w4:
            g[key_fn(r)].append(r)
        return {
            k: {
                "count": len(v),
                "w4_weight_sqnr_db_mean": sum(x["weight_sqnr_db"] for x in v) / len(v),
                "w4_weight_nmse_mean": sum(x["weight_nmse"] for x in v) / len(v),
                "w4_output_nmse_mean": sum(x["output_nmse"] for x in v) / len(v),
                "w4_output_sqnr_db_mean": sum(x["output_sqnr_db"] for x in v) / len(v),
                "w4_unique_codes_mean": sum(x["unique_quant_codes"] for x in v) / len(v),
            }
            for k, v in sorted(g.items())
        }

    def write_dict_csv(path: Path, data: dict, keyname: str):
        if not data:
            return
        keys = [keyname] + list(next(iter(data.values())).keys())
        with path.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for k, v in data.items():
                wr.writerow({keyname: k, **v})

    write_dict_csv(out_dir / "component_summary.csv",
                   agg(lambda r: r["component"]), "component")
    write_dict_csv(out_dir / "operator_summary.csv",
                   agg(lambda r: r["op_type"]), "op_type")
    write_dict_csv(out_dir / "layer_summary.csv",
                   agg(lambda r: (r["component"], r["layer_idx"])),
                   "component_layer")

    print(f"\n✓ CSV written to {out_dir}")
    print(f"  - {stats_p.name}, linear_error_ranked.csv, "
          f"component_summary.csv, operator_summary.csv, layer_summary.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-fp8", required=True, type=Path)
    ap.add_argument("--config-w4", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-samples", type=int, default=8,
                    help="calibration batches used as forward data")
    args = ap.parse_args()

    # FP8 protocol runs first and its activation cache is reused for W4
    # (identical model, identical calibration batches -> identical inputs).
    rows_fp8, acts = run_one_protocol(
        "fp8", args.config_fp8, None, collect_acts=True,
        max_samples=args.max_samples,
    )
    rows_w4, _ = run_one_protocol(
        "w4", args.config_w4, acts, collect_acts=False,
        max_samples=args.max_samples,
    )
    write_csvs(rows_fp8 + rows_w4, args.out_dir)


if __name__ == "__main__":
    main()
