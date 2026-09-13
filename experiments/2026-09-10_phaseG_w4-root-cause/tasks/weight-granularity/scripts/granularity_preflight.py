#!/usr/bin/env python
"""G2-P0 granularity preflight — offline numerical audit (no rollout).

For every VLM QuantizedLinear, compare weight quantization error under
per-tensor vs per-output-channel (or groupwise) W4 scale granularity.
Scales are computed offline from the layer weight itself (normal part
only, outlier-masked — same semantics as scales_with_outlier), so no
prior calibration run is required.

Outputs (into --out-dir):
  granularity_stats.csv    # per-layer metrics for the requested granularity
  operator_summary.csv
  layer_summary.csv

Usage:
  python granularity_preflight.py --config <g2 yaml> --out-dir <dir> \
      --granularity per_output_channel [--group-size 128]
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


def _sqnr_db(ref, test, eps=1e-12):
    sig = torch.sum(ref.float() ** 2).item()
    noi = torch.sum((ref.float() - test.float()) ** 2).item()
    return float("inf") if noi <= eps else 10.0 * math.log10((sig + eps) / (noi + eps))


def _nmse(ref, test, eps=1e-12):
    return torch.sum((ref.float() - test.float()) ** 2).item() / (torch.sum(ref.float() ** 2).item() + eps)


def _cosine(ref, test):
    return F.cosine_similarity(ref.float().flatten(), test.float().flatten(), dim=0).clamp(-1, 1).item()


@torch.no_grad()
def compute_scale(w_normal: torch.Tensor, granularity: str, group_size: int | None, qmax: int):
    """W4 scale over the normal (outlier-masked) weight.

    per_tensor        -> scalar
    per_output_channel-> [N_out]
    groupwise         -> [N_out, N_group] (along in_features K)
    """
    if granularity == "per_tensor":
        return (w_normal.abs().max() / qmax).item()
    if granularity == "per_output_channel":
        per_row = w_normal.abs().amax(dim=1) / qmax      # [N_out]
        per_row = per_row.clamp_min(1e-12)
        return per_row
    if granularity == "groupwise":
        n_out, k = w_normal.shape
        g = int(group_size)
        pad = (-k) % g
        wp = F.pad(w_normal, (0, pad))                    # [N_out, K_pad]
        wg = wp.view(n_out, -1, g)                        # [N_out, N_grp, g]
        sc = wg.abs().amax(dim=2) / qmax                  # [N_out, N_grp]
        sc = sc.clamp_min(1e-12)
        if pad:
            # masked-out padding region is zero → amax unaffected; keep shape
            pass
        return sc
    raise ValueError(granularity)


@torch.no_grad()
def quant_weight(w_normal, scale, spec, granularity, group_size):
    """Dequantized W4 weight under the given scale granularity."""
    if granularity == "per_tensor":
        code = quant_awo(w_normal, scale, spec, out_dtype=torch.float32)
        return code * scale
    if granularity == "per_output_channel":
        code = quant_awo(w_normal, scale.view(-1, 1), spec, out_dtype=torch.float32)
        return code * scale.view(-1, 1)
    if granularity == "groupwise":
        n_out, k = w_normal.shape
        g = int(group_size)
        pad = (-k) % g
        wp = F.pad(w_normal, (0, pad)).view(n_out, -1, g)
        sc = scale.unsqueeze(-1)                          # [N_out, N_grp, 1]
        code = quant_awo(wp, sc, spec, out_dtype=torch.float32)
        dq = code * sc
        return dq.view(n_out, -1)[:, :k]
    raise ValueError(granularity)


@torch.no_grad()
def audit_layer(m, granularity: str, group_size: int | None) -> dict:
    from vla_tcs2.quant.scale_methods import get_outlier_mask_1d

    ratio = getattr(m, "outlier_ratio", 0.01)
    w = m.weight.detach().float()
    n_bits = int(getattr(m.w_spec, "bits", 0))
    qmax = 2 ** (n_bits - 1)

    w_mask = get_outlier_mask_1d(m.weight, ratio)
    w_normal = w * (~w_mask).to(torch.float32)
    protected_ratio = w_mask.float().mean().item()

    scale = compute_scale(w_normal, granularity, group_size, qmax)
    w_sim = quant_weight(w_normal, scale, m.w_spec, granularity, group_size)

    if granularity == "per_tensor":
        scale_shape = "scalar"
        codes = (w_normal / scale).round()
        sat = ((w_normal.abs() / scale) > qmax + 0.5).float().mean().item()
    else:
        scale_shape = "x".join(map(str, scale.shape))
        if granularity == "per_output_channel":
            codes = (w_normal / scale.view(-1, 1)).round()
            sat = ((w_normal.abs() / scale.view(-1, 1)) > qmax + 0.5).float().mean().item()
        else:
            n_out, k = w_normal.shape
            g = int(group_size)
            pad = (-k) % g
            wp = F.pad(w_normal, (0, pad)).view(n_out, -1, g)
            codes = (wp / scale.unsqueeze(-1)).round().view(n_out, -1)[:, :k]
            sat = ((wp.abs() / scale.unsqueeze(-1)) > qmax + 0.5).float().mean().item()

    # activation from the SAME cached calibration store if available, else
    # use a deterministic probe to keep output metrics well-defined.
    x = getattr(m, "_g2_probe_x", None)
    if x is None:
        gen = torch.Generator(device="cpu").manual_seed(hash(m.module_id) % (2**31))
        x = torch.randn(2, 8, w.size(1), generator=gen).abs() * 0.1

    out_ref = F.linear(x.to(w.device), w_normal.to(w.device), None)
    out_sim = F.linear(x.to(w.device), w_sim.to(w.device), None)

    return {
        "module_name": m.module_id,
        "component": m.module_id.split(".")[0],
        "layer_idx": int(m.layer_idx),
        "op_type": m.layer_name,
        "weight_shape": "x".join(map(str, m.weight.shape)),
        "weight_scale_shape": scale_shape,
        "weight_nmse": _nmse(w_normal, w_sim),
        "weight_sqnr_db": _sqnr_db(w_normal, w_sim),
        "weight_cosine": _cosine(w_normal, w_sim),
        "quant_zero_ratio": (w_sim == 0).float().mean().item(),
        "saturation_ratio": sat,
        "unique_quant_codes": int(torch.unique(codes.to(torch.int64)).numel()),
        "protected_ratio": protected_ratio,
        "output_nmse": _nmse(out_ref, out_sim),
        "output_sqnr_db": _sqnr_db(out_ref, out_sim),
        "output_cosine": _cosine(out_ref, out_sim),
        "scale_isfinite": bool(torch.isfinite(torch.as_tensor(scale)).all().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--granularity", required=True,
                    choices=["per_tensor", "per_output_channel", "groupwise"])
    ap.add_argument("--group-size", type=int, default=None)
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    w = ModelWrapper(config)
    w.build(mode="scale_inspection")   # raw weights, no scale loading needed

    from vla_tcs2.quant_linear import QuantizedLinear

    rows = []
    for m in w.model.modules():
        if not isinstance(m, QuantizedLinear):
            continue
        mid = getattr(m, "module_id", "") or ""
        if not mid.startswith("vlm."):
            continue  # G2 只审计 VLM（G1 Gate 决策）
        rows.append(audit_layer(m, args.granularity, args.group_size))
        print(f"[{args.granularity}] {mid}: nmse={rows[-1]['weight_nmse']:.3e} "
              f"sqnr={rows[-1]['weight_sqnr_db']:.1f}dB sat={rows[-1]['saturation_ratio']:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (args.out_dir / "granularity_stats.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    def agg(keyfn, path):
        g = defaultdict(list)
        for r in rows:
            g[keyfn(r)].append(r)
        out = {k: {"count": len(v),
                   "weight_nmse_mean": sum(x["weight_nmse"] for x in v) / len(v),
                   "weight_sqnr_db_mean": sum(x["weight_sqnr_db"] for x in v) / len(v),
                   "output_nmse_mean": sum(x["output_nmse"] for x in v) / len(v),
                   "unique_codes_mean": sum(x["unique_quant_codes"] for x in v) / len(v)}
               for k, v in sorted(g.items())}
        with (args.out_dir / path).open("w", newline="") as f:
            keys = ["key", "count", "weight_nmse_mean", "weight_sqnr_db_mean",
                    "output_nmse_mean", "unique_codes_mean"]
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for k, v in out.items():
                wr.writerow({"key": k, **v})

    agg(lambda r: r["op_type"], "operator_summary.csv")
    agg(lambda r: (r["component"], r["layer_idx"]), "layer_summary.csv")

    n_bad_finite = sum(1 for r in rows if not r["scale_isfinite"])
    mean_nmse = sum(r["weight_nmse"] for r in rows) / len(rows)
    print(f"\n✓ {args.granularity}: n={len(rows)} layers, "
          f"mean_weight_nmse={mean_nmse:.3e}, nonfinite_scales={n_bad_finite}")
    print(f"✓ CSV -> {args.out_dir}")


if __name__ == "__main__":
    main()
