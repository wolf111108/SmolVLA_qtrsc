# quant/scale_methods.py
"""
Pluggable scale-inspection (calibration) methods.

Mirrors quant_awo's role on the quantization side: calibration picks a
scale method by name (config: quantization.scale_method or per-layer
scale_method), evaluation picks the matching quantization path.

Each method has the signature:

    fn(x, weight, out, a_spec, w_spec, o_spec, **kwargs) -> (a, w, o)

where x/weight/out are the FP activation / weight / FP output of the
layer, *_spec are QuantSpec, and the return is a 3-tuple of intervals
(scales). kwargs carry method-specific parameters (e.g. outlier_ratio).

Available methods:
  - "per_tensor"      : global absmax scale (classic per-tensor symmetric)
  - "outlier"         : per-tensor over the *normal* part only; outlier
                        channels (ranked by absmax) excluded from absmax
  - "per_channel"     : (reserved) per-output-channel weight scale —
                        returns tensor scales, needs tensor-aware
                        quant_awo (already supported) — placeholder.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import math

import torch

from vla_tcs2.quant.quant_spec import (
    QuantSpec,
    safe_scale_from_tensor,
)


# ============================================================================
# Shared mask helpers (moved from QuantizedLinear so methods stay stateless)
# ============================================================================


def get_outlier_mask_channel(tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Channel-level outlier mask (True = outlier / keep FP).

    Compatible with [B, S, H] / [1, H] / [S, H] / [H]: channels are the
    last dimension; a channel's score is its absmax over all rows.
    """
    tensor_2d = tensor.reshape(-1, tensor.shape[-1])   # [N, H]
    channel_score = tensor_2d.abs().amax(dim=0)        # [H]

    k = max(1, int(channel_score.numel() * ratio))

    protected_idx = torch.topk(channel_score, k).indices

    channel_mask = torch.zeros_like(channel_score, dtype=torch.bool)
    channel_mask[protected_idx] = True

    return channel_mask


def get_outlier_mask_1d(tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Element-level outlier mask (True = outlier). Top-ratio fraction of
    elements by abs value. Returns all-False when all magnitudes equal.
    """
    if ratio <= 0.0:
        return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)

    numel = tensor.numel()
    if numel == 0:
        return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)

    k = max(1, min(int(numel * ratio), numel - 1))

    flat_abs = tensor.abs().flatten()
    threshold = torch.topk(flat_abs, k).values.min()

    min_val = flat_abs.min()
    if threshold == min_val:
        outlier_mask_flat = flat_abs > min_val
    else:
        outlier_mask_flat = flat_abs >= threshold

    if outlier_mask_flat.sum() == 0:
        return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)

    return outlier_mask_flat.view(tensor.shape)


# ============================================================================
# Scale methods
# ============================================================================


def scales_per_tensor(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    a_spec: QuantSpec,
    w_spec: QuantSpec,
    o_spec: QuantSpec,
    **kwargs,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Classic per-tensor symmetric scale from global absmax."""
    a = safe_scale_from_tensor(x, a_spec)
    w = safe_scale_from_tensor(weight, w_spec)
    o = safe_scale_from_tensor(out, o_spec)
    return a, w, o


def scales_with_outlier(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    a_spec: QuantSpec,
    w_spec: QuantSpec,
    o_spec: QuantSpec,
    outlier_ratio: float = 0.01,
    **kwargs,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Per-tensor scale over the *normal* part only (outlier channels/elements
    excluded from the absmax, they keep FP at quantization time).

    - x: outlier channels (absmax-ranked along hidden dim) excluded.
    - weight: same channel mask OR'ed with element-level outlier mask.
    - out: outlier channels excluded via index_select.
    """
    if outlier_ratio <= 0.0:
        return scales_per_tensor(x, weight, out, a_spec, w_spec, o_spec)

    channel_mask = get_outlier_mask_channel(x, outlier_ratio)

    x_channel_mask = channel_mask.view(1, 1, -1)   # [1, 1, H]
    w_channel_mask = channel_mask.view(1, -1)      # [1, H]

    # Element-level weight outliers additionally protected.
    w_outlier_mask = get_outlier_mask_1d(weight, outlier_ratio)
    w_channel_mask = w_channel_mask | w_outlier_mask
    del w_outlier_mask

    x_normal = (x * (~x_channel_mask).to(dtype=x.dtype)).to(torch.float32)
    w_normal = (weight * (~w_channel_mask).to(dtype=weight.dtype)).to(torch.float32)
    del w_channel_mask, x_channel_mask

    a = safe_scale_from_tensor(x_normal, a_spec)
    if a == 0:
        a = None
    w = safe_scale_from_tensor(w_normal, w_spec)

    del x_normal, w_normal

    o_channel_mask = get_outlier_mask_channel(out, outlier_ratio)
    normal_idx = torch.nonzero(~o_channel_mask, as_tuple=False).flatten()
    o_normal = out.index_select(dim=-1, index=normal_idx).to(torch.float32)
    o = safe_scale_from_tensor(o_normal, o_spec)

    return a, w, o

def scales_with_pot_fp8_outlier(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    a_spec: QuantSpec,
    w_spec: QuantSpec,
    o_spec: QuantSpec,
    outlier_ratio: float = 0.01,
    **kwargs,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """
    PoT-FP8 + Outlier Protection.

    First reuse exactly the same outlier filtering as the
    existing `outlier` method, then quantize each resulting
    scalar scale onto the power-of-two grid.

        arbitrary scale
              ↓
        2 ^ ceil(log2(scale))

    Therefore the only intended algorithmic difference versus
    the existing FP8+outlier path is the scale representation.
    """

    _validate_fp8_or_passthrough(
        a_spec,
        "activation",
    )
    _validate_fp8_or_passthrough(
        w_spec,
        "weight",
    )
    _validate_fp8_or_passthrough(
        o_spec,
        "output",
    )

    # Existing outlier calibration.
    a, w, o = scales_with_outlier(
        x,
        weight,
        out,
        a_spec,
        w_spec,
        o_spec,
        outlier_ratio=outlier_ratio,
        **kwargs,
    )

    # New part: force every scale to power-of-two.
    a_pot = _ceil_power_of_two_scale(a)
    w_pot = _ceil_power_of_two_scale(w)
    o_pot = _ceil_power_of_two_scale(o)

    return a_pot, w_pot, o_pot

# ============================================================================
# Tools
# ============================================================================

def _ceil_power_of_two_scale(
    scale: Optional[float],
) -> Optional[float]:
    """
    Convert a positive scalar scale to the smallest power-of-two
    scale >= original scale.

    Example:
        0.00513 -> 2^-7 = 0.0078125

    This preserves the no-overflow property of absmax calibration.
    """
    if scale is None:
        return None

    scale = float(scale)

    if (not math.isfinite(scale)) or scale <= 0.0:
        return 1.0

    exponent = math.ceil(math.log2(scale))

    # math.ldexp(1.0, exponent) == 2**exponent
    # and is exactly representable in binary floating point.
    return math.ldexp(1.0, exponent)

def _validate_fp8_or_passthrough(
    spec: QuantSpec,
    name: str,
) -> None:
    """
    pot_fp8_outlier is intended for FP8 tensors.

    Disabled fp16 / bf16 passthrough is also allowed,
    so later A8W8O16 experiments remain possible.
    """
    if not spec.enabled:
        return

    if (
        spec.kind != "fp"
        or (spec.fmt or "").lower()
        not in {"e4m3", "e4m3fn", "e5m2"}
    ):
        raise ValueError(
            f"pot_fp8_outlier requires FP8 or passthrough "
            f"for {name}, got {spec}"
        )

# ============================================================================
# Registry
# ============================================================================

SCALE_METHODS: dict[str, Callable] = {
    "per_tensor": scales_per_tensor,
    "outlier": scales_with_outlier,
    "pot_fp8_outlier": scales_with_pot_fp8_outlier,
}


def get_scale_method(name: str) -> Callable:
    """Look up a scale method by name (config: quantization.scale_method)."""
    method = SCALE_METHODS.get(name)
    if method is None:
        raise ValueError(
            f"Unknown scale_method '{name}', valid: {sorted(SCALE_METHODS)}"
        )
    return method
