# quant/quant_methods.py
"""
Pluggable quantized-forward methods (evaluation side).

Mirrors quant/scale_methods.py on the calibration side: each method has a
uniform signature

    fn(layer, x, stat_collector=None) -> Tensor

where `layer` is a QuantizedLinear (already scale-loaded) and `x` is the
input activation. Methods read the layer's intervals / specs / weight /
bias and produce the quantized output.

Available methods (paired with scale_methods by name):
  - "per_tensor" : single global scale, no outlier protection
  - "outlier"    : outlier channels/elements kept FP, the rest quantized

A method name drives BOTH calibration (scale_methods) and evaluation
(quant_methods) so a config value like `method: outlier` is sufficient.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
import math

from vla_tcs2.quant.utils import Round, LINEAR_SHIFT_NUM, log_layer_sqnr
from vla_tcs2.quant.scale_methods import (
    get_outlier_mask_channel,
    get_outlier_mask_1d,
)
from vla_tcs2.quant.quant_spec import (
    quant_awo,
    fp8_dtype,
    fp8_max,
)


def quant_forward_per_tensor(layer, x, stat_collector=None) -> torch.Tensor:
    """Quantized forward with a single global per-tensor scale (no outliers)."""
    # Full-precision reference output, used for SQNR logging.
    ref = F.linear(x, layer.weight, layer.bias)

    M0 = torch.tensor(
        layer.w_interval * layer.a_interval / layer.o_interval,
        device=x.device,
        dtype=torch.float32,
    )
    M0 = layer.round(M0 * LINEAR_SHIFT_NUM)

    x_code = quant_awo(
        x,
        layer.a_interval,
        layer.a_spec,
        out_dtype=torch.float32,
        chunk_size=1_048_576,
    )

    w_code = quant_awo(
        layer.weight,
        layer.w_interval,
        layer.w_spec,
        out_dtype=torch.float32,
        chunk_size=1_048_576,
    )

    # Log SQNR for activation and weight quantization (dequant = code * scale).
    log_layer_sqnr(
        x,
        x_code.mul(layer.a_interval),
        layer.layer_name,
        layer.layer_idx,
        kind="activation",
        extra={"a_bit": layer.a_bit, "w_bit": layer.w_bit, "o_bit": layer.o_bit},
    )
    log_layer_sqnr(
        layer.weight,
        w_code.mul(layer.w_interval),
        layer.layer_name,
        layer.layer_idx,
        kind="weight",
        extra={"a_bit": layer.a_bit, "w_bit": layer.w_bit, "o_bit": layer.o_bit},
    )

    if layer.bias is not None:
        bias_sim = layer.quant_bias(layer.bias)
    else:
        bias_sim = None

    in_features = layer.weight.size(1)
    out_features = layer.weight.size(0)
    if stat_collector is not None:
        stat_collector.collect_quant_activation(
            layer.layer_name,
            layer.layer_idx,
            x_code,
            x_code,
            layer.a_spec,
            layer.digit_size,
            layer.parallelism,
            in_features,
            out_features,
        )

    if bias_sim is not None:
        bias_code = bias_sim.to(torch.float32)
    else:
        bias_code = None

    acc_code = F.linear(x_code, w_code, bias_code)

    scale_to_output = layer.a_interval * layer.w_interval / layer.o_interval

    if layer.o_spec.kind == "int":
        M0 = torch.tensor(scale_to_output, device=x.device, dtype=torch.float32)
        M0 = layer.round(M0 * LINEAR_SHIFT_NUM)

        out_code = acc_code.mul(M0)
        out_code = torch.div(out_code, LINEAR_SHIFT_NUM, rounding_mode="floor")

        out = out_code.mul(layer.o_interval).to(x.dtype)
        log_layer_sqnr(
            ref,
            out,
            layer.layer_name,
            layer.layer_idx,
            extra={"a_bit": layer.a_bit, "w_bit": layer.w_bit, "o_bit": layer.o_bit},
        )
        return out

    if layer.o_spec.kind == "fp" and layer.o_spec.enabled:
        out_scaled = acc_code.mul(scale_to_output)

        dtype = fp8_dtype(layer.o_spec.fmt)
        max_val = fp8_max(layer.o_spec.fmt)

        out_code = out_scaled.clamp(-max_val, max_val).to(dtype).float()
        out = out_code.mul(layer.o_interval).to(x.dtype)
        log_layer_sqnr(
            ref,
            out,
            layer.layer_name,
            layer.layer_idx,
            extra={"a_bit": layer.a_bit, "w_bit": layer.w_bit, "o_bit": layer.o_bit},
        )
        return out

    # output not quantized
    return F.linear(x, layer.weight, layer.bias)


def quant_forward_with_outlier(layer, x, stat_collector=None) -> torch.Tensor:
    """Quantized forward with outlier channels/elements kept in FP."""
    outliermore = True
    ratio = layer.outlier_ratio

    channel_mask = get_outlier_mask_channel(x, ratio)

    x_channel_mask = channel_mask.view(1, 1, -1)   # [1, 1, H]
    w_channel_mask = channel_mask.view(1, -1)      # [1, H]

    if outliermore:
        w_outlier_mask = get_outlier_mask_1d(layer.weight, ratio)
        w_channel_mask = w_channel_mask | w_outlier_mask
        del w_outlier_mask

    x_fp = x * x_channel_mask.to(torch.float32)        # outlier activations, FP
    x_normal_fp = x * (~x_channel_mask).to(dtype=x.dtype)
    x_normal_fp = x_normal_fp.to(torch.float32)

    w_fp = layer.weight * w_channel_mask.to(torch.float32)        # outlier weights, FP
    w_normal_fp = layer.weight * (~w_channel_mask).to(dtype=layer.weight.dtype)
    w_normal_fp = w_normal_fp.to(torch.float32)

    del w_channel_mask
    del x_channel_mask

    M_q = torch.tensor(layer.o_interval)
    M_q = layer.round(M_q * (2 ** 16))

    M_aw = torch.tensor(layer.a_interval * layer.w_interval)
    M_aw = layer.round(M_aw * 2 ** 48)

    M_fa_qb = torch.tensor(layer.w_interval)
    M_fa_qb = layer.round(M_fa_qb * 2 ** 24)

    M_qa_fb = torch.tensor(layer.a_interval)
    M_qa_fb = layer.round(M_qa_fb * 2 ** 24)

    # float32 (not bf16) so high-bit codes are not re-rounded by bf16 mantissa.
    x_sim = quant_awo(
        x_normal_fp,
        layer.a_interval,
        layer.a_spec,
        out_dtype=torch.float32,
        chunk_size=1_048_576,
    )
    w_sim = quant_awo(
        w_normal_fp,
        layer.w_interval,
        layer.w_spec,
        out_dtype=torch.float32,
        chunk_size=1_048_576,
    )

    in_features = layer.weight.size(1)
    out_features = layer.weight.size(0)
    if stat_collector is not None:
        stat_collector.collect_quant_activation(
            layer.layer_name,
            layer.layer_idx,
            x_sim,
            x_sim.to(torch.float16),
            layer.a_spec,
            layer.digit_size,
            layer.parallelism,
            in_features,
            out_features,
        )

    if layer.bias is not None:
        bias_sim = layer.quant_bias(layer.bias).to(torch.float32)
        bias = layer.bias.to(torch.float32)
    else:
        bias_sim = None
        bias = None

    x_sim_fp32 = x_sim.to(torch.float32)
    w_sim_fp32 = w_sim.to(torch.float32)

    out_qa_qb = F.linear(x_sim_fp32, w_sim_fp32, bias_sim)  # normal part, INT32 range
    out_fa_fb = F.linear(x_fp, w_fp)                         # outlier part, FP32
    out_fa_qb = F.linear(x_fp, w_sim_fp32)                   # x FP, w quantized
    out_qa_fb = F.linear(x_sim_fp32, w_fp)                   # x quantized, w FP

    out_qa_qb = out_qa_qb.mul_(M_aw)
    out_qa_qb = torch.div(out_qa_qb, 2 ** 48)

    out_fa_qb = out_fa_qb.mul_(M_fa_qb)
    out_fa_qb = torch.div(out_fa_qb, 2 ** 24)

    out_qa_fb = out_qa_fb.mul_(M_qa_fb)
    out_qa_fb = torch.div(out_qa_fb, 2 ** 24)

    if outliermore:
        out_with_outlier = out_qa_qb + out_fa_fb + out_fa_qb + out_qa_fb
    else:
        out_with_outlier = out_qa_qb + out_fa_fb

    out_with_outlier_mask = get_outlier_mask_channel(out_with_outlier, ratio)
    out_without_outlier_mask = ~out_with_outlier_mask

    out_outlier = out_with_outlier * out_with_outlier_mask.to(torch.float32)
    out_normal = out_with_outlier * out_without_outlier_mask.to(torch.float32)

    out_normal_quant = quant_awo(
        out_normal,
        layer.o_interval,
        layer.o_spec,
        out_dtype=out_normal.dtype,
        chunk_size=1_048_576,
    )

    out_normal_dequant = out_normal_quant.to(torch.float32).mul_(M_q).to(x.dtype)
    out_normal_dequant = torch.div(out_normal_dequant, 2 ** 16).to(x.dtype)
    out_outlier = out_outlier.to(x.dtype)

    out = out_normal_dequant + out_outlier

    if torch.isnan(out).max():
        pass

    return out

def quant_forward_pot_fp8_outlier(
    layer,
    x,
    stat_collector=None,
) -> torch.Tensor:
    """
    PoT-FP8 + outlier forward.

    The actual datapath intentionally reuses the existing
    outlier implementation. The only difference is that all
    calibrated scales must be exact powers of two.
    """

    # Verify only once.
    if not getattr(
        layer,
        "_pot_scales_verified",
        False,
    ):
        for name, interval in (
            ("activation", layer.a_interval),
            ("weight", layer.w_interval),
            ("output", layer.o_interval),
        ):
            if not _is_power_of_two_scalar(
                interval
            ):
                raise ValueError(
                    f"{layer.layer_name}_{layer.layer_idx}: "
                    f"{name} scale {interval!r} is not "
                    f"power-of-two. Recalibrate into a "
                    f"fresh scale_dir before using "
                    f"method=pot_fp8_outlier."
                )

        layer._pot_scales_verified = True

    return quant_forward_with_outlier(
        layer,
        x,
        stat_collector,
    )

# ============================================================================
# Tools
# ============================================================================

def _is_power_of_two_scalar(value) -> bool:
    if value is None:
        return False

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return False
        value = value.item()

    value = float(value)

    if (not math.isfinite(value)) or value <= 0.0:
        return False

    exponent = math.log2(value)

    return abs(
        exponent - round(exponent)
    ) < 1e-6

# ============================================================================
# Registry
# ============================================================================

QUANT_METHODS: dict[str, Callable] = {
    "per_tensor": quant_forward_per_tensor,
    "outlier": quant_forward_with_outlier,
    "pot_fp8_outlier":
        quant_forward_pot_fp8_outlier,
}

def get_quant_method(name: str) -> Callable:
    """Look up a quantized-forward method by name (paired with scale_methods)."""
    method = QUANT_METHODS.get(name)
    if method is None:
        raise ValueError(
            f"Unknown quant method '{name}', valid: {sorted(QUANT_METHODS)}"
        )
    return method
