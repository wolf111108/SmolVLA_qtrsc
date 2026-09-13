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

MatMul variants (attention QK^T / PV) take (layer, A, B, stat_collector)
and are registered with a "matmul_" prefix, paired with the matmul_*
entries in scale_methods:
  - "matmul_per_tensor" / "matmul_outlier" / "matmul_pot_fp8_outlier"
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
import math

from vla_tcs2.quant.utils import Round, LINEAR_SHIFT_NUM, MATMUL_SHIFT_NUM, log_layer_sqnr
from vla_tcs2.quant.scale_methods import (
    get_outlier_mask_channel,
    get_outlier_mask_1d,
    _matmul_B_channel_mask,
)
from vla_tcs2.quant.quant_spec import (
    quant_awo,
    fp8_dtype,
    fp8_max,
)


def _module_label(layer) -> str:
    """Return the physical operator identity for logging (module_id first)."""
    return getattr(layer, "module_id", "") or getattr(layer, "layer_name", "")


def _collect_linear_runtime(
    layer,
    stat_collector,
    *,
    input_code=None,
    input_spec=None,
    output_code=None,
    output_spec=None,
    operand_origin=None,
):
    """Structured runtime collection for a Linear site (manual §30-§31).

    Collects input AND output codes with module_id + runtime context.
    No-op when the collector does not expose collect_quant_tensor.
    """
    if stat_collector is None or not hasattr(
        stat_collector, "collect_quant_tensor"
    ):
        return

    module_id = getattr(layer, "module_id", "") or layer.layer_name
    metadata = {"operand_origin": operand_origin} if operand_origin else None

    if input_code is not None and input_spec is not None:
        stat_collector.collect_quant_tensor(
            module_id=module_id,
            tensor_role="activation",
            tensor_code=input_code,
            spec=input_spec,
            metadata=metadata,
        )
    if output_code is not None and output_spec is not None:
        stat_collector.collect_quant_tensor(
            module_id=module_id,
            tensor_role="output",
            tensor_code=output_code,
            spec=output_spec,
            metadata=metadata,
        )


def _collect_outlier_partition(
    layer,
    stat_collector,
    *,
    tensor_role: str,
    total_elements: int,
    protected_elements: int,
    spec,
):
    """Record the outlier FP side-path partition (Phase H, manual §3).

    No-op when the collector does not expose collect_outlier_partition.
    """
    if stat_collector is None or not hasattr(
        stat_collector, "collect_outlier_partition"
    ):
        return

    module_id = getattr(layer, "module_id", "") or layer.layer_name
    stat_collector.collect_outlier_partition(
        module_id=module_id,
        tensor_role=tensor_role,
        total_elements=total_elements,
        protected_elements=protected_elements,
        spec=spec,
        layer_idx=getattr(layer, "layer_idx", None),
    )


def _collect_matmul_runtime(
    layer,
    stat_collector,
    *,
    A_code=None,
    A_spec=None,
    B_code=None,
    B_spec=None,
    O_code=None,
    O_spec=None,
    A_origin=None,
    B_origin=None,
):
    """Structured runtime collection for a MatMul site (manual §33).

    A / B / O are collected as separate roles with module_id + runtime
    context (attention_kind comes from the runtime context).
    """
    if stat_collector is None or not hasattr(
        stat_collector, "collect_quant_tensor"
    ):
        return

    module_id = getattr(layer, "module_id", "") or layer.layer_name

    if A_code is not None and A_spec is not None:
        stat_collector.collect_quant_tensor(
            module_id=module_id,
            tensor_role="A",
            tensor_code=A_code,
            spec=A_spec,
            metadata={"operand_origin": A_origin} if A_origin else None,
        )
    if B_code is not None and B_spec is not None:
        stat_collector.collect_quant_tensor(
            module_id=module_id,
            tensor_role="B",
            tensor_code=B_code,
            spec=B_spec,
            metadata={"operand_origin": B_origin} if B_origin else None,
        )
    if O_code is not None and O_spec is not None:
        stat_collector.collect_quant_tensor(
            module_id=module_id,
            tensor_role="O",
            tensor_code=O_code,
            spec=O_spec,
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
        _module_label(layer),
        layer.layer_idx,
        kind="activation",
        extra={"a_bit": layer.a_bit, "w_bit": layer.w_bit, "o_bit": layer.o_bit},
    )
    log_layer_sqnr(
        layer.weight,
        w_code.mul(layer.w_interval),
        _module_label(layer),
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
        _collect_linear_runtime(
            layer,
            stat_collector,
            input_code=x_code,
            input_spec=layer.a_spec,
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
        _collect_linear_runtime(
            layer,
            stat_collector,
            output_code=out_code,
            output_spec=layer.o_spec,
        )
        log_layer_sqnr(
            ref,
            out,
            _module_label(layer),
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
        _collect_linear_runtime(
            layer,
            stat_collector,
            output_code=out_code,
            output_spec=layer.o_spec,
        )
        log_layer_sqnr(
            ref,
            out,
            _module_label(layer),
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

    # Phase H (manual §3.4): record the FP side-path partition BEFORE the
    # masks are deleted so the native quant-path sparsity can exclude the
    # positions this forward artificially zeroes in the normal path.
    if stat_collector is not None and hasattr(
        stat_collector, "collect_outlier_partition"
    ):
        # Activation: channel mask is [1, 1, K]; protected elements repeat
        # across every token row.
        protected_channels = int(x_channel_mask.sum().item())
        repeat = x.numel() // max(x.shape[-1], 1)
        _collect_outlier_partition(
            layer,
            stat_collector,
            tensor_role="activation",
            total_elements=x.numel(),
            protected_elements=protected_channels * repeat,
            spec=layer.a_spec,
        )
        # Weight: w_channel_mask is the broadcast union of the activation
        # channel mask and the weight's own element-level outlier mask, so
        # its sum is the true protected element count of the weight tensor.
        _collect_outlier_partition(
            layer,
            stat_collector,
            tensor_role="weight_runtime_mask",
            total_elements=layer.weight.numel(),
            protected_elements=int(w_channel_mask.sum().item()),
            spec=layer.w_spec,
        )

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
        _collect_linear_runtime(
            layer,
            stat_collector,
            input_code=x_sim,
            input_spec=layer.a_spec,
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

    # Phase H (manual §3.4): output FP side-path partition (role="output"
    # so the native correction matches the structured output record).
    if stat_collector is not None and hasattr(
        stat_collector, "collect_outlier_partition"
    ):
        out_protected_channels = int(out_with_outlier_mask.sum().item())
        out_repeat = out_with_outlier.numel() // max(
            out_with_outlier.shape[-1], 1
        )
        _collect_outlier_partition(
            layer,
            stat_collector,
            tensor_role="output",
            total_elements=out_with_outlier.numel(),
            protected_elements=out_protected_channels * out_repeat,
            spec=layer.o_spec,
        )

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

    # Output collection: the normal-path quantized code (manual §25-§26;
    # outlier side-path fraction accounting is future work).
    _collect_linear_runtime(
        layer,
        stat_collector,
        output_code=out_normal_quant.to(torch.float32),
        output_spec=layer.o_spec,
    )

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


def quant_forward_pot_ao_outlier(
    layer,
    x,
    stat_collector=None,
) -> torch.Tensor:
    """
    PoT-FP8 activation/output + continuous weight scale + outlier forward.

    Companion forward for ``scales_with_pot_ao_outlier`` (mixed precision,
    e.g. INT4 weights + FP8 activations/output): only the activation and
    output scales are guaranteed to be exact powers of two — the weight
    scale keeps its calibrated continuous value and is NOT checked.
    """

    # Verify only once.
    if not getattr(
        layer,
        "_pot_ao_scales_verified",
        False,
    ):
        for name, interval in (
            ("activation", layer.a_interval),
            ("output", layer.o_interval),
        ):
            if not _is_power_of_two_scalar(
                interval,
            ):
                raise ValueError(
                    f"{layer.layer_name}_{layer.layer_idx}: "
                    f"{name} scale {interval!r} is not "
                    "power-of-two. Recalibrate into a "
                    "fresh scale_dir before using "
                    "method=pot_ao_outlier."
                )

        layer._pot_ao_scales_verified = True

    return quant_forward_with_outlier(
        layer,
        x,
        stat_collector,
    )


def quant_forward_pot_fp8_per_tensor(
    layer,
    x,
    stat_collector=None,
) -> torch.Tensor:
    """
    PoT-FP8 per-tensor forward (NO outlier protection).

    Companion forward for ``scales_with_pot_fp8_per_tensor``: every scale
    must be an exact power of two (verified once per layer), then the
    datapath delegates to quant_forward_per_tensor.
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
                interval,
            ):
                raise ValueError(
                    f"{layer.layer_name}_{layer.layer_idx}: "
                    f"{name} scale {interval!r} is not "
                    "power-of-two. Recalibrate into a "
                    "fresh scale_dir before using "
                    "method=pot_fp8_per_tensor."
                )

        layer._pot_scales_verified = True

    return quant_forward_per_tensor(
        layer,
        x,
        stat_collector,
    )


def _w_scale_is_per_channel(layer) -> bool:
    """True when layer.w_interval is a per-output-channel [N_out] tensor."""
    return (
        isinstance(layer.w_interval, torch.Tensor)
        and layer.w_interval.dim() == 1
    )


def quant_forward_pot_ao_outlier_channel(
    layer,
    x,
    stat_collector=None,
) -> torch.Tensor:
    """
    Per-output-channel W4 forward (G2-B), companion to
    ``scales_with_pot_ao_outlier_channel``.

    Same outlier decomposition as ``quant_forward_with_outlier``
    (Y = Y_QaQw + Y_FaFw + Y_FaQw + Y_QaFw), but the weight scale is a
    [N_out] tensor applied EXPLICITLY on the output feature dimension:

        Y_QaQw = s_a * diag(s_w) @ (q(x) @ q(W)^T)   → per output channel j:
                 s_a * s_w[j] * Σ_k q(a_k) q(w_jk)
        Y_FaQw = diag(s_w) @ (x_fp @ q(W)^T)
        Y_QaFw / Y_FaFw: FP weight path, unaffected by s_w granularity.

    The scalar-M integer compensation trick (M_aw / M_fa_qb in
    quant_forward_with_outlier) does not extend to per-channel s_w, so this
    path dequantizes explicitly: w_sim = q(w_normal) * s_w.view(-1, 1).
    """
    ratio = layer.outlier_ratio

    # a/o scales must be scalar PoT (same contract as pot_ao_outlier).
    if not getattr(layer, "_pot_ao_ch_scales_verified", False):
        for name, interval in (
            ("activation", layer.a_interval),
            ("output", layer.o_interval),
        ):
            if not _is_power_of_two_scalar(interval):
                raise ValueError(
                    f"{layer.layer_name}_{layer.layer_idx}: "
                    f"{name} scale {interval!r} is not power-of-two. "
                    "Recalibrate into a fresh scale_dir before using "
                    "method=pot_ao_outlier_channel."
                )
        if not _w_scale_is_per_channel(layer):
            raise ValueError(
                f"{layer.layer_name}_{layer.layer_idx}: w_interval is not "
                "a per-output-channel [N_out] tensor. Use method="
                "pot_ao_outlier_channel with scales calibrated by the "
                "matching scale method."
            )
        if not torch.isfinite(layer.w_interval).all():
            raise ValueError(
                f"{layer.layer_name}_{layer.layer_idx}: per-channel w "
                "scale contains non-finite values."
            )
        layer._pot_ao_ch_scales_verified = True

    w_scale = layer.w_interval.view(-1, 1)                # [N_out, 1]

    channel_mask = get_outlier_mask_channel(x, ratio)
    x_channel_mask = channel_mask.view(1, 1, -1)
    w_channel_mask = channel_mask.view(1, -1)
    w_outlier_mask = get_outlier_mask_1d(layer.weight, ratio)
    w_channel_mask = w_channel_mask | w_outlier_mask

    x_fp = x * x_channel_mask.to(torch.float32)
    x_normal_fp = (x * (~x_channel_mask).to(dtype=x.dtype)).to(torch.float32)
    w_fp = layer.weight * w_channel_mask.to(torch.float32)
    w_normal_fp = (
        layer.weight * (~w_channel_mask).to(dtype=layer.weight.dtype)
    ).to(torch.float32)

    # normal-part weight quantization with per-row scale (dequantized form)
    w_code = quant_awo(
        w_normal_fp,
        w_scale,
        layer.w_spec,
        out_dtype=torch.float32,
        chunk_size=1_048_576,
    )
    w_sim = w_code * w_scale                              # [N_out, K]

    # normal-part activation quantization (scalar PoT scale).
    # quant_awo returns the scaled CODE (x / a_interval); dequantize by
    # multiplying a_interval back — the four-path sum below operates on
    # dequantized magnitudes (unlike the scalar path's integer-compensation
    # M_aw trick, which does the rescaling at the end).
    x_code = quant_awo(
        x_normal_fp,
        layer.a_interval,
        layer.a_spec,
        out_dtype=torch.float32,
        chunk_size=1_048_576,
    )
    x_sim = x_code * layer.a_interval

    in_features = layer.weight.size(1)
    out_features = layer.weight.size(0)
    if stat_collector is not None:
        stat_collector.collect_quant_activation(
            layer.layer_name,
            layer.layer_idx,
            x_code,
            x_sim.to(torch.float16),
            layer.a_spec,
            layer.digit_size,
            layer.parallelism,
            in_features,
            out_features,
        )
        _collect_linear_runtime(
            layer,
            stat_collector,
            input_code=x_code,
            input_spec=layer.a_spec,
        )
        # Phase H: record FP side-path partition (matching the scalar
        # outlier path) so native sparsity is not over-estimated here.
        if hasattr(stat_collector, "collect_outlier_partition"):
            protected_channels = int(x_channel_mask.sum().item())
            repeat = x.numel() // max(x.shape[-1], 1)
            _collect_outlier_partition(
                layer,
                stat_collector,
                tensor_role="activation",
                total_elements=x.numel(),
                protected_elements=protected_channels * repeat,
                spec=layer.a_spec,
            )
            _collect_outlier_partition(
                layer,
                stat_collector,
                tensor_role="weight_runtime_mask",
                total_elements=layer.weight.numel(),
                protected_elements=int(w_channel_mask.sum().item()),
                spec=layer.w_spec,
            )

    if layer.bias is not None:
        bias = layer.bias.to(torch.float32)
    else:
        bias = None

    x_sim_fp32 = x_sim.to(torch.float32)
    w_sim_fp32 = w_sim.to(torch.float32)

    # Four-path outlier decomposition (§8.4): per-channel s_w enters only
    # the quantized-weight paths, already embedded in w_sim.
    out_qa_qb = F.linear(x_sim_fp32, w_sim_fp32)          # q(a)·q(w)·s_w
    out_fa_fb = F.linear(x_fp, w_fp)                      # FP outliers
    out_fa_qb = F.linear(x_fp, w_sim_fp32)                # FP x · q(w)·s_w
    out_qa_fb = F.linear(x_sim_fp32, w_fp)                # q(x) · FP w

    out_with_outlier = out_qa_qb + out_fa_fb + out_fa_qb + out_qa_fb
    if bias is not None:
        out_with_outlier = out_with_outlier + bias

    # output re-quantization: same as quant_forward_with_outlier
    M_q = layer.round(torch.tensor(layer.o_interval) * (2 ** 16))
    out_outlier_mask = get_outlier_mask_channel(out_with_outlier, ratio)
    # Phase H: output FP side-path partition.
    if stat_collector is not None and hasattr(
        stat_collector, "collect_outlier_partition"
    ):
        o_protected_channels = int(out_outlier_mask.sum().item())
        o_repeat = out_with_outlier.numel() // max(
            out_with_outlier.shape[-1], 1
        )
        _collect_outlier_partition(
            layer,
            stat_collector,
            tensor_role="output",
            total_elements=out_with_outlier.numel(),
            protected_elements=o_protected_channels * o_repeat,
            spec=layer.o_spec,
        )
    out_normal = out_with_outlier * (~out_outlier_mask).to(torch.float32)
    out_outlier = out_with_outlier * out_outlier_mask.to(torch.float32)

    out_normal_quant = quant_awo(
        out_normal,
        layer.o_interval,
        layer.o_spec,
        out_dtype=out_normal.dtype,
        chunk_size=1_048_576,
    )
    out_normal_dequant = out_normal_quant.to(torch.float32).mul_(M_q)
    out_normal_dequant = torch.div(out_normal_dequant, 2 ** 16).to(x.dtype)
    out_outlier = out_outlier.to(x.dtype)

    _collect_linear_runtime(
        layer,
        stat_collector,
        output_code=out_normal_quant.to(torch.float32),
        output_spec=layer.o_spec,
    )

    return out_normal_dequant + out_outlier

# ============================================================================
# MatMul quantized-forward methods (attention QK^T / PV)
# Signature: fn(layer, A, B, stat_collector=None) -> Tensor
# ============================================================================


def matmul_quant_forward_per_tensor(layer, A, B, stat_collector=None) -> torch.Tensor:
    """Matmul quantized forward with a single global per-tensor scale."""
    A_sim = quant_awo(
        A,
        layer.A_interval,
        layer.A_spec,
        out_dtype=torch.float32,
        chunk_size=1_048_576,
    )

    B_sim = quant_awo(
        B,
        layer.B_interval,
        layer.B_spec,
        out_dtype=torch.float32,
        chunk_size=1_048_576,
    )

    in_features = A.size(-1)
    out_features = B.size(-1)

    if stat_collector is not None:
        stat_collector.collect_quant_activation(
            f"{layer.layer_name}",
            layer.layer_idx,
            A_sim,
            A,
            B_sim,
            layer.B_spec,
            layer.A_spec,
            layer.digit_size,
            layer.parallelism,
            in_features,
            out_features,
        )
        _collect_matmul_runtime(
            layer,
            stat_collector,
            A_code=A_sim,
            A_spec=layer.A_spec,
            B_code=B_sim,
            B_spec=layer.B_spec,
        )

    acc_code = torch.matmul(A_sim, B_sim)

    scale_to_output = layer.A_interval * layer.B_interval / layer.O_interval

    if layer.O_spec.kind == "int":
        M0 = torch.tensor(
            scale_to_output,
            device=A.device,
            dtype=torch.float32,
        )
        M0 = layer.round(M0 * MATMUL_SHIFT_NUM)

        out_code = acc_code.mul(M0)
        out_code = torch.div(
            out_code,
            MATMUL_SHIFT_NUM,
            rounding_mode="floor",
        )

        out = out_code.mul(layer.O_interval).to(A.dtype)
        _collect_matmul_runtime(
            layer, stat_collector, O_code=out_code, O_spec=layer.O_spec,
        )

    elif layer.O_spec.kind == "fp":
        out_scaled = acc_code.mul(scale_to_output)

        dtype = fp8_dtype(layer.O_spec.fmt)
        max_val = fp8_max(layer.O_spec.fmt)

        out_code = out_scaled.clamp(-max_val, max_val).to(dtype).float()
        out = out_code.mul(layer.O_interval).to(A.dtype)
        _collect_matmul_runtime(
            layer, stat_collector, O_code=out_code, O_spec=layer.O_spec,
        )

    elif layer.O_spec.kind == "bf":
        M0 = torch.tensor(
            scale_to_output,
            device=A.device,
            dtype=torch.float32,
        )
        M0 = layer.round(M0 * MATMUL_SHIFT_NUM)

        out_code = acc_code.mul(M0)
        out_code = torch.div(
            out_code,
            MATMUL_SHIFT_NUM,
            rounding_mode="floor",
        )

        out = out_code.mul(layer.O_interval).to(A.dtype)

    else:
        out = torch.matmul(A, B)

    log_layer_sqnr(
        torch.matmul(A, B),
        out,
        _module_label(layer),
        layer.layer_idx,
        extra={
            "A_bit": layer.A_bit,
            "B_bit": layer.B_bit,
            "O_bit": layer.O_bit,
        },
    )

    return out


def matmul_quant_forward_with_outlier(layer, A, B, stat_collector=None) -> torch.Tensor:
    """Matmul quantized forward with outlier channels/elements kept in FP."""
    ratio = layer.outlier_ratio

    channel_mask = get_outlier_mask_channel(A, ratio)

    A_channel_mask = channel_mask.view(*([1] * (A.dim() - 1)), -1)
    B_channel_mask = _matmul_B_channel_mask(B, channel_mask, A)
    del channel_mask
    B_channel_mask = B_channel_mask | get_outlier_mask_1d(B, ratio)

    A_fp = A * A_channel_mask.to(torch.float32)        # A outliers, keep FP
    A_normal_fp = A * (~A_channel_mask).to(dtype=A.dtype)
    A_normal_fp = A_normal_fp.to(torch.float32)

    B_fp = B * B_channel_mask.to(torch.float32)        # B outliers, keep FP
    B_normal_fp = B * (~B_channel_mask).to(dtype=B.dtype)
    B_normal_fp = B_normal_fp.to(torch.float32)

    # Phase H (manual §3): record MatMul A/B FP side-path partitions.
    if stat_collector is not None and hasattr(
        stat_collector, "collect_outlier_partition"
    ):
        a_protected_channels = int(A_channel_mask.sum().item())
        a_repeat = A.numel() // max(A.shape[-1], 1)
        _collect_outlier_partition(
            layer,
            stat_collector,
            tensor_role="A",
            total_elements=A.numel(),
            protected_elements=a_protected_channels * a_repeat,
            spec=layer.A_spec,
        )
        _collect_outlier_partition(
            layer,
            stat_collector,
            tensor_role="B",
            total_elements=B.numel(),
            protected_elements=int(B_channel_mask.sum().item()),
            spec=layer.B_spec,
        )

    del A_channel_mask, B_channel_mask

    M_q = torch.tensor(layer.O_interval)
    M_q = layer.round(M_q * (2 ** 16))

    M_aw = torch.tensor(layer.A_interval * layer.B_interval)
    M_aw = layer.round(M_aw * 2 ** 48)

    M_fa_qb = torch.tensor(layer.B_interval)
    M_fa_qb = layer.round(M_fa_qb * 2 ** 24)

    M_qa_fb = torch.tensor(layer.A_interval)
    M_qa_fb = layer.round(M_qa_fb * 2 ** 24)

    A_sim = quant_awo(
        A_normal_fp,
        layer.A_interval,
        layer.A_spec,
        out_dtype=A.dtype,
        chunk_size=1_048_576,
    )

    B_sim = quant_awo(
        B_normal_fp,
        layer.B_interval,
        layer.B_spec,
        out_dtype=B.dtype,
        chunk_size=1_048_576,
    )

    in_features = A.size(-1)
    out_features = B.size(-1)
    if stat_collector is not None:
        stat_collector.collect_quant_activation(
            layer.layer_name,
            layer.layer_idx,
            A_sim.to(torch.float16),
            A_sim.to(torch.float16),
            B_sim,
            layer.B_spec,
            layer.A_spec,
            layer.digit_size,
            layer.parallelism,
            in_features,
            out_features,
        )
        _collect_matmul_runtime(
            layer,
            stat_collector,
            A_code=A_sim.to(torch.float32),
            A_spec=layer.A_spec,
            B_code=B_sim.to(torch.float32),
            B_spec=layer.B_spec,
        )

    A_sim_fp32 = A_sim.to(torch.float32)
    B_sim_fp32 = B_sim.to(torch.float32)

    out_qa_qb = torch.matmul(A_sim_fp32, B_sim_fp32)
    out_fa_fb = torch.matmul(A_fp, B_fp)
    out_fa_qb = torch.matmul(A_fp, B_sim_fp32)
    out_qa_fb = torch.matmul(A_sim_fp32, B_fp)

    out_qa_qb = out_qa_qb.mul_(M_aw)
    out_qa_qb = torch.div(out_qa_qb, 2 ** 48)

    out_fa_qb = out_fa_qb.mul_(M_fa_qb)
    out_fa_qb = torch.div(out_fa_qb, 2 ** 24)

    out_qa_fb = out_qa_fb.mul_(M_qa_fb)
    out_qa_fb = torch.div(out_qa_fb, 2 ** 24)

    # outliermore=True in the original: include both cross terms.
    out_with_outlier = out_qa_qb + out_fa_fb + out_fa_qb + out_qa_fb

    out_with_outlier_mask = get_outlier_mask_channel(out_with_outlier, ratio)
    out_without_outlier_mask = ~out_with_outlier_mask

    # Phase H (manual §3): MatMul O FP side-path partition.
    if stat_collector is not None and hasattr(
        stat_collector, "collect_outlier_partition"
    ):
        o_protected_channels = int(out_with_outlier_mask.sum().item())
        o_repeat = out_with_outlier.numel() // max(
            out_with_outlier.shape[-1], 1
        )
        _collect_outlier_partition(
            layer,
            stat_collector,
            tensor_role="O",
            total_elements=out_with_outlier.numel(),
            protected_elements=o_protected_channels * o_repeat,
            spec=layer.O_spec,
        )

    out_outlier = out_with_outlier * out_with_outlier_mask.to(torch.float32)
    out_normal = out_with_outlier * out_without_outlier_mask.to(torch.float32)

    out_normal_quant = quant_awo(
        out_normal,
        layer.O_interval,
        layer.O_spec,
        out_dtype=out_normal.dtype,
        chunk_size=1_048_576,
    )

    out_normal_dequant = out_normal_quant.to(torch.float32).mul_(M_q).to(A.dtype)
    out_normal_dequant = torch.div(out_normal_dequant, 2 ** 16).to(A.dtype)
    out_outlier = out_outlier.to(A.dtype)

    out = out_normal_dequant + out_outlier

    _collect_matmul_runtime(
        layer,
        stat_collector,
        O_code=out_normal_quant.to(torch.float32),
        O_spec=layer.O_spec,
    )

    log_layer_sqnr(
        torch.matmul(A, B),
        out,
        _module_label(layer),
        layer.layer_idx,
        extra={
            "A_bit": layer.A_bit,
            "B_bit": layer.B_bit,
            "O_bit": layer.O_bit,
        },
    )

    return out


def matmul_quant_forward_pot_fp8_outlier(
    layer,
    A,
    B,
    stat_collector=None,
) -> torch.Tensor:
    """
    PoT-FP8 + outlier matmul forward: identical datapath to
    matmul_quant_forward_with_outlier, but every scale must be an exact
    power of two (verified once per layer, like the linear variant).
    """
    if not getattr(layer, "_pot_scales_verified", False):
        for name, interval in (
            ("activation", layer.A_interval),
            ("weight", layer.B_interval),
            ("output", layer.O_interval),
        ):
            if not _is_power_of_two_scalar(interval):
                raise ValueError(
                    f"{layer.layer_name}_{layer.layer_idx}: "
                    f"{name} scale {interval!r} is not power-of-two. "
                    f"Recalibrate into a fresh scale_dir before using "
                    f"method=pot_fp8_outlier."
                )
        layer._pot_scales_verified = True

    return matmul_quant_forward_with_outlier(layer, A, B, stat_collector)


def matmul_quant_forward_pot_fp8_per_tensor(
    layer,
    A,
    B,
    stat_collector=None,
) -> torch.Tensor:
    """
    PoT-FP8 per-tensor matmul forward (NO outlier protection): identical
    datapath to matmul_quant_forward_per_tensor, but every scale must be
    an exact power of two (verified once per layer).
    """
    if not getattr(layer, "_pot_scales_verified", False):
        for name, interval in (
            ("activation", layer.A_interval),
            ("weight", layer.B_interval),
            ("output", layer.O_interval),
        ):
            if not _is_power_of_two_scalar(interval):
                raise ValueError(
                    f"{layer.layer_name}_{layer.layer_idx}: "
                    f"{name} scale {interval!r} is not power-of-two. "
                    f"Recalibrate into a fresh scale_dir before using "
                    f"method=pot_fp8_per_tensor."
                )
        layer._pot_scales_verified = True

    return matmul_quant_forward_per_tensor(layer, A, B, stat_collector)

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
    "pot_ao_outlier":
        quant_forward_pot_ao_outlier,
    "pot_ao_outlier_channel":
        quant_forward_pot_ao_outlier_channel,
    "pot_fp8_per_tensor":
        quant_forward_pot_fp8_per_tensor,
    # matmul variants (attention QK^T / PV)
    "matmul_per_tensor": matmul_quant_forward_per_tensor,
    "matmul_outlier": matmul_quant_forward_with_outlier,
    "matmul_pot_fp8_outlier": matmul_quant_forward_pot_fp8_outlier,
    "matmul_pot_fp8_per_tensor": matmul_quant_forward_pot_fp8_per_tensor,
}

def get_quant_method(name: str) -> Callable:
    """Look up a quantized-forward method by name (paired with scale_methods)."""
    method = QUANT_METHODS.get(name)
    if method is None:
        raise ValueError(
            f"Unknown quant method '{name}', valid: {sorted(QUANT_METHODS)}"
        )
    return method


def get_matmul_quant_method(name: str) -> Callable:
    """
    Look up a matmul quantized-forward method by its *linear-style* name
    (e.g. "pot_fp8_outlier" -> "matmul_pot_fp8_outlier").
    """
    method = QUANT_METHODS.get(f"matmul_{name}")
    if method is None:
        raise ValueError(
            f"Unknown matmul quant method '{name}', valid: "
            f"{sorted(k for k in QUANT_METHODS if k.startswith('matmul_'))}"
        )
    return method
