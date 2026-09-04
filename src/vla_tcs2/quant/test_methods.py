# src/vla_tcs2/quant/test_methods.py
"""
Noise / sensitivity-test methods for QuantizedLinear (and optional MatMul).

Linear interface:
    fn(layer, x, stat_collector=None) -> Tensor

Supported experiment families:
- raw
- gaussian_rms[_input|_weight|_output][_op]
- quant_residual[_input|_weight|_output][_op]

Gaussian:
    T_test = T + alpha * RMS(T_normal) * N(0,1)

Quantization residual:
    E_q    = QDQ(T_normal) - T_normal
    T_test = T + lambda * E_q

For *_op variants, outliers remain FP and only the normal part is perturbed.

Expected per-layer attributes (set by model_wrapper):
    test_method
    test_noise_alpha = 0.03
    test_noise_seed = 0
    test_outlier_ratio = 0.01
    test_residual_lambda = 1.0
    test_site = "output"
    test_use_outlier_protection = False

For reproducible layer-wise experiments, also set a unique:
    module_id = "vlm.text_model.layers.3.self_attn.q_proj"

IMPORTANT:
Gaussian methods do not need calibration scales.
quant_residual methods do need a/w/o intervals.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from vla_tcs2.quant.quant_spec import quant_awo
from vla_tcs2.quant.scale_methods import (
    get_outlier_mask_channel,
    get_outlier_mask_1d,
    _matmul_B_channel_mask,
)


# ============================================================================
# Config helpers
# ============================================================================

def _get(layer, names, default):
    for name in names:
        if hasattr(layer, name):
            value = getattr(layer, name)
            if value is not None:
                return value
    return default


def _alpha(layer) -> float:
    x = float(_get(layer, ("test_noise_alpha", "noise_alpha", "alpha"), 0.01))
    if x < 0:
        raise ValueError(f"noise alpha must be >= 0, got {x}")
    return x


def _seed(layer) -> int:
    return int(_get(layer, ("test_noise_seed", "noise_seed"), 0))


def _lam(layer) -> float:
    return float(
        _get(
            layer,
            ("test_residual_lambda", "test_noise_lambda", "residual_lambda"),
            1.0,
        )
    )


def _ratio(layer) -> float:
    x = float(
        _get(
            layer,
            ("test_outlier_ratio", "noise_outlier_ratio", "outlier_ratio"),
            0.01,
        )
    )
    if not 0.0 <= x < 1.0:
        raise ValueError(f"outlier ratio must be in [0,1), got {x}")
    return x


def _site(layer, default="output") -> str:
    return str(_get(layer, ("test_site", "noise_site"), default))


def _use_op(layer) -> bool:
    return bool(
        _get(
            layer,
            ("test_use_outlier_protection", "test_outlier_protection"),
            False,
        )
    )


def _identity(layer) -> str:
    module_id = getattr(layer, "module_id", None)
    if module_id:
        return str(module_id)
    return f"{getattr(layer, 'layer_name', type(layer).__name__)}_" \
           f"{getattr(layer, 'layer_idx', 0)}"


# ============================================================================
# RNG / RMS helpers
# ============================================================================

def _stable_hash(s: str) -> int:
    d = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little") & 0x7FFFFFFF


def _generator(layer, tensor, stream: str) -> torch.Generator:
    """
    Stateful deterministic RNG:
    - same module/test seed => reproducible run
    - different modules/sites => independent stream
    - repeated forwards => new samples from the same stream
    """
    derived_seed = (_seed(layer) + _stable_hash(f"{_identity(layer)}|{stream}")) & 0x7FFFFFFF

    cache = getattr(layer, "_test_generators", None)
    if cache is None:
        cache = {}
        layer._test_generators = cache

    key = (str(tensor.device), stream, derived_seed)
    if key not in cache:
        g = torch.Generator(device=tensor.device)
        g.manual_seed(derived_seed)
        cache[key] = g
    return cache[key]


def reset_test_noise_state(layer) -> None:
    """Reset RNG/static-weight-noise state before a new noise-seed run."""
    for name in ("_test_generators", "_static_test_weight_z", "_last_test_stats"):
        if hasattr(layer, name):
            delattr(layer, name)


def _expand_last(mask_1d: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return mask_1d.view(*([1] * (tensor.dim() - 1)), -1).expand_as(tensor)


def _rms(t: torch.Tensor, normal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = t.detach().float()
    if normal_mask is None:
        return torch.sqrt((x * x).mean()) if x.numel() else x.new_zeros(())

    m = normal_mask.bool().expand_as(x)
    n = m.sum()
    if n.item() == 0:
        return x.new_zeros(())
    e = torch.where(m, x * x, torch.zeros_like(x)).sum()
    return torch.sqrt(e / n.float())


def _normal_channel_mask(t: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio <= 0:
        return torch.ones_like(t, dtype=torch.bool)
    outlier_ch = get_outlier_mask_channel(t, ratio)
    return ~_expand_last(outlier_ch, t)


def _linear_weight_normal_mask(layer, x, ratio: float) -> torch.Tensor:
    """
    Match current outlier quantization policy:
    activation-outlier columns OR element-level weight outliers stay FP.
    """
    if ratio <= 0:
        return torch.ones_like(layer.weight, dtype=torch.bool)

    ch = get_outlier_mask_channel(x, ratio)                    # [K]
    protected_cols = ch.view(1, -1).expand_as(layer.weight)
    protected_elems = get_outlier_mask_1d(layer.weight, ratio)
    return ~(protected_cols | protected_elems)


def _record(layer, method, site, ref, perturb, protected_fraction, strength_name, strength):
    ref_rms = float(_rms(ref).item())
    err_rms = float(_rms(perturb).item())
    snr = math.inf if err_rms == 0 else (
        -math.inf if ref_rms == 0 else 20.0 * math.log10(ref_rms / err_rms)
    )
    layer._last_test_stats = {
        "module_id": _identity(layer),
        "method": method,
        "site": site,
        strength_name: float(strength),
        "reference_rms": ref_rms,
        "perturbation_rms": err_rms,
        "snr_db": snr,
        "protected_fraction": float(protected_fraction),
    }


def _collect(layer, stat_collector):
    if stat_collector is None:
        return
    fn = getattr(stat_collector, "collect_test_noise", None)
    if callable(fn):
        fn(dict(layer._last_test_stats))


# ============================================================================
# Gaussian RMS noise
# ============================================================================

def _gaussian(layer, t, *, stream, normal_mask=None, static=False):
    alpha = _alpha(layer)

    if normal_mask is None:
        normal_mask = torch.ones_like(t, dtype=torch.bool)
    else:
        normal_mask = normal_mask.bool().expand_as(t)

    protected_fraction = 1.0 - float(normal_mask.float().mean().item())
    sigma = _rms(t, normal_mask) * alpha

    if alpha == 0 or sigma.item() == 0:
        perturb = torch.zeros_like(t)
        return t, perturb, protected_fraction

    if static:
        cache = getattr(layer, "_static_test_weight_z", None)
        key = (stream, str(t.device), str(t.dtype), tuple(t.shape), _seed(layer))
        if cache is None or cache[0] != key:
            g = _generator(layer, t, stream + ":static")
            z = torch.randn(t.shape, device=t.device, dtype=torch.float32, generator=g)
            layer._static_test_weight_z = (key, z.cpu())
        else:
            z = cache[1].to(t.device)
    else:
        g = _generator(layer, t, stream + ":dynamic")
        z = torch.randn(t.shape, device=t.device, dtype=torch.float32, generator=g)

    perturb = (z * sigma * normal_mask.float()).to(t.dtype)
    return t + perturb, perturb, protected_fraction


def _linear_gaussian(layer, x, site, use_op, stat_collector=None):
    ratio = _ratio(layer) if use_op else 0.0
    s = site.lower()

    if s in {"input", "activation", "a", "x"}:
        mask = _normal_channel_mask(x, ratio) if use_op else None
        x2, e, pf = _gaussian(layer, x, stream="input", normal_mask=mask)
        out = F.linear(x2, layer.weight, layer.bias)
        ref = x
        site_name = "input"

    elif s in {"weight", "w"}:
        mask = _linear_weight_normal_mask(layer, x, ratio) if use_op else None
        w2, e, pf = _gaussian(
            layer, layer.weight, stream="weight", normal_mask=mask, static=True
        )
        out = F.linear(x, w2, layer.bias)
        ref = layer.weight
        site_name = "weight"

    elif s in {"output", "out", "o", "operator_output"}:
        ref = F.linear(x, layer.weight, layer.bias)
        mask = _normal_channel_mask(ref, ratio) if use_op else None
        out, e, pf = _gaussian(layer, ref, stream="output", normal_mask=mask)
        site_name = "output"

    else:
        raise ValueError(f"Linear noise site must be input/weight/output, got {site!r}")

    _record(
        layer,
        "gaussian_rms_op" if use_op else "gaussian_rms",
        site_name,
        ref,
        e,
        pf,
        "alpha",
        _alpha(layer),
    )
    _collect(layer, stat_collector)
    return out


def test_raw(layer, x, stat_collector=None):
    return F.linear(x, layer.weight, layer.bias)


def test_gaussian_rms(layer, x, stat_collector=None):
    return _linear_gaussian(
        layer, x, _site(layer), _use_op(layer), stat_collector
    )


def test_gaussian_rms_input(layer, x, stat_collector=None):
    return _linear_gaussian(layer, x, "input", False, stat_collector)


def test_gaussian_rms_input_op(layer, x, stat_collector=None):
    return _linear_gaussian(layer, x, "input", True, stat_collector)


def test_gaussian_rms_weight(layer, x, stat_collector=None):
    return _linear_gaussian(layer, x, "weight", False, stat_collector)


def test_gaussian_rms_weight_op(layer, x, stat_collector=None):
    return _linear_gaussian(layer, x, "weight", True, stat_collector)


def test_gaussian_rms_output(layer, x, stat_collector=None):
    return _linear_gaussian(layer, x, "output", False, stat_collector)


def test_gaussian_rms_output_op(layer, x, stat_collector=None):
    return _linear_gaussian(layer, x, "output", True, stat_collector)


# ============================================================================
# Real quantization-residual injection
# ============================================================================

def _qdq(t: torch.Tensor, scale, spec) -> torch.Tensor:
    if spec is None or not getattr(spec, "enabled", True):
        return t
    code = quant_awo(
        t.float(),
        scale,
        spec,
        out_dtype=torch.float32,
        chunk_size=1_048_576,
    )
    return code.mul(scale).to(t.dtype)


def _linear_residual(layer, x, site, use_op, stat_collector=None):
    lam = _lam(layer)
    ratio = _ratio(layer) if use_op else 0.0
    s = site.lower()

    if s in {"input", "activation", "a", "x"}:
        normal = _normal_channel_mask(x, ratio) if use_op else torch.ones_like(x, dtype=torch.bool)
        base = x
        q = _qdq(base * normal.to(base.dtype), layer.a_interval, layer.a_spec)
        residual = (q - base * normal.to(base.dtype)) * normal.to(base.dtype)
        x2 = base + lam * residual
        out = F.linear(x2, layer.weight, layer.bias)
        ref = base
        site_name = "input"

    elif s in {"weight", "w"}:
        normal = (
            _linear_weight_normal_mask(layer, x, ratio)
            if use_op else torch.ones_like(layer.weight, dtype=torch.bool)
        )
        base = layer.weight
        q = _qdq(base * normal.to(base.dtype), layer.w_interval, layer.w_spec)
        residual = (q - base * normal.to(base.dtype)) * normal.to(base.dtype)
        w2 = base + lam * residual
        out = F.linear(x, w2, layer.bias)
        ref = base
        site_name = "weight"

    elif s in {"output", "out", "o", "operator_output"}:
        base = F.linear(x, layer.weight, layer.bias)
        normal = (
            _normal_channel_mask(base, ratio)
            if use_op else torch.ones_like(base, dtype=torch.bool)
        )
        q = _qdq(base * normal.to(base.dtype), layer.o_interval, layer.o_spec)
        residual = (q - base * normal.to(base.dtype)) * normal.to(base.dtype)
        out = base + lam * residual
        ref = base
        site_name = "output"

    else:
        raise ValueError(f"Linear residual site must be input/weight/output, got {site!r}")

    pf = 1.0 - float(normal.float().mean().item())
    _record(
        layer,
        "quant_residual_op" if use_op else "quant_residual",
        site_name,
        ref,
        lam * residual,
        pf,
        "lambda",
        lam,
    )
    _collect(layer, stat_collector)
    return out


def test_quant_residual(layer, x, stat_collector=None):
    return _linear_residual(
        layer, x, _site(layer), _use_op(layer), stat_collector
    )


def test_quant_residual_input(layer, x, stat_collector=None):
    return _linear_residual(layer, x, "input", False, stat_collector)


def test_quant_residual_input_op(layer, x, stat_collector=None):
    return _linear_residual(layer, x, "input", True, stat_collector)


def test_quant_residual_weight(layer, x, stat_collector=None):
    return _linear_residual(layer, x, "weight", False, stat_collector)


def test_quant_residual_weight_op(layer, x, stat_collector=None):
    return _linear_residual(layer, x, "weight", True, stat_collector)


def test_quant_residual_output(layer, x, stat_collector=None):
    return _linear_residual(layer, x, "output", False, stat_collector)


def test_quant_residual_output_op(layer, x, stat_collector=None):
    return _linear_residual(layer, x, "output", True, stat_collector)


# ============================================================================
# Optional MatMul support
# ============================================================================

def _matmul_A_normal(A, ratio):
    return _normal_channel_mask(A, ratio)


def _matmul_B_normal(A, B, ratio):
    if ratio <= 0:
        return torch.ones_like(B, dtype=torch.bool)
    ch = get_outlier_mask_channel(A, ratio)
    protected_ch = _matmul_B_channel_mask(B, ch, A).expand_as(B)
    protected_elem = get_outlier_mask_1d(B, ratio)
    return ~(protected_ch | protected_elem)


def _matmul_gaussian(layer, A, B, site, use_op, stat_collector=None):
    ratio = _ratio(layer) if use_op else 0.0
    s = site.lower()

    if s == "a":
        mask = _matmul_A_normal(A, ratio) if use_op else None
        A2, e, pf = _gaussian(layer, A, stream="matmul_A", normal_mask=mask)
        out = torch.matmul(A2, B)
        ref, site_name = A, "A"

    elif s == "b":
        mask = _matmul_B_normal(A, B, ratio) if use_op else None
        B2, e, pf = _gaussian(layer, B, stream="matmul_B", normal_mask=mask)
        out = torch.matmul(A, B2)
        ref, site_name = B, "B"

    elif s in {"output", "out", "o", "operator_output"}:
        ref = torch.matmul(A, B)
        mask = _normal_channel_mask(ref, ratio) if use_op else None
        out, e, pf = _gaussian(layer, ref, stream="matmul_output", normal_mask=mask)
        site_name = "output"

    else:
        raise ValueError(f"MatMul noise site must be A/B/output, got {site!r}")

    _record(
        layer,
        "gaussian_rms_op" if use_op else "gaussian_rms",
        f"matmul_{site_name}",
        ref,
        e,
        pf,
        "alpha",
        _alpha(layer),
    )
    _collect(layer, stat_collector)
    return out


def test_matmul_raw(layer, A, B, stat_collector=None):
    return torch.matmul(A, B)


def test_matmul_gaussian_rms(layer, A, B, stat_collector=None):
    return _matmul_gaussian(layer, A, B, _site(layer), _use_op(layer), stat_collector)


def test_matmul_gaussian_rms_A(layer, A, B, stat_collector=None):
    return _matmul_gaussian(layer, A, B, "A", False, stat_collector)


def test_matmul_gaussian_rms_A_op(layer, A, B, stat_collector=None):
    return _matmul_gaussian(layer, A, B, "A", True, stat_collector)


def test_matmul_gaussian_rms_B(layer, A, B, stat_collector=None):
    return _matmul_gaussian(layer, A, B, "B", False, stat_collector)


def test_matmul_gaussian_rms_B_op(layer, A, B, stat_collector=None):
    return _matmul_gaussian(layer, A, B, "B", True, stat_collector)


def test_matmul_gaussian_rms_output(layer, A, B, stat_collector=None):
    return _matmul_gaussian(layer, A, B, "output", False, stat_collector)


def test_matmul_gaussian_rms_output_op(layer, A, B, stat_collector=None):
    return _matmul_gaussian(layer, A, B, "output", True, stat_collector)


def _matmul_residual(layer, A, B, site, use_op, stat_collector=None):
    lam = _lam(layer)
    ratio = _ratio(layer) if use_op else 0.0
    s = site.lower()

    if s == "a":
        normal = _matmul_A_normal(A, ratio) if use_op else torch.ones_like(A, dtype=torch.bool)
        q = _qdq(A * normal.to(A.dtype), layer.A_interval, layer.A_spec)
        residual = (q - A * normal.to(A.dtype)) * normal.to(A.dtype)
        out = torch.matmul(A + lam * residual, B)
        ref, site_name = A, "A"

    elif s == "b":
        normal = _matmul_B_normal(A, B, ratio) if use_op else torch.ones_like(B, dtype=torch.bool)
        q = _qdq(B * normal.to(B.dtype), layer.B_interval, layer.B_spec)
        residual = (q - B * normal.to(B.dtype)) * normal.to(B.dtype)
        out = torch.matmul(A, B + lam * residual)
        ref, site_name = B, "B"

    elif s in {"output", "out", "o", "operator_output"}:
        ref = torch.matmul(A, B)
        normal = _normal_channel_mask(ref, ratio) if use_op else torch.ones_like(ref, dtype=torch.bool)
        q = _qdq(ref * normal.to(ref.dtype), layer.O_interval, layer.O_spec)
        residual = (q - ref * normal.to(ref.dtype)) * normal.to(ref.dtype)
        out = ref + lam * residual
        site_name = "output"

    else:
        raise ValueError(f"MatMul residual site must be A/B/output, got {site!r}")

    pf = 1.0 - float(normal.float().mean().item())
    _record(
        layer,
        "quant_residual_op" if use_op else "quant_residual",
        f"matmul_{site_name}",
        ref,
        lam * residual,
        pf,
        "lambda",
        lam,
    )
    _collect(layer, stat_collector)
    return out


def test_matmul_quant_residual(layer, A, B, stat_collector=None):
    return _matmul_residual(layer, A, B, _site(layer), _use_op(layer), stat_collector)


def test_matmul_quant_residual_A(layer, A, B, stat_collector=None):
    return _matmul_residual(layer, A, B, "A", False, stat_collector)


def test_matmul_quant_residual_A_op(layer, A, B, stat_collector=None):
    return _matmul_residual(layer, A, B, "A", True, stat_collector)


def test_matmul_quant_residual_B(layer, A, B, stat_collector=None):
    return _matmul_residual(layer, A, B, "B", False, stat_collector)


def test_matmul_quant_residual_B_op(layer, A, B, stat_collector=None):
    return _matmul_residual(layer, A, B, "B", True, stat_collector)


def test_matmul_quant_residual_output(layer, A, B, stat_collector=None):
    return _matmul_residual(layer, A, B, "output", False, stat_collector)


def test_matmul_quant_residual_output_op(layer, A, B, stat_collector=None):
    return _matmul_residual(layer, A, B, "output", True, stat_collector)


# ============================================================================
# Registry
# ============================================================================

TEST_METHODS: dict[str, Callable] = {
    "raw": test_raw,

    "gaussian_rms": test_gaussian_rms,
    "gaussian_rms_input": test_gaussian_rms_input,
    "gaussian_rms_input_op": test_gaussian_rms_input_op,
    "gaussian_rms_weight": test_gaussian_rms_weight,
    "gaussian_rms_weight_op": test_gaussian_rms_weight_op,
    "gaussian_rms_output": test_gaussian_rms_output,
    "gaussian_rms_output_op": test_gaussian_rms_output_op,

    "quant_residual": test_quant_residual,
    "quant_residual_input": test_quant_residual_input,
    "quant_residual_input_op": test_quant_residual_input_op,
    "quant_residual_weight": test_quant_residual_weight,
    "quant_residual_weight_op": test_quant_residual_weight_op,
    "quant_residual_output": test_quant_residual_output,
    "quant_residual_output_op": test_quant_residual_output_op,

    "matmul_raw": test_matmul_raw,
    "matmul_gaussian_rms": test_matmul_gaussian_rms,
    "matmul_gaussian_rms_A": test_matmul_gaussian_rms_A,
    "matmul_gaussian_rms_A_op": test_matmul_gaussian_rms_A_op,
    "matmul_gaussian_rms_B": test_matmul_gaussian_rms_B,
    "matmul_gaussian_rms_B_op": test_matmul_gaussian_rms_B_op,
    "matmul_gaussian_rms_output": test_matmul_gaussian_rms_output,
    "matmul_gaussian_rms_output_op": test_matmul_gaussian_rms_output_op,

    "matmul_quant_residual": test_matmul_quant_residual,
    "matmul_quant_residual_A": test_matmul_quant_residual_A,
    "matmul_quant_residual_A_op": test_matmul_quant_residual_A_op,
    "matmul_quant_residual_B": test_matmul_quant_residual_B,
    "matmul_quant_residual_B_op": test_matmul_quant_residual_B_op,
    "matmul_quant_residual_output": test_matmul_quant_residual_output,
    "matmul_quant_residual_output_op": test_matmul_quant_residual_output_op,
}


def get_test_method(name: str) -> Callable:
    """Linear test-method resolver."""
    fn = TEST_METHODS.get(name)
    if fn is None or name.startswith("matmul_"):
        valid = sorted(k for k in TEST_METHODS if not k.startswith("matmul_"))
        raise ValueError(f"Unknown Linear test method {name!r}; valid: {valid}")
    return fn


def get_matmul_test_method(name: str) -> Callable:
    """Map linear-style name to MatMul test-method name."""
    key = name if name.startswith("matmul_") else f"matmul_{name}"
    fn = TEST_METHODS.get(key)
    if fn is None:
        valid = sorted(k for k in TEST_METHODS if k.startswith("matmul_"))
        raise ValueError(f"Unknown MatMul test method {name!r}; valid: {valid}")
    return fn


def test_method_requires_scales(name: str) -> bool:
    """Only quant_residual methods require calibrated scale files."""
    return "quant_residual" in str(name)
