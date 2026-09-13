# quant/quant_spec.py
# Ported verbatim from opt-qt/quant/quant_spec.py (verified working).

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class QuantSpec:
    """
    Unified quantization format descriptor.

    kind:
      - "int": symmetric signed integer quantization
      - "fp": FP8 fake quantization, e4m3/e5m2
      - "bf": bfloat16 pass-through (no quantization)
      - "none": no quantization
    """
    kind: str
    bits: Optional[int] = None
    fmt: Optional[str] = None
    enabled: bool = True

    def name(self) -> str:
        if self.kind == "int":
            return f"int{self.bits}"
        if self.kind == "fp":
            return self.fmt
        return "none"


def parse_quant_spec(value: Any, default: Any = 8) -> QuantSpec:
    """
    Parse config value such as:
      8, 4, "8", "4", "e4m3", "e5m2", "fp16", "none"
    """
    if value is None:
        value = default

    if isinstance(value, int):
        return QuantSpec(kind="int", bits=value)

    if isinstance(value, str):
        v = value.lower().strip()

        if v.isdigit():
            return QuantSpec(kind="int", bits=int(v))

        if v in {"e4m3", "e4m3fn", "fp8_e4m3", "fp8_e4m3fn"}:
            return QuantSpec(kind="fp", fmt="e4m3")

        if v in {"e5m2", "fp8_e5m2"}:
            return QuantSpec(kind="fp", fmt="e5m2")

        if v in {"e2m1", "fp4", "fp4_e2m1"}:
            return QuantSpec(kind="fp", fmt="e2m1")

        if v in {"fp16"}:
            return QuantSpec(kind="fp", fmt="e5m10", enabled=False)

        if v in {"bf16"}:
            return QuantSpec(kind="bf", fmt="bf16", enabled=False)

    raise ValueError(f"Unsupported quant spec: {value}")


def fp8_dtype(fmt: str):
    fmt = fmt.lower()
    if fmt in {"e4m3", "e4m3fn"}:
        return torch.float8_e4m3fn
    if fmt == "e5m2":
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FP8 format: {fmt}")


def fp8_max(fmt: str) -> float:
    return torch.finfo(fp8_dtype(fmt)).max


# ------------------------------------------------------------
# FP4 (E2M1) fake quantization support
# ------------------------------------------------------------

FP4_E2M1_MAX: float = 6.0
FP4_E2M1_VALUES: tuple = (
    0.0, 0.5, 1.0, 1.5,
    2.0, 3.0, 4.0, 6.0,
)


def fp4_e2m1_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """Quantize FP tensor to nearest E2M1-representable value."""
    sign = torch.sign(x)
    abs_x = x.abs()
    levels = torch.tensor(
        FP4_E2M1_VALUES,
        device=x.device,
        dtype=torch.float32,
    )
    abs_q = levels[
        torch.argmin((abs_x.unsqueeze(-1) - levels).abs(), dim=-1)
    ]
    return abs_q * sign


def fp_format_max(fmt: str) -> float:
    fmt = fmt.lower()
    if fmt == "e2m1":
        return FP4_E2M1_MAX
    return fp8_max(fmt)


def int_qmax(bits: int) -> int:
    return 2 ** (bits - 1)


def safe_scale_from_tensor(x: torch.Tensor, spec: QuantSpec) -> float:
    """
    Compute interval/scale for INT or FP8 from tensor absmax.
    """
    if spec.kind == "none" or not spec.enabled:
        return 1.0

    max_abs = x.detach().abs().max()

    if max_abs == 0 or torch.isnan(max_abs) or torch.isinf(max_abs):
        return 1.0

    if spec.kind == "int":
        qmax = int_qmax(spec.bits)
        return (max_abs / (qmax - 0.5)).item()

    if spec.kind == "fp":
        return (max_abs / fp8_max(spec.fmt)).item()

    if spec.kind == "bf":
        return 1.0

    raise ValueError(f"Unsupported spec: {spec}")


def safe_scale_per_token(
    x: torch.Tensor,
    spec: QuantSpec,
    dim: int = -1,
) -> torch.Tensor:
    """
    Per-token (per-row) dynamic scale.
    """
    if spec.kind == "none" or not spec.enabled:
        return torch.ones(1, device=x.device, dtype=torch.float32)

    max_abs = x.detach().abs().amax(dim=dim, keepdim=True)
    max_abs = torch.clamp(max_abs, min=1e-10)

    if spec.kind == "int":
        qmax = int_qmax(spec.bits)
        return max_abs / (qmax - 0.5)

    if spec.kind == "fp":
        return max_abs / fp_format_max(spec.fmt)

    if spec.kind == "bf":
        return torch.ones_like(max_abs)

    raise ValueError(f"Unsupported spec: {spec}")


def safe_scale_per_output_channel(
    weight: torch.Tensor,
    spec: QuantSpec,
) -> torch.Tensor:
    """
    Per-output-channel scale for a weight tensor [N_out, K].

    Mirrors safe_scale_from_tensor semantics (absmax / (qmax - 0.5) for
    INT, absmax / fp8_max for FP), but computed per output row so each
    row gets its own scale. Returns a [N_out] float32 tensor.
    """
    if spec.kind == "none" or not spec.enabled:
        return torch.ones(
            weight.size(0), device=weight.device, dtype=torch.float32
        )

    max_abs = weight.detach().abs().amax(dim=1)          # [N_out]
    max_abs = torch.clamp(max_abs, min=1e-10)

    if spec.kind == "int":
        qmax = int_qmax(spec.bits)
        return (max_abs / (qmax - 0.5)).to(torch.float32)

    if spec.kind == "fp":
        return (max_abs / fp8_max(spec.fmt)).to(torch.float32)

    if spec.kind == "bf":
        return torch.ones_like(max_abs)

    raise ValueError(f"Unsupported spec: {spec}")


def quant_awo(
    x: torch.Tensor,
    scale,
    spec: QuantSpec,
    out_dtype=None,
    chunk_size: int = 1_048_576,
) -> torch.Tensor:
    """
    Chunked fake quantization for INT / FP8 / FP4.
    """
    if out_dtype is None:
        out_dtype = x.dtype

    if spec.kind == "none" or not spec.enabled:
        return x

    if scale is None:
        raise ValueError(f"Scale is None for spec={spec}")

    if chunk_size is None or chunk_size <= 0:
        chunk_size = x.numel()

    scale_is_tensor = isinstance(scale, torch.Tensor) and scale.dim() > 0

    if scale_is_tensor:
        if spec.kind == "int":
            qmax = int_qmax(spec.bits)
            q = torch.div(x, scale)
            q = torch.round(q)
            q = torch.clamp(q, -qmax, qmax - 1)
            return q.to(out_dtype)

        if spec.kind == "fp":
            fmt = spec.fmt.lower() if spec.fmt else ""
            if fmt == "e2m1":
                q = torch.div(x, scale)
                q = torch.clamp(q, -FP4_E2M1_MAX, FP4_E2M1_MAX)
                q = fp4_e2m1_fake_quant(q)
                return q.to(out_dtype)

            dtype = fp8_dtype(spec.fmt)
            max_val = fp8_max(spec.fmt)
            q = torch.div(x, scale)
            q = torch.clamp(q, -max_val, max_val)
            return q.to(dtype)

        return x

    # ---- scalar scale: 1D chunked path ----
    flat = x.reshape(-1)
    out = torch.empty_like(flat, dtype=out_dtype)

    for start in range(0, flat.numel(), chunk_size):
        end = min(start + chunk_size, flat.numel())
        chunk = flat[start:end]

        if spec.kind == "int":
            qmax = int_qmax(spec.bits)
            q = torch.div(chunk, scale)
            q = torch.round(q)
            q = torch.clamp(q, -qmax, qmax - 1)
            out[start:end] = q.to(out_dtype)

        elif spec.kind == "fp":
            fmt = spec.fmt.lower() if spec.fmt else ""
            if fmt == "e2m1":
                q = torch.div(chunk, scale)
                q = torch.clamp(q, -FP4_E2M1_MAX, FP4_E2M1_MAX)
                q = fp4_e2m1_fake_quant(q)
                out[start:end] = q.to(out_dtype)
            else:
                dtype = fp8_dtype(spec.fmt)
                max_val = fp8_max(spec.fmt)
                q = torch.div(chunk, scale)
                q = torch.clamp(q, -max_val, max_val)
                out[start:end] = q.to(dtype)

        else:
            out[start:end] = chunk.to(out_dtype)

    return out.view(x.shape)
