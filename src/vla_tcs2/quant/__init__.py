"""
VLA-TCS2 Quantization Framework.

Ported from opt-qt/quant (verified working), adapted for the
src-layout package vla_tcs2.
"""

from vla_tcs2.quant.quant_spec import (
    QuantSpec,
    parse_quant_spec,
    safe_scale_from_tensor,
    safe_scale_per_token,
    quant_awo,
    fp8_dtype,
    fp8_max,
)
from vla_tcs2.quant.utils import (
    Round,
    Floor,
    LINEAR_SHIFT_NUM,
    MATMUL_SHIFT_NUM,
    load_config,
    save_scales,
    load_scales,
    compute_sqnr,
)
from vla_tcs2.quant.stat_manager import (
    QuantStatManager,
    QuantStatistics,
)

__version__ = "0.1.0"

__all__ = [
    "QuantSpec",
    "parse_quant_spec",
    "safe_scale_from_tensor",
    "safe_scale_per_token",
    "quant_awo",
    "fp8_dtype",
    "fp8_max",
    "Round",
    "Floor",
    "LINEAR_SHIFT_NUM",
    "MATMUL_SHIFT_NUM",
    "load_config",
    "save_scales",
    "load_scales",
    "compute_sqnr",
    "QuantStatManager",
    "QuantStatistics",
]
