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
from vla_tcs2.quant.scale_methods import (
    SCALE_METHODS,
    get_scale_method,
    get_matmul_scale_method,
    scales_per_tensor,
    scales_with_outlier,
    matmul_scales_per_tensor,
    matmul_scales_with_outlier,
    get_outlier_mask_channel,
    get_outlier_mask_1d,
)
from vla_tcs2.quant.quant_methods import (
    QUANT_METHODS,
    get_quant_method,
    get_matmul_quant_method,
    quant_forward_per_tensor,
    quant_forward_with_outlier,
    matmul_quant_forward_per_tensor,
    matmul_quant_forward_with_outlier,
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
    "SCALE_METHODS",
    "get_scale_method",
    "get_matmul_scale_method",
    "scales_per_tensor",
    "scales_with_outlier",
    "matmul_scales_per_tensor",
    "matmul_scales_with_outlier",
    "get_outlier_mask_channel",
    "get_outlier_mask_1d",
    "QUANT_METHODS",
    "get_quant_method",
    "get_matmul_quant_method",
    "quant_forward_per_tensor",
    "quant_forward_with_outlier",
    "matmul_quant_forward_per_tensor",
    "matmul_quant_forward_with_outlier",
]
