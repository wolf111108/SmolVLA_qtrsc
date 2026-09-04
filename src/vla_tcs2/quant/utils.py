"""
Utility functions for quantization.

Ported from opt-qt/quant/utils.py (verified working).
Only the parts needed by the VLA-TCS2 quantization path are kept:
- Round / Floor STE functions
- YAML config loading
- pickle scale save/load helpers
- fixed-point shift constants
"""

import os
import pickle

import yaml
import torch
from torch.autograd import Function
from typing import Dict, Any, Optional


# ============================================================================
# STE (Straight-Through Estimator) Functions
# ============================================================================

class RoundSTE(Function):
    """Straight-through estimator for rounding operation."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clone()


class FloorSTE(Function):
    """Straight-through estimator for floor operation."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        return torch.floor(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clone()


Round = RoundSTE.apply
Floor = FloorSTE.apply


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


# ============================================================================
# Scale Management (pickle, same naming scheme as opt-qt)
# ============================================================================

def save_scales(scales: Dict[str, float], scale_dir: str, layer_name: str):
    """
    Save quantization scales to pickle files.

    File naming: {layer_name}_{scale_type}.p  (matches QuantizedLinear._load_scales)
    """
    os.makedirs(scale_dir, exist_ok=True)

    for scale_type, scale_value in scales.items():
        filename = f"{layer_name}_{scale_type}.p"
        filepath = os.path.join(scale_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(scale_value, f)


def load_scales(scale_dir: str, layer_name: str, scale_types: list) -> Dict[str, float]:
    """
    Load quantization scales from pickle files.
    """
    scales = {}

    for scale_type in scale_types:
        filename = f"{layer_name}_{scale_type}.p"
        filepath = os.path.join(scale_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scale file not found: {filepath}")

        with open(filepath, 'rb') as f:
            scales[scale_type] = pickle.load(f)

    return scales


# ============================================================================
# Quantization Constants
# ============================================================================

LINEAR_SHIFT_NUM = 2 ** 48
MATMUL_SHIFT_NUM = 2 ** 20

DEFAULT_DIGIT_SIZE = 4
DEFAULT_PARALLELISM = 4


# ============================================================================
# Signal-to-Quantization-Noise Ratio (SQNR)
# ============================================================================

def compute_sqnr(reference: torch.Tensor, quantized: torch.Tensor) -> float:
    """
    Compute SQNR (Signal-to-Quantization-Noise Ratio) in dB between a
    reference (FP) tensor and a quantized/dequantized tensor.

        SQNR = 10 * log10( signal_power / noise_power )

    where signal_power is the mean square of the reference and noise_power
    is the mean square error between reference and quantized.

    Args:
        reference: the ground-truth (full-precision) tensor.
        quantized: the quantized (or dequantized) tensor to compare.

    Returns:
        SQNR in dB (float). Higher is better; +inf if the two tensors are
        identical.
    """
    ref = reference.detach().float()
    qnt = quantized.detach().float()

    if ref.shape != qnt.shape:
        raise ValueError(
            f"Shape mismatch: reference {tuple(ref.shape)} vs "
            f"quantized {tuple(qnt.shape)}"
        )

    signal_power = (ref ** 2).mean()
    noise_power = ((ref - qnt) ** 2).mean()

    if noise_power == 0:
        return float("inf")

    sqnr = 10.0 * torch.log10(signal_power / noise_power)
    return sqnr.item()


# Hard-coded output path for per-layer SQNR logging.
SQNR_LOG_PATH = "/home/zyzhao/VLA_tcs2/outputs/quant_sqnr/sqnrs_12mixed_new.jsonl"

# SQNR logging is opt-in (env var): the linear paths log per-layer SQNR on
# every forward, which is useful for offline analysis but slows eval down.
# Keep the default OFF so eval runs are not penalized unless requested.
import os as _os
SQNR_LOG_ENABLED = _os.environ.get("VLA_SQNR_LOG", "0") == "1"


def log_layer_sqnr(
    reference: torch.Tensor,
    quantized: torch.Tensor,
    layer_name: str = "",
    layer_idx: int = 0,
    kind: str = "output",
    extra: Optional[Dict] = None,
) -> float:
    """
    Compute SQNR between a reference (FP) tensor and a quantized tensor, then
    append a JSON record (one per line) to SQNR_LOG_PATH.

    Args:
        reference: full-precision reference tensor.
        quantized: quantized/dequantized tensor.
        layer_name: layer identifier (e.g. 'q_proj').
        layer_idx: layer index.
        kind: which tensor is being compared ('activation' / 'weight' / 'output').
        extra: optional dict of additional fields (e.g. a_bit/w_bit/o_bit).
            A 'module_id' key is included automatically by callers that pass
            it via layer_name (see quant_methods); keep layer_name as-is.

    Returns:
        SQNR in dB (float). Also writes the record to disk (when enabled).
    """
    if not SQNR_LOG_ENABLED:
        return float("nan")

    import json as _json

    sqnr_db = compute_sqnr(reference, quantized)

    entry = {
        "layer_name": layer_name,
        "layer_idx": layer_idx,
        "kind": kind,
        "sqnr_db": sqnr_db,
    }
    if extra:
        entry.update(extra)

    os.makedirs(os.path.dirname(SQNR_LOG_PATH), exist_ok=True)
    with open(SQNR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")

    return sqnr_db
