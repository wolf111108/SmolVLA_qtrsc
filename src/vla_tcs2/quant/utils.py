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
from typing import Dict, Any


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
