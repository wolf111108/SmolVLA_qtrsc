"""
Quantized Linear Layer Implementation.

This module provides a quantized linear layer that supports:
- Multiple quantization modes (raw, scale_inspection, quant_forward)
- Configurable bit widths for activations, weights, and outputs
- Scale calibration and quantized inference
"""

import os
import sys
import math
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional, Tuple, Dict

from vla_tcs2.quant.utils import Round
from vla_tcs2.quant.scale_methods import get_scale_method
from vla_tcs2.quant.quant_methods import get_quant_method
from vla_tcs2.quant.test_methods import get_test_method, test_method_requires_scales
from vla_tcs2.quant.quant_spec import (
    QuantSpec,
    parse_quant_spec,
)


# ============================================================================
# Tensor distribution dump (calibration-time, for offline outlier analysis)
# ============================================================================
#
# During scale_inspection we can dump the raw activation / weight / output
# tensors to disk so a separate script can aggregate and plot their value
# distributions (to choose an outlier_ratio).
#
# Storage is controlled by environment variables:
#   VLA_TENSOR_DUMP_DIR   output directory (default outputs/tensor_dump)
#   VLA_TENSOR_DUMP_STEPS number of leading calibration steps to dump (default 3)
#
# Weight is static: dumped only once per layer. Activation/output are dumped
# for each of the first VLA_TENSOR_DUMP_STEPS calibration steps.

DUMP_TENSORS_DIR = os.environ.get(
    "VLA_TENSOR_DUMP_DIR",
    "/home/zyzhao/VLA_tcs2/outputs/tensor_dump",
)
DUMP_MAX_STEPS = int(os.environ.get("VLA_TENSOR_DUMP_STEPS", "3"))

_dump_step = 0  # current calibration step (advanced by calibration.py)


def set_dump_step(step: int) -> None:
    """Advance the global calibration step counter (called by calibration.py)."""
    global _dump_step
    _dump_step = step


def get_dump_step() -> int:
    return _dump_step


class QuantizedLinear(nn.Linear):
    """
    Quantized linear layer supporting multiple quantization modes.
    
    Modes:
        - 'raw': No quantization, standard linear layer
        - 'scale_inspection': Collect scale statistics for calibration
        - 'quant_forward': Quantized forward pass using pre-calibrated scales
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "raw",
        a_bit: int = 8,
        w_bit: int = 8,
        o_bit: int = 8,
        d_bit: Optional[int] = None,
        p: Optional[int] = None,
        scale_root_str: str = "",
        outlier_ratio: float = 0.0,   # 新增，默认0表示不做outlier处理
        **kwargs
    ):
        """
        Initialize quantized linear layer.
        
        Args:
            in_features: Input feature size
            out_features: Output feature size
            bias: Whether to use bias
            mode: Quantization mode ('raw', 'scale_inspection', 'quant_forward')
            a_bit: Activation quantization bits
            w_bit: Weight quantization bits
            o_bit: Output quantization bits
            d_bit: Digit size for quantization
            p: Parallelism parameter
            scale_root_str: Root directory for scale files
        """
        super().__init__(in_features, out_features, bias)
        
        self.a_spec: QuantSpec = parse_quant_spec(a_bit)
        self.w_spec: QuantSpec = parse_quant_spec(w_bit)
        self.o_spec: QuantSpec = parse_quant_spec(o_bit)

        # Quantization parameters
        self.mode = mode
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.o_bit = o_bit
        self.digit_size = d_bit
        self.parallelism = p
        
        self.bitnet_weight_scale = None
        self.bitnet_online_quant = False
        self.is_bitnet = False


        # Quantization intervals (scales)
        self.w_interval: Optional[float] = None
        self.a_interval: Optional[float] = None
        self.o_interval: Optional[float] = None

        if self.a_spec.kind == "int":
            self.a_qmax = 2 ** (self.a_bit - 1)
        else:
            self.a_qmax = None
        if self.w_spec.kind == "int":
            self.w_qmax = 2 ** (self.w_bit - 1)
        else:
            self.w_qmax = None
        if self.o_spec.kind == "int":
            self.o_qmax = 2 ** (self.o_bit - 1)
        else:
            self.o_qmax = None
        
        # Scale root directory
        self.scale_root_str = scale_root_str
        
        # Round function
        self.round = Round
        
        # Layer identification
        self.layer_name = ""
        self.layer_idx = 0

        # Physical operator identity (unique; used for sensitivity / dump /
        # SQNR / test stats). NEVER used for scale sharing.
        self.module_id = ""

        # Scale-sharing identity (configurable; used for calibration
        # aggregation and scale persistence). Falls back to layer_name/idx
        # when unset, preserving legacy behaviour.
        self.scale_group_name = ""
        self.scale_group_idx = 0
        
        self.outlier_ratio = outlier_ratio
        self.calibration_policy = "recalibrate"
        self._calibration_action_cache = None

        # Pluggable quantization method name. Drives BOTH the scale-inspection
        # (quant/scale_methods.py) and quantized-forward (quant/quant_methods.py)
        # registries — the two are paired by the same key. Resolved at wrap time
        # by create_quantized_linear from config (quantization.method or
        # per-layer method).
        self.method = "per_tensor"

        # Pluggable test-forward method name (quant/test_methods.py). Resolved
        # at wrap time from config (quantization.test_method or per-layer
        # test_method); dispatched by self.test_forward (mode "test_forward").
        self.test_method = "raw"

        # Tensor dump state: weight is static, so dump it only once per layer.
        self._weight_dumped = False
    
    def set_layer_info(self, layer_name: str, layer_idx: int, module_id: Optional[str] = None):
        """Set layer name/index and the physical operator identity.

        `module_id` uniquely identifies the real computation site (e.g.
        "vlm.layers.3.self_attn.q_proj") and is used only for sensitivity /
        dump / SQNR / test stats — never for scale sharing.
        """
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.module_id = (
            module_id if module_id is not None else f"{layer_name}_{layer_idx}"
        )

    def set_scale_group(self, name: str, idx: int):
        """Set the scale-sharing identity (calibration aggregation + files)."""
        self.scale_group_name = name
        self.scale_group_idx = idx

    def _scale_identity(self):
        """Return (name, idx) used for scale files / calibration aggregation."""
        if self.scale_group_name:
            return self.scale_group_name, self.scale_group_idx
        return self.layer_name, self.layer_idx
    
    
    def _scale_file_paths(self):
        name, idx = self._scale_identity()
        return (
            os.path.join(self.scale_root_str, f"{name}_w_scale_{idx}.p"),
            os.path.join(self.scale_root_str, f"{name}_a_scale_{idx}.p"),
            os.path.join(self.scale_root_str, f"{name}_o_scale_{idx}.p"),
        )

    def _scale_files_exist(self) -> bool:
        return all(os.path.exists(p) for p in self._scale_file_paths())

    def _resolve_calibration_action(self) -> str:
        if self._calibration_action_cache is not None:
            return self._calibration_action_cache

        p = str(self.calibration_policy).lower()
        if p == "auto":
            action = "reuse" if self._scale_files_exist() else "recalibrate"
        elif p in {"reuse", "recalibrate"}:
            action = p
        else:
            raise ValueError(
                f"Invalid calibration_policy={self.calibration_policy} "
                f"for {self.layer_name}_{self.layer_idx}"
            )

        self._calibration_action_cache = action
        return action

    def forward(
        self, 
        x: torch.Tensor,
        collect_stats: bool = False,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor
            collect_stats: Whether to collect statistics
            stat_collector: Statistics collector object
            
        Returns:
            Output tensor (quantized or not depending on mode)
        """


        # Use stored stat_manager if not provided
        if stat_collector is None and hasattr(self, '_stat_manager'):
            stat_collector = self._stat_manager
        
        if self.mode == 'raw':
            return F.linear(x, self.weight, self.bias)
        elif self.mode == "scale_inspection":
            return self.scale_inspection(x, stat_collector)
        elif self.mode == "quant_forward":
            return self.quant_forward(x, stat_collector)
        elif self.mode == "test_forward":
            return self.test_forward(x, stat_collector)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
    
    def quant_bias(self, b: torch.Tensor) -> torch.Tensor:
        # Quantize bias tensor.
        biasfp32 = b.to(torch.float32)
        bias_sim = self.round(
            biasfp32 / (self.a_interval * self.w_interval) 
        )
        return bias_sim

    def scale_inspection(
        self, 
        x: torch.Tensor,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Inspect and collect scale statistics for calibration.
        
        w_scale and a_scale are from FP values.
        o_scale is from quantized output (to match quant_forward behavior).
        But we return FP output for numerical stability.
        """
        out = F.linear(x, self.weight, self.bias)
        action = self._resolve_calibration_action()
        if action == "reuse":
            return out

        # Delegate to the pluggable scale method (quant/scale_methods.py).
        # Signature: fn(x, weight, out, a_spec, w_spec, o_spec, **kwargs)
        # Returns: (a_interval, w_interval, o_interval)
        scale_fn = get_scale_method(self.method)

        self.a_interval, self.w_interval, self.o_interval = scale_fn(
            x,
            self.weight,
            out,
            self.a_spec,
            self.w_spec,
            self.o_spec,
            outlier_ratio=self.outlier_ratio,
        )

        # Collect statistics if collector is provided
        if stat_collector is not None:
            scale_name, scale_idx = self._scale_identity()
            stat_collector.collect_linear_stats(
                scale_name,
                scale_idx,
                self.w_interval,
                self.a_interval,
                self.o_interval
            )

        # Dump raw tensors for offline distribution analysis (outlier ratio
        # selection). Weight is dumped once; activation/output per leading step.
        self._maybe_dump_tensors(x, self.weight, out)
        
        # Return FP output for numerical stability during calibration
        return out


    def _maybe_dump_tensors(self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
        """
        Dump raw activation / weight / output tensors to DUMP_TENSORS_DIR for
        offline distribution analysis.

        - Weight is static: dumped once per layer (guarded by self._weight_dumped).
        - Activation and output are dumped for each of the first DUMP_MAX_STEPS
          calibration steps (guarded by the global _dump_step counter).

        File naming:
            weight_{module_id}.pt
            activation_{module_id}_step{step}.pt
            output_{module_id}_step{step}.pt

        Tensors are detached and moved to CPU before saving (torch.save).
        """
        if not DUMP_TENSORS_DIR:
            return

        os.makedirs(DUMP_TENSORS_DIR, exist_ok=True)

        # Sanitize module_id for use as a filename fragment (dots -> underscores).
        dump_id = self.module_id.replace(".", "_")

        # Weight: once per layer.
        if not self._weight_dumped:
            w_path = os.path.join(
                DUMP_TENSORS_DIR,
                f"weight_{dump_id}.pt",
            )
            torch.save(weight.detach().cpu(), w_path)
            self._weight_dumped = True

        # Activation / output: only for the leading steps.
        step = get_dump_step()
        if step >= DUMP_MAX_STEPS:
            return

        x_path = os.path.join(
            DUMP_TENSORS_DIR,
            f"activation_{dump_id}_step{step}.pt",
        )
        o_path = os.path.join(
            DUMP_TENSORS_DIR,
            f"output_{dump_id}_step{step}.pt",
        )
        torch.save(x.detach().cpu(), x_path)
        torch.save(out.detach().cpu(), o_path)
    
    
    def quant_forward(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Quantized forward — delegates to the pluggable quant method
        (quant/quant_methods.py). This method only handles flow control:
        load scales, then dispatch on self.method.
        """
        self._load_scales()
        return get_quant_method(self.method)(self, x, stat_collector)

    def test_forward(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Test/experimental forward — delegates to the pluggable test method
        (quant/test_methods.py). Flow control mirrors quant_forward, but
        scales are loaded on demand: only quant_residual methods need them
        (gaussian/raw methods work without calibration).
        """
        if test_method_requires_scales(self.test_method):
            self._load_scales()
        return get_test_method(self.test_method)(self, x, stat_collector)

    def _load_scales(self):
        """Load quantization scales from files."""
        # Construct scale file paths
        w_scale_file, a_scale_file, o_scale_file = self._scale_file_paths()
        
        # Load scales
        with open(w_scale_file, 'rb') as f:
            self.w_interval = pickle.load(f)
        
        with open(a_scale_file, 'rb') as f:
            self.a_interval = pickle.load(f)
        
        with open(o_scale_file, 'rb') as f:
            self.o_interval = pickle.load(f)
    
    def save_scales(self):
        """Save current scales to files."""
        if self.w_interval is None or self.a_interval is None or self.o_interval is None:
            raise ValueError("Scales not computed yet. Run scale_inspection first.")
        
        os.makedirs(self.scale_root_str, exist_ok=True)
        
        # Save weight scale
        w_scale_file, a_scale_file, o_scale_file = self._scale_file_paths()
        with open(w_scale_file, 'wb') as f:
            pickle.dump(self.w_interval, f)
        
        # Save activation scale
        with open(a_scale_file, 'wb') as f:
            pickle.dump(self.a_interval, f)
        
        # Save output scale
        with open(o_scale_file, 'wb') as f:
            pickle.dump(self.o_interval, f)
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'mode={self.mode}, '
            f'a_bit={self.a_bit}, '
            f'w_bit={self.w_bit}, '
            f'o_bit={self.o_bit}'
        )


    def log_quant_error(self, value: float, layer_name: str = None, layer_idx: int = None):
        import json as _json
        import os as _os
        layer_name = layer_name if layer_name is not None else self.layer_name
        layer_idx = layer_idx if layer_idx is not None else self.layer_idx
        json_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "quant-error", "error.jsonl"
        )
        _os.makedirs(_os.path.dirname(json_path), exist_ok=True)
        entry = {"layer_name": layer_name, "layer_idx": layer_idx, "value": float(value)}
        with open(json_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
