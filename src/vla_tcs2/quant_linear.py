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

from vla_tcs2.quant.utils import Round, LINEAR_SHIFT_NUM
from vla_tcs2.quant.quant_spec import (
    QuantSpec,
    parse_quant_spec,
    safe_scale_from_tensor,
    safe_scale_per_token,
    quant_awo,
    fp8_dtype,
    fp8_max,
)

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
        
        self.outlier_ratio = outlier_ratio
        self.calibration_policy = "recalibrate"
        self._calibration_action_cache = None
    
    def set_layer_info(self, layer_name: str, layer_idx: int):
        """Set layer name and index for scale file management."""
        self.layer_name = layer_name
        self.layer_idx = layer_idx
    
    
    def _scale_file_paths(self):
        return (
            os.path.join(self.scale_root_str, f"{self.layer_name}_w_scale_{self.layer_idx}.p"),
            os.path.join(self.scale_root_str, f"{self.layer_name}_a_scale_{self.layer_idx}.p"),
            os.path.join(self.scale_root_str, f"{self.layer_name}_o_scale_{self.layer_idx}.p"),
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
        
        # Calculate weight and activation scales from FP values
        if self.outlier_ratio > 0.0:
            outliermore = True
            channel_mask = self.get_outlier_mask_channel(x, self.outlier_ratio)

            x_channel_mask = channel_mask.view(1, 1, -1)   # [1, 1, H]
            w_channel_mask = channel_mask.view(1, -1)      # [1, H]
            del channel_mask
            if outliermore:
                w_outlier_mask = self._get_outlier_mask_1d(self.weight, self.outlier_ratio)
                w_channel_mask = w_channel_mask | w_outlier_mask
                del w_outlier_mask
            else:
                pass
            x_normal_fp = x * (~x_channel_mask).to(dtype=x.dtype)
            x_normal_fp = x_normal_fp.to(torch.float32)

            w_normal_fp = self.weight * (~w_channel_mask).to(dtype=self.weight.dtype)
            w_normal_fp = w_normal_fp.to(torch.float32)

            del w_channel_mask
            del x_channel_mask

            self.a_interval = safe_scale_from_tensor(x_normal_fp, self.a_spec)
            if self.a_interval == 0:
                self.a_interval = None

            self.w_interval = safe_scale_from_tensor(w_normal_fp, self.w_spec)
            
            del x_normal_fp
            del w_normal_fp

            channel_mask = self.get_outlier_mask_channel(out, self.outlier_ratio)

            normal_idx = torch.nonzero(~channel_mask, as_tuple=False).flatten()
    
            o_normal_fp = out.index_select(dim=-1, index=normal_idx).to(torch.float32)

            self.o_interval = safe_scale_from_tensor(o_normal_fp, self.o_spec)
        else:
            self.a_interval = safe_scale_from_tensor(x, self.a_spec)
            self.w_interval = safe_scale_from_tensor(self.weight, self.w_spec)
            self.o_interval = safe_scale_from_tensor(out, self.o_spec)
        
            # Collect statistics if collector is provided
        if stat_collector is not None:
            stat_collector.collect_linear_stats(
                self.layer_name,
                self.layer_idx,
                self.w_interval,
                self.a_interval,
                self.o_interval
            )
        
        # Return FP output for numerical stability during calibration
        return out
    
    
    def quant_forward(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:

        self._load_scales()

        if self.outlier_ratio > 0.0:
            return self._quant_forward_with_outlier(x, stat_collector)

        M0 = torch.tensor(
            self.w_interval * self.a_interval / self.o_interval,
            device=x.device,
            dtype=torch.float32,
        )
        M0 = self.round(M0 * LINEAR_SHIFT_NUM)

        x_code = quant_awo(
            x,
            self.a_interval,
            self.a_spec,
            out_dtype=x.dtype,
            chunk_size=1_048_576,
        )

        w_code = quant_awo(
            self.weight,
            self.w_interval,
            self.w_spec,
            out_dtype=self.weight.dtype,
            chunk_size=1_048_576,
        )

        if self.bias is not None:
            bias_sim = self.quant_bias(self.bias)
        else:
            bias_sim = None

        x_code = x_code.to(torch.float32)
        w_code = w_code.to(torch.float32)


        in_features = self.weight.size(1)
        out_features = self.weight.size(0)
        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                x_code,
                x_code,
                self.a_spec,
                self.digit_size,
                self.parallelism,
                in_features,
                out_features
            )

        if bias_sim is not None:
            bias_code = bias_sim.to(torch.float32)
        else:
            bias_code = None

        acc_code = F.linear(x_code, w_code, bias_code)

        scale_to_output = self.a_interval * self.w_interval / self.o_interval

        if self.o_spec.kind == "int":
            M0 = torch.tensor(scale_to_output, device=x.device, dtype=torch.float32)
            M0 = self.round(M0 * LINEAR_SHIFT_NUM)

            out_code = acc_code.mul(M0)
            out_code = torch.div(
                out_code,
                LINEAR_SHIFT_NUM,
                rounding_mode="floor",
            )

            out = out_code.mul(self.o_interval).to(x.dtype)
            return out

        if self.o_spec.kind == "fp" and self.o_spec.enabled:
            out_scaled = acc_code.mul(scale_to_output)

            dtype = fp8_dtype(self.o_spec.fmt)
            max_val = fp8_max(self.o_spec.fmt)

            out_code = out_scaled.clamp(-max_val, max_val).to(dtype).float()
            out = out_code.mul(self.o_interval).to(x.dtype)
            return out

        # output 不量化
        out = F.linear(x, self.weight, self.bias)

        return out
    
    def _quant_forward_with_outlier(self, x, stat_collector=None):
        """带 outlier 保护的量化前向计算。"""
        outliermore = True
        channel_mask = self.get_outlier_mask_channel(x, self.outlier_ratio)

        x_channel_mask = channel_mask.view(1, 1, -1)   # [1, 1, H]
        w_channel_mask = channel_mask.view(1, -1)      # [1, H]
        
        if outliermore:
            w_outlier_mask = self._get_outlier_mask_1d(self.weight, self.outlier_ratio)
            w_channel_mask = w_channel_mask | w_outlier_mask
            del w_outlier_mask
        else:
            pass

        x_fp = x * x_channel_mask.to(torch.float32)        # x 的 outlier，保留 FP16
        x_normal_fp = x * (~x_channel_mask).to(dtype=x.dtype)
        x_normal_fp = x_normal_fp.to(torch.float32)

        w_fp = self.weight * w_channel_mask.to(torch.float32)        # w 的 outlier，保留 FP16
        w_normal_fp = self.weight * (~w_channel_mask).to(dtype=self.weight.dtype)
        w_normal_fp = w_normal_fp.to(torch.float32)

        del w_channel_mask
        del x_channel_mask
        """
        x_normal_mask = ~x_outlier_mask
        w_normal_mask = ~w_outlier_mask

        # === 分离 outlier 和 normal 部分（FP16）===
        x_fp = x * x_outlier_mask.to(torch.float32)        # x 的 outlier，保留 FP16
        x_normal_fp = x * x_normal_mask.to(torch.float32)  # x 的 normal，FP16（待量化）
        w_fp = self.weight * w_outlier_mask.to(torch.float32)        # w 的 outlier，保留 FP16
        w_normal_fp = self.weight * w_normal_mask.to(torch.float32)  # w 的 normal，FP16（待量化）
        """
        # === 量化 normal 部分 ===
        """
        M_qa_qb = torch.tensor(self.a_interval * self.w_interval / self.o_interval)
        M_qa_qb = self.round(M_qa_qb * LINEAR_SHIFT_NUM)
        """

        M_q   = torch.tensor(self.o_interval)
        M_q   = self.round(M_q * (2**16))

        M_aw    = torch.tensor(self.a_interval * self.w_interval)
        M_aw    = self.round(M_aw * 2**48)

        M_fa_qb = torch.tensor(self.w_interval)
        M_fa_qb = self.round(M_fa_qb * 2**24)

        M_qa_fb = torch.tensor(self.a_interval)
        M_qa_fb = self.round(M_qa_fb * 2**24)

        x_sim = quant_awo(
            x_normal_fp,
            self.a_interval,
            self.a_spec,
            out_dtype=x.dtype,
            chunk_size=1_048_576,
        )

        w_sim = quant_awo(
            w_normal_fp,
            self.w_interval,
            self.w_spec,
            out_dtype=self.weight.dtype,
            chunk_size=1_048_576,
        )

        in_features = self.weight.size(1)
        out_features = self.weight.size(0)
        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                x_sim,
                x_sim.to(torch.float16),
                self.a_spec,
                self.digit_size,
                self.parallelism,
                in_features,
                out_features
            )

        if self.bias is not None:
            bias_sim = self.quant_bias(self.bias).to(torch.float32)
            bias     = self.bias.to(torch.float32)
        else:
            bias_sim = None
            bias     = None

        x_sim_fp32 = x_sim.to(torch.float32)
        w_sim_fp32 = w_sim.to(torch.float32)
        """
        out_qa_qb = F.linear(x_normal_fp, w_normal_fp, bias)  # 量化的 normal 部分乘积，INT32 范围
        out_fa_fb = F.linear(x_fp, w_fp)  # 保留 FP16 的 outlier 部分乘积，FP32 范围
        out_fa_qb = F.linear(x_fp, w_normal_fp)  # 保留 FP16 的 outlier 部分乘积，FP32 范围
        out_qa_fb = F.linear(x_normal_fp, w_fp)  # 保留 FP16 的 outlier 部分乘积，FP32 范围

        """
        out_qa_qb = F.linear(x_sim_fp32, w_sim_fp32, bias_sim)  # 量化的 normal 部分乘积，INT32 范围
        out_fa_fb = F.linear(x_fp, w_fp)  # 保留 FP16 的 outlier 部分乘积，FP32 范围
        out_fa_qb = F.linear(x_fp, w_sim_fp32)  # 保留 FP16 的 outlier 部分乘积，FP32 范围
        out_qa_fb = F.linear(x_sim_fp32, w_fp)  # 保留 FP16 的 outlier 部分乘积，FP32 范围

        out_qa_qb = out_qa_qb.mul_(M_aw)
        out_qa_qb = torch.div(out_qa_qb, 2**48)

        out_fa_qb = out_fa_qb.mul_(M_fa_qb)
        out_fa_qb = torch.div(out_fa_qb, 2**24)

        out_qa_fb = out_qa_fb.mul_(M_qa_fb)
        out_qa_fb = torch.div(out_qa_fb, 2**24)

        #out_ref_qa_qb = F.linear(x_normal_fp.to(torch.float32), w_normal_fp.to(torch.float32), self.bias.to(torch.float32))
        #mse_qaqb = F.mse_loss(out_qa_qb, out_ref_qa_qb).item()
        if outliermore:
            out_with_outlier = out_qa_qb + out_fa_fb + out_fa_qb + out_qa_fb
        else:
            out_with_outlier = out_qa_qb + out_fa_fb
        #return  out_with_outlier.to(x.dtype)

        #out_ref = F.linear(x, self.weight, self.bias)
        #mse = F.mse_loss(out_with_outlier, out_ref).item()

        out_with_outlier_mask = self.get_outlier_mask_channel(out_with_outlier, self.outlier_ratio)
        out_without_outlier_mask = ~out_with_outlier_mask

        out_outlier = out_with_outlier * out_with_outlier_mask.to(torch.float32)        # outlier，保留 FP16
        out_normal = out_with_outlier * out_without_outlier_mask.to(torch.float32)  # normal，FP16（待量化）

        out_normal_quant = quant_awo(
            out_normal,
            self.o_interval,
            self.o_spec,
            out_dtype=out_normal.dtype,
            chunk_size=1_048_576,
        )

        out_normal_dequant = out_normal_quant.to(torch.float32).mul_(M_q).to(x.dtype)
        out_normal_dequant = torch.div(out_normal_dequant, 2**16).to(x.dtype)
        out_outlier = out_outlier.to(x.dtype)

        out = out_normal_dequant + out_outlier

        if torch.isnan(out).max():
            pass

        return out
    
    def _get_outlier_mask_1d(self, tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        返回 bool mask，True 表示是 outlier（保留 FP16）。
        离群值定义为绝对值最大的元素，数量约占总元素数的 ratio（至少1个，最多 numel-1 个）。
        当所有元素绝对值相等时，返回全 False。
        """
        if ratio <= 0.0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)
    
        numel = tensor.numel()
        if numel == 0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)
    
        # 至少选1个，最多选 numel-1 个，确保正常部分非空
        k = max(1, min(int(numel * ratio), numel - 1))
    
        flat_abs = tensor.abs().flatten()
        # 获取第 k 大的值
        threshold = torch.topk(flat_abs, k).values.min()
    
        # 处理阈值等于最小值的情况
        min_val = flat_abs.min()
        if threshold == min_val:
            # 只选严格大于最小值的元素，避免全选
            outlier_mask_flat = flat_abs > min_val
        else:
            outlier_mask_flat = flat_abs >= threshold
    
        # 如果离群数量为0（如全等值），返回全False
        if outlier_mask_flat.sum() == 0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)
    
        # 恢复原始形状
        return outlier_mask_flat.view(tensor.shape)


    def get_outlier_mask_channel(self, tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """带 outlier 保护的量化前向计算。"""

        # 兼容 [B, S, H] 和 [1, H] / [S, H] / [H] 等各种情况
        tensor_2d = tensor.reshape(-1, tensor.shape[-1])   # [N, H]
        channel_score = tensor_2d.abs().amax(dim=0)        # [H]

        k = max(1, int(channel_score.numel() * ratio))

        protected_idx = torch.topk(channel_score, k).indices

        channel_mask = torch.zeros_like(channel_score, dtype=torch.bool)
        channel_mask[protected_idx] = True

        return channel_mask

    def _load_scales(self):
        """Load quantization scales from files."""
        # Construct scale file paths
        w_scale_file = os.path.join(
            self.scale_root_str, 
            f"{self.layer_name}_w_scale_{self.layer_idx}.p"
        )
        a_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_a_scale_{self.layer_idx}.p"
        )
        o_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_o_scale_{self.layer_idx}.p"
        )
        
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
        w_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_w_scale_{self.layer_idx}.p"
        )
        with open(w_scale_file, 'wb') as f:
            pickle.dump(self.w_interval, f)
        
        # Save activation scale
        a_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_a_scale_{self.layer_idx}.p"
        )
        with open(a_scale_file, 'wb') as f:
            pickle.dump(self.a_interval, f)
        
        # Save output scale
        o_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_o_scale_{self.layer_idx}.p"
        )
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
