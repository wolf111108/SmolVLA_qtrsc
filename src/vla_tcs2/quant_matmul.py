"""
Quantized Matrix Multiplication Implementation.

Ported from opt-qt/quant/quant_matmul.py (verified working).

Provides quantized matmul for attention:
  - qk_matmul: Q @ K^T
  - pv_matmul: P @ V

Supported modes: raw / scale_inspection / quant_forward.
Supports 2D/3D/4D batched matmul via torch.matmul.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from vla_tcs2.quant.utils import Round, MATMUL_SHIFT_NUM
from vla_tcs2.quant.quant_spec import (
    QuantSpec,
    parse_quant_spec,
    safe_scale_from_tensor,
    safe_scale_per_token,
    quant_awo,
    fp8_dtype,
    fp8_max,
)


class MatMul(nn.Module):
    """Simple non-quantized matrix multiplication module."""

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


class QuantizedMatMul(nn.Module):
    """
    Quantized matrix multiplication layer (qk_matmul / pv_matmul).

    Supports flexible A/B/O formats:
      - INT: 8, 4, ...
      - FP: e4m3, e5m2
      - disabled output: fp16 / none (QuantSpec behavior)
    """

    def __init__(
        self,
        mode: str = "raw",
        A_bit: Optional[int] = None,
        B_bit: Optional[int] = None,
        O_bit: Optional[int] = None,
        scale_root_str: Optional[str] = None,
        d_bit: Optional[int] = None,
        p: Optional[int] = None,
        outlier_ratio: float = 0.0,
    ):
        super().__init__()

        self.mode = mode

        self.A_bit = A_bit
        self.B_bit = B_bit
        self.O_bit = O_bit

        self.A_spec: QuantSpec = parse_quant_spec(A_bit)
        self.B_spec: QuantSpec = parse_quant_spec(B_bit)
        self.O_spec: QuantSpec = parse_quant_spec(O_bit)

        self.digit_size = d_bit
        self.parallelism = p

        self.scale_root_str = scale_root_str or ""
        self.round = Round
        self.is_bitnet = False

        self.A_interval: Optional[float] = None
        self.B_interval: Optional[float] = None
        self.O_interval: Optional[float] = None

        self.layer_name = ""
        self.layer_idx = 0

        self.outlier_ratio = outlier_ratio

        self.calibration_policy = "recalibrate"
        self._calibration_action_cache = None

        # Mixed-precision per-token activation quantization
        self.mixed_precision = 0
        self.mp_high_ratio = 0.25
        self.mp_low_ratio = 0.2

    # =========================================================================
    # Layer info
    # =========================================================================

    def set_layer_info(self, layer_name: str, layer_idx: int):
        self.layer_name = layer_name
        self.layer_idx = layer_idx

    # =========================================================================
    # Helpers
    # =========================================================================

    def _matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """General batched matmul (2D/3D/4D/...)."""
        return torch.matmul(A, B)

    def _check_bits(self):
        if self.A_bit is None or self.B_bit is None or self.O_bit is None:
            raise ValueError(
                f"{self.layer_name}_{self.layer_idx}: "
                f"A_bit/B_bit/O_bit must be set. "
                f"Got A_bit={self.A_bit}, B_bit={self.B_bit}, O_bit={self.O_bit}."
            )

        if self.A_spec is None or self.B_spec is None or self.O_spec is None:
            raise ValueError(
                f"{self.layer_name}_{self.layer_idx}: invalid QuantSpec. "
                f"A_spec={self.A_spec}, B_spec={self.B_spec}, O_spec={self.O_spec}"
            )

    def _scale_file_paths(self):
        return (
            os.path.join(self.scale_root_str, f"{self.layer_name}_A_scale_{self.layer_idx}.p"),
            os.path.join(self.scale_root_str, f"{self.layer_name}_B_scale_{self.layer_idx}.p"),
            os.path.join(self.scale_root_str, f"{self.layer_name}_O_scale_{self.layer_idx}.p"),
        )

    def _scale_files_exist(self) -> bool:
        return all(os.path.exists(p) for p in self._scale_file_paths())

    def _resolve_calibration_action(self) -> str:
        if self._calibration_action_cache is not None:
            return self._calibration_action_cache

        policy = str(self.calibration_policy).lower()

        if policy == "auto":
            action = "reuse" if self._scale_files_exist() else "recalibrate"
        elif policy in {"reuse", "recalibrate"}:
            action = policy
        else:
            raise ValueError(
                f"Invalid calibration_policy={self.calibration_policy} "
                f"for {self.layer_name}_{self.layer_idx}"
            )

        self._calibration_action_cache = action
        return action

    # =========================================================================
    # Forward dispatch
    # =========================================================================

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        collect_stats: bool = False,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        if stat_collector is None and hasattr(self, "_stat_manager"):
            stat_collector = self._stat_manager

        if self.mode == "raw":
            if self.is_bitnet:
                return self.bitnet_forward(A, B, stat_collector)
            return self._matmul(A, B)

        if self.mode == "scale_inspection":
            if self.is_bitnet:
                return self.bitnet_forward(A, B, stat_collector)
            return self.scale_inspection(A, B, stat_collector)

        if self.mode == "quant_forward":
            if self.is_bitnet:
                return self.bitnet_forward(A, B, stat_collector)
            return self.raw_forward(A, B, stat_collector)

        raise NotImplementedError(f"Mode {self.mode} not implemented")

    # =========================================================================
    # Scale inspection
    # =========================================================================

    def scale_inspection(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """Inspect and collect scale statistics for calibration."""
        out = self._matmul(A, B)
        action = self._resolve_calibration_action()
        if action == "reuse":
            return out

        if self.mixed_precision and self.outlier_ratio == 0.0:
            return self._scale_inspection_mixed_precision(A, B, stat_collector)

        if self.outlier_ratio > 0.0:
            outliermore = True
            channel_mask = self.get_outlier_mask_channel(A, self.outlier_ratio)

            # A: feature dim is always dim=-1
            A_channel_mask = channel_mask.view(*([1] * (A.dim() - 1)), -1)
            # B: feature dim may be -1 (pv_matmul) or -2 (qk_matmul after transpose)
            if B.shape[-1] == channel_mask.numel():
                B_channel_mask = channel_mask.view(*([1] * (B.dim() - 1)), -1)
            elif B.dim() >= 2 and B.shape[-2] == channel_mask.numel():
                B_channel_mask = channel_mask.view(*([1] * (B.dim() - 2)), -1, 1)
            else:
                B_channel_mask = torch.zeros(1, dtype=torch.bool, device=A.device)
            del channel_mask
            if outliermore:
                B_outlier_mask = self._get_outlier_mask_1d(B, self.outlier_ratio)
                B_channel_mask = B_channel_mask | B_outlier_mask
                del B_outlier_mask
            else:
                pass

            A_normal_fp = A * (~A_channel_mask).to(dtype=A.dtype)
            A_normal_fp = A_normal_fp.to(torch.float32)

            B_normal_fp = B * (~B_channel_mask).to(dtype=B.dtype)
            B_normal_fp = B_normal_fp.to(torch.float32)

            del B_channel_mask
            del A_channel_mask

            self.A_interval = safe_scale_from_tensor(A_normal_fp, self.A_spec)
            if self.A_interval == 0:
                self.A_interval = None

            self.B_interval = safe_scale_from_tensor(B_normal_fp, self.B_spec)

            del A_normal_fp
            del B_normal_fp

            channel_mask = self.get_outlier_mask_channel(out, self.outlier_ratio)

            normal_idx = torch.nonzero(~channel_mask, as_tuple=False).flatten()

            O_normal_fp = out.index_select(dim=-1, index=normal_idx).to(torch.float32)

            self.O_interval = safe_scale_from_tensor(O_normal_fp, self.O_spec)
        else:
            self.A_interval = safe_scale_from_tensor(A, self.A_spec)
            self.B_interval = safe_scale_from_tensor(B, self.B_spec)
            self.O_interval = safe_scale_from_tensor(out, self.O_spec)

        if stat_collector is not None:
            stat_collector.collect_matmul_stats(
                self.layer_name,
                self.layer_idx,
                self.A_interval,
                self.B_interval,
                self.O_interval,
            )

        # Return FP output for numerical stability during calibration
        return out

    def _scale_inspection_mixed_precision(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """Calibration aligned with the mixed-precision inference path."""
        out_real = self._matmul(A, B)

        self.O_interval = safe_scale_from_tensor(out_real, self.O_spec)
        self.B_interval = safe_scale_from_tensor(B, self.B_spec)
        self.A_interval = 0

        if stat_collector is not None:
            stat_collector.collect_matmul_stats(
                self.layer_name,
                self.layer_idx,
                self.A_interval,
                self.B_interval,
                self.O_interval,
            )

        return out_real

    # =========================================================================
    # Quantized forward
    # =========================================================================

    def raw_forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """Forward without quantization (mode 'raw')."""
        out = self._matmul(A, B)

        in_features = A.size(-1)
        out_features = B.size(-1)

        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                A,
                A,
                B,
                self.B_spec,
                self.A_spec,
                self.digit_size,
                self.parallelism,
                in_features,
                out_features,
            )

        return out

    def quant_forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """
        Flexible quantized matmul.

        A ~= A_interval * A_code
        B ~= B_interval * B_code
        acc_code = A_code @ B_code

        Output is handled according to O_spec:
          - INT: floor/round into integer output code
          - FP: cast into FP8 output code
          - disabled: no output quantization
        """
        self._check_bits()
        self._load_scales()

        if self.outlier_ratio > 0.0:
            return self._quant_forward_with_outlier(A, B, stat_collector)

        A_sim = quant_awo(
            A,
            self.A_interval,
            self.A_spec,
            out_dtype=torch.float32,
            chunk_size=1_048_576,
        )

        B_sim = quant_awo(
            B,
            self.B_interval,
            self.B_spec,
            out_dtype=torch.float32,
            chunk_size=1_048_576,
        )

        in_features = A.size(-1)
        out_features = B.size(-1)

        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                f"{self.layer_name}",
                self.layer_idx,
                A_sim,
                A,
                B_sim,
                self.B_spec,
                self.A_spec,
                self.digit_size,
                self.parallelism,
                in_features,
                out_features,
            )

        acc_code = self._matmul(A_sim, B_sim)

        scale_to_output = self.A_interval * self.B_interval / self.O_interval

        if self.O_spec.kind == "int":
            M0 = torch.tensor(
                scale_to_output,
                device=A.device,
                dtype=torch.float32,
            )
            M0 = self.round(M0 * MATMUL_SHIFT_NUM)

            out_code = acc_code.mul(M0)
            out_code = torch.div(
                out_code,
                MATMUL_SHIFT_NUM,
                rounding_mode="floor",
            )

            out = out_code.mul(self.O_interval).to(A.dtype)

        elif self.O_spec.kind == "fp":
            out_scaled = acc_code.mul(scale_to_output)

            dtype = fp8_dtype(self.O_spec.fmt)
            max_val = fp8_max(self.O_spec.fmt)

            out_code = out_scaled.clamp(-max_val, max_val).to(dtype).float()
            out = out_code.mul(self.O_interval).to(A.dtype)

        elif self.O_spec.kind == "bf":
            M0 = torch.tensor(
                scale_to_output,
                device=A.device,
                dtype=torch.float32,
            )
            M0 = self.round(M0 * MATMUL_SHIFT_NUM)

            out_code = acc_code.mul(M0)
            out_code = torch.div(
                out_code,
                MATMUL_SHIFT_NUM,
                rounding_mode="floor",
            )

            out = out_code.mul(self.O_interval).to(A.dtype)

        else:
            out = self._matmul(A, B)

        return out

    def _quant_forward_with_outlier(self, A, B, stat_collector=None):
        """Outlier-protected quantized forward."""
        outliermore = True
        channel_mask = self.get_outlier_mask_channel(A, self.outlier_ratio)

        # A: feature dim is always dim=-1
        A_channel_mask = channel_mask.view(*([1] * (A.dim() - 1)), -1)
        # B: feature dim may be -1 (pv_matmul) or -2 (qk_matmul after transpose)
        if B.shape[-1] == channel_mask.numel():
            B_channel_mask = channel_mask.view(*([1] * (B.dim() - 1)), -1)
        elif B.dim() >= 2 and B.shape[-2] == channel_mask.numel():
            B_channel_mask = channel_mask.view(*([1] * (B.dim() - 2)), -1, 1)
        else:
            B_channel_mask = torch.zeros(1, dtype=torch.bool, device=A.device)
        del channel_mask
        if outliermore:
            B_outlier_mask = self._get_outlier_mask_1d(B, self.outlier_ratio)
            B_channel_mask = B_channel_mask | B_outlier_mask
            del B_outlier_mask
        else:
            pass

        A_fp = A * A_channel_mask.to(torch.float32)        # A outlier, keep FP
        A_normal_fp = A * (~A_channel_mask).to(dtype=A.dtype)
        A_normal_fp = A_normal_fp.to(torch.float32)

        B_fp = B * B_channel_mask.to(torch.float32)        # B outlier, keep FP
        B_normal_fp = B * (~B_channel_mask).to(dtype=B.dtype)
        B_normal_fp = B_normal_fp.to(torch.float32)

        del B_channel_mask
        del A_channel_mask

        M_q = torch.tensor(self.O_interval)
        M_q = self.round(M_q * (2 ** 16))

        M_aw = torch.tensor(self.A_interval * self.B_interval)
        M_aw = self.round(M_aw * 2 ** 48)

        M_fa_qb = torch.tensor(self.B_interval)
        M_fa_qb = self.round(M_fa_qb * 2 ** 24)

        M_qa_fb = torch.tensor(self.A_interval)
        M_qa_fb = self.round(M_qa_fb * 2 ** 24)

        A_sim = quant_awo(
            A_normal_fp,
            self.A_interval,
            self.A_spec,
            out_dtype=A.dtype,
            chunk_size=1_048_576,
        )

        B_sim = quant_awo(
            B_normal_fp,
            self.B_interval,
            self.B_spec,
            out_dtype=B.dtype,
            chunk_size=1_048_576,
        )

        in_features = A.size(-1)
        out_features = B.size(-1)
        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                A_sim.to(torch.float16),
                A_sim.to(torch.float16),
                B_sim,
                self.B_spec,
                self.A_spec,
                self.digit_size,
                self.parallelism,
                in_features,
                out_features,
            )

        A_sim_fp32 = A_sim.to(torch.float32)
        B_sim_fp32 = B_sim.to(torch.float32)

        out_qa_qb = self._matmul(A_sim_fp32, B_sim_fp32)
        out_fa_fb = self._matmul(A_fp, B_fp)
        out_fa_qb = self._matmul(A_fp, B_sim_fp32)
        out_qa_fb = self._matmul(A_sim_fp32, B_fp)

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

        out_with_outlier_mask = self.get_outlier_mask_channel(out_with_outlier, self.outlier_ratio)
        out_without_outlier_mask = ~out_with_outlier_mask

        out_outlier = out_with_outlier * out_with_outlier_mask.to(torch.float32)
        out_normal = out_with_outlier * out_without_outlier_mask.to(torch.float32)

        out_normal_quant = quant_awo(
            out_normal,
            self.O_interval,
            self.O_spec,
            out_dtype=out_normal.dtype,
            chunk_size=1_048_576,
        )

        out_normal_dequant = out_normal_quant.to(torch.float32).mul_(M_q).to(A.dtype)
        out_normal_dequant = torch.div(out_normal_dequant, 2 ** 16).to(A.dtype)
        out_outlier = out_outlier.to(A.dtype)

        out = out_normal_dequant + out_outlier

        return out

    # =========================================================================
    # Mixed precision
    # =========================================================================

    def _classify_token_importance(
        self,
        x: torch.Tensor,
    ) -> tuple:
        """Classify tokens by peak-to-average ratio (PAR), preserving execution order."""
        if x.dim() != 2:
            raise ValueError(f"token classifier expects [N, H], got {tuple(x.shape)}")
        if not (0.0 <= self.mp_high_ratio <= 1.0):
            raise ValueError("mp_high_ratio must be in [0, 1]")
        if not (0.0 <= self.mp_low_ratio <= 1.0):
            raise ValueError("mp_low_ratio must be in [0, 1]")
        if self.mp_high_ratio + self.mp_low_ratio > 1.0:
            raise ValueError("mp_high_ratio + mp_low_ratio must be <= 1")

        x_f32 = x.detach().to(torch.float32).abs()
        mean_abs = x_f32.mean(dim=-1)
        max_abs = x_f32.amax(dim=-1)
        token_scores = max_abs / mean_abs.clamp(min=1e-8)
        n_tokens = token_scores.numel()
        if n_tokens == 0:
            empty = torch.empty(0, dtype=torch.long, device=x.device)
            return empty, empty, empty

        n_high = int(round(n_tokens * self.mp_high_ratio))
        n_low = int(round(n_tokens * self.mp_low_ratio))
        n_high = min(max(n_high, 0), n_tokens)
        n_low = min(max(n_low, 0), n_tokens - n_high)
        n_mid = n_tokens - n_high - n_low

        sorted_indices = torch.argsort(token_scores, descending=True)
        high_idx = sorted_indices[:n_high]
        mid_idx = sorted_indices[n_high:n_high + n_mid]
        low_idx = sorted_indices[n_high + n_mid:]
        return high_idx, mid_idx, low_idx

    def _mixed_precision_specs(self):
        """Return the three activation formats used by the synthetic MP policy."""
        return (
            QuantSpec(kind="fp", fmt="e5m10", enabled=False),
            QuantSpec(kind="fp", fmt="e4m3", enabled=True),
            QuantSpec(kind="fp", fmt="e2m1", enabled=True),
        )

    def _quant_forward_mixed_precision(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """Per-token FP16/FP8/FP4 activation quantization."""
        original_shape = A.shape
        A_2d = A.reshape(-1, original_shape[-1]).to(torch.float32)
        n_tokens = A_2d.shape[0]

        high_idx, mid_idx, low_idx = self._classify_token_importance(A_2d)
        high_spec, mid_spec, low_spec = self._mixed_precision_specs()

        A_deq_all = torch.empty_like(A_2d)
        code_all = torch.empty_like(A_2d)
        mp_codes = []  # [(local_code, global_indices, spec), ...]

        for indices, spec in (
            (high_idx, high_spec),
            (mid_idx, mid_spec),
            (low_idx, low_spec),
        ):
            if indices.numel() == 0:
                continue
            code, _, dequantized = self._quantize_activation_per_token(
                A_2d.index_select(0, indices), spec
            )
            A_deq_all.index_copy_(0, indices, dequantized)
            code_all.index_copy_(0, indices, code)
            mp_codes.append((code, indices, spec))

        # Keep 2D code_all for statistics, reshape A_deq_all for matmul
        code_all_2d = code_all
        A_deq_all = A_deq_all.reshape(original_shape)

        B_code = quant_awo(
            B,
            self.B_interval,
            self.B_spec,
            out_dtype=torch.float32,
            chunk_size=1_048_576,
        ).to(torch.float32)

        out_real = self._matmul(A_deq_all, B_code) * self.B_interval
        in_features = A.size(-1)
        out_features = B.size(-1)
        if stat_collector is not None:
            collect_fn = getattr(
                stat_collector, "collect_quant_activation_mixed_precision", None
            )
            if callable(collect_fn):
                collect_fn(
                    self.layer_name,
                    self.layer_idx,
                    code_all_2d,
                    mp_codes,
                    B_code,
                    self.B_spec,
                    self.digit_size,
                    self.parallelism,
                    in_features,
                    out_features,
                )

        # ---- output quantization ----
        if self.O_spec.kind in {"int", "fp", "bf"} and getattr(
            self.O_spec, "enabled", True
        ):
            out_code = quant_awo(
                out_real,
                self.O_interval,
                self.O_spec,
                out_dtype=torch.float32,
                chunk_size=1_048_576,
            ).to(torch.float32)
            out = (out_code * self.O_interval).to(A.dtype)
        else:
            out = out_real.to(A.dtype)
        return out.reshape(*original_shape[:-1], out_features)

    @staticmethod
    def _safe_token_scale_from_max(
        x: torch.Tensor,
        format_max: float,
    ) -> torch.Tensor:
        """Return a finite positive per-token scale with shape [..., 1]."""
        x_f32 = torch.nan_to_num(
            x.detach().to(torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        max_abs = x_f32.abs().amax(dim=-1, keepdim=True)
        scale = max_abs / float(format_max)
        return torch.where(
            torch.isfinite(scale) & (scale > 0),
            scale,
            torch.ones_like(scale),
        )

    @staticmethod
    def _fake_quantize_e2m1(normalized: torch.Tensor) -> torch.Tensor:
        """Quantize normalized values to the signed E2M1 finite grid."""
        v = torch.nan_to_num(
            normalized.to(torch.float32),
            nan=0.0,
            posinf=6.0,
            neginf=-6.0,
        ).clamp(-6.0, 6.0)
        abs_v = v.abs()
        qmag = torch.zeros_like(abs_v)
        qmag = torch.where(abs_v >= 0.25, torch.full_like(qmag, 0.5), qmag)
        qmag = torch.where(abs_v >= 0.75, torch.full_like(qmag, 1.0), qmag)
        qmag = torch.where(abs_v >= 1.25, torch.full_like(qmag, 1.5), qmag)
        qmag = torch.where(abs_v >= 1.75, torch.full_like(qmag, 2.0), qmag)
        qmag = torch.where(abs_v >= 2.50, torch.full_like(qmag, 3.0), qmag)
        qmag = torch.where(abs_v >= 3.50, torch.full_like(qmag, 4.0), qmag)
        qmag = torch.where(abs_v >= 5.00, torch.full_like(qmag, 6.0), qmag)
        return torch.copysign(qmag, v)

    def _quantize_activation_per_token(
        self,
        x: torch.Tensor,
        spec: QuantSpec,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize one token subset and return (code, scale, dequantized)."""
        x_f32 = torch.nan_to_num(
            x.to(torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        fmt = (getattr(spec, "fmt", "") or "").lower().strip()

        if fmt in {"e5m10", "fp16", "float16"}:
            scale = torch.ones(
                *x_f32.shape[:-1], 1,
                dtype=torch.float32,
                device=x_f32.device,
            )
            code = x_f32.to(torch.float16).to(torch.float32)
            return code, scale, code

        if fmt in {"e4m3", "fp8_e4m3", "float8_e4m3"}:
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError(
                    "This PyTorch build does not support torch.float8_e4m3fn"
                )
            scale = self._safe_token_scale_from_max(x_f32, 448.0)
            normalized = (x_f32 / scale).clamp(-448.0, 448.0)
            code = normalized.to(torch.float8_e4m3fn).to(torch.float32)
            return code, scale, code * scale

        if fmt in {"e2m1", "fp4", "fp4_e2m1"}:
            scale = self._safe_token_scale_from_max(x_f32, 6.0)
            code = self._fake_quantize_e2m1(x_f32 / scale)
            return code, scale, code * scale

        # Fixed-format dynamic activation (INT or other formats)
        scale = safe_scale_per_token(x_f32, spec, dim=-1).to(torch.float32)
        scale = torch.where(
            torch.isfinite(scale) & (scale > 0),
            scale,
            torch.ones_like(scale),
        )
        code = quant_awo(
            x_f32,
            scale,
            spec,
            out_dtype=torch.float32,
            chunk_size=1_048_576,
        ).to(torch.float32)
        return code, scale, code * scale

    # =========================================================================
    # Outlier helpers
    # =========================================================================

    def get_outlier_mask_channel(self, tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """Compute per-channel outlier mask (True = protected channel)."""
        tensor_2d = tensor.reshape(-1, tensor.shape[-1])   # [N, H]
        channel_score = tensor_2d.abs().amax(dim=0)        # [H]

        k = max(1, int(channel_score.numel() * ratio))

        protected_idx = torch.topk(channel_score, k).indices

        channel_mask = torch.zeros_like(channel_score, dtype=torch.bool)
        channel_mask[protected_idx] = True

        return channel_mask

    def _get_outlier_mask_1d(self, tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """Return bool mask, True = outlier (kept in FP)."""
        if ratio <= 0.0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)

        numel = tensor.numel()
        if numel == 0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)

        k = max(1, min(int(numel * ratio), numel - 1))

        flat_abs = tensor.abs().flatten()
        threshold = torch.topk(flat_abs, k).values.min()

        min_val = flat_abs.min()
        if threshold == min_val:
            outlier_mask_flat = flat_abs > min_val
        else:
            outlier_mask_flat = flat_abs >= threshold

        if outlier_mask_flat.sum() == 0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)

        return outlier_mask_flat.view(tensor.shape)

    # =========================================================================
    # BitNet (kept for compatibility; unused on SmolVLA path)
    # =========================================================================

    def bitnet_forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        if B.dim() >= 3:
            in_features = A.shape[-1]
            out_features = B.shape[-1]
        else:
            in_features = B.size(1)
            out_features = B.size(0)

        A_sim = A
        B_sim = B

        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                f"{self.layer_name}",
                self.layer_idx,
                A_sim,
                A,
                B_sim,
                self.B_spec,
                self.A_spec,
                self.digit_size,
                self.parallelism,
                in_features,
                out_features,
            )

        return self._matmul(A, B)

    # =========================================================================
    # Scale persistence
    # =========================================================================

    def _load_scales(self):
        A_scale_file, B_scale_file, O_scale_file = self._scale_file_paths()

        missing = [
            path for path in (A_scale_file, B_scale_file, O_scale_file)
            if not os.path.exists(path)
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing scale files for {self.layer_name}_{self.layer_idx}: "
                f"{missing}"
            )

        with open(A_scale_file, "rb") as f:
            self.A_interval = pickle.load(f)

        with open(B_scale_file, "rb") as f:
            self.B_interval = pickle.load(f)

        with open(O_scale_file, "rb") as f:
            self.O_interval = pickle.load(f)

        if self.A_interval is None or self.B_interval is None or self.O_interval is None:
            raise ValueError(
                f"Invalid loaded scales for {self.layer_name}_{self.layer_idx}: "
                f"A={self.A_interval}, B={self.B_interval}, O={self.O_interval}"
            )

    def save_scales(self):
        if self.A_interval is None or self.B_interval is None or self.O_interval is None:
            raise ValueError(
                f"Scales not computed for {self.layer_name}_{self.layer_idx}. "
                f"Run scale_inspection first."
            )

        os.makedirs(self.scale_root_str, exist_ok=True)

        A_scale_file, B_scale_file, O_scale_file = self._scale_file_paths()

        with open(A_scale_file, "wb") as f:
            pickle.dump(self.A_interval, f)

        with open(B_scale_file, "wb") as f:
            pickle.dump(self.B_interval, f)

        with open(O_scale_file, "wb") as f:
            pickle.dump(self.O_interval, f)

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, "
            f"A_bit={self.A_bit}, "
            f"B_bit={self.B_bit}, "
            f"O_bit={self.O_bit}, "
            f"A_spec={self.A_spec.name()}, "
            f"B_spec={self.B_spec.name()}, "
            f"O_spec={self.O_spec.name()}, "
            f"layer={self.layer_name}_{self.layer_idx}"
        )

    def log_quant_error(
        self,
        value: float,
        layer_name: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ):
        """Append quantization MSE log."""
        layer_name = layer_name if layer_name is not None else self.layer_name
        layer_idx = layer_idx if layer_idx is not None else self.layer_idx

        if self.scale_root_str is None or self.scale_root_str == "":
            return

        try:
            os.makedirs(self.scale_root_str, exist_ok=True)
            log_file = os.path.join(self.scale_root_str, "quant_error_log.jsonl")

            record = {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "mse": float(value),
                "mode": self.mode,
                "A_format": self.A_spec.name(),
                "B_format": self.B_spec.name(),
                "O_format": self.O_spec.name(),
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        except Exception:
            pass
