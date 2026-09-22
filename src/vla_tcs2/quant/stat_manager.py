"""
Statistics Manager for Quantization Calibration.

Ported (simplified) from opt-qt/quant/stat_manager.py.

Kept:
    - QuantStatistics: per-layer scale collection (w/a/o, A/B for matmul)
    - QuantStatManager: registry + pickle save/load + summary printing
      (file naming matches QuantizedLinear._load_scales /
       QuantizedLinear.save_scales: {layer_name}_{type}_{layer_idx}.p)
    - Sparsity accounting ported from opt-qt stat_manager_old.py (opt-in):
        * global + per-phase (full_forward/prefill/decode) counters
        * chunked INT bit statistics (compute_sparse_stats)
        * chunked FP sign+mantissa bit statistics (compute_sparse_stats_fp,
          incl. hidden-leading-1 semantics, E4M3/E2M1/E5M10 raw-bit paths)
        * unit/block sparsity (a-bit x b-dim groups)
        * per-layer sparsity records + CSV export + printing

Dropped (not needed on the VLA-TCS2 path):
    - HW mapping / latency statistics (Mapping_stat, CIM_sys, SACIM, ...)
      -> old code stored sparsity inside per_layer_latency; the port uses a
         dedicated per_layer_sparsity dict instead.

Usage notes:
    - Sparsity collection is OFF by default (zero overhead / no behavior
      change for existing runs). Call enable_sparsity() on the manager to
      turn it on; set_phase("prefill" | "decode" | "full_forward") tags
      which phase counters are accumulated.
    - collect_quant_activation dispatches on the two existing call-site
      signatures (linear 7 positional args, matmul 9 positional args after
      layer_name/layer_idx) and collects for A and B separately on matmul.
"""

import math
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .quant_spec import QuantSpec
from ..runtime_context import get_runtime_context


# =============================================================================
# Per-layer statistics
# =============================================================================


class QuantStatistics:
    """
    Statistics collector for a single layer.

    Collects scale information for activations, weights, and outputs
    during calibration.
    """

    def __init__(self, layer_name: str, layer_idx: int):
        self.layer_name = layer_name
        self.layer_idx = layer_idx

        # Scale statistics (linear)
        self.w_scales: List[float] = []
        self.a_scales: List[float] = []
        self.o_scales: List[float] = []

        # For matmul layers
        self.A_scales: List[float] = []
        self.B_scales: List[float] = []

        # Sample count
        self.sample_count = 0

    def collect_linear_stats(
        self,
        w_scale: float,
        a_scale: float,
        o_scale: float,
    ):
        """Collect statistics for linear layer."""
        self.w_scales.append(w_scale)
        self.a_scales.append(a_scale)
        self.o_scales.append(o_scale)
        self.sample_count += 1

    def collect_matmul_stats(
        self,
        A_scale: float,
        B_scale: float,
        O_scale: float,
    ):
        """Collect statistics for matmul."""
        self.A_scales.append(A_scale)
        self.B_scales.append(B_scale)
        self.o_scales.append(O_scale)
        self.sample_count += 1

    def get_final_scales(self) -> Dict[str, float]:
        """
        Finalize scales across calibration samples.
        Uses maximum value across all samples for robustness.

        Per-output-channel weight scales (G2-B, [N_out] tensors from
        scales_with_pot_ao_outlier_channel) aggregate elementwise via
        torch.maximum; scalar scales keep the legacy max() path.
        """
        scales = {}

        if self.w_scales:
            if any(
                isinstance(s, torch.Tensor) for s in self.w_scales
            ):
                # Elementwise max over per-channel tensors (all samples
                # must share shape [N_out]; pads with the scalar side by
                # broadcasting a full-tensor fallback).
                tensors = [
                    s if isinstance(s, torch.Tensor)
                    else torch.full_like(self.w_scales[0], float(s))
                    for s in self.w_scales
                ]
                scales['w_scale'] = torch.stack(tensors).amax(dim=0)
            else:
                scales['w_scale'] = max(self.w_scales)
            scales['a_scale'] = max(self.a_scales)
            scales['o_scale'] = max(self.o_scales)

        if self.A_scales:
            scales['A_scale'] = max(self.A_scales)
            scales['B_scale'] = max(self.B_scales)
            if self.o_scales:
                scales['O_scale'] = max(self.o_scales)

        return scales

    def reset(self):
        """Reset all collected statistics."""
        self.w_scales.clear()
        self.a_scales.clear()
        self.o_scales.clear()
        self.A_scales.clear()
        self.B_scales.clear()
        self.sample_count = 0


# =============================================================================
# Global manager
# =============================================================================


class QuantStatManager:
    """
    Global statistics manager for quantization calibration.
    """

    # Valid phase tags for sparsity accounting.
    _PHASES = ("full_forward", "prefill", "decode")

    def __init__(self, scale_dir: str):
        self.scale_dir = scale_dir
        self.stats: Dict[str, QuantStatistics] = {}

        os.makedirs(scale_dir, exist_ok=True)

        # Bookkeeping for collect_quant_activation auditing (VLA path).
        self.quant_activation_calls: Dict[str, int] = {}

        # ------------------------------------------------------------------
        # Sparsity accounting (ported from opt-qt stat_manager_old.py).
        # Disabled by default; enable_sparsity() switches it on.
        # ------------------------------------------------------------------
        self.sparsity_enabled = False
        self.current_phase = "full_forward"
        self.sparse_stat_chunk_size = 1_048_576

        # Global counters (all phases combined).
        self.total_zero_count = 0
        self.total_element_count = 0
        self.total_bit_count = 0
        self.total_0bit_count = 0
        self.total_sparsebit_count = 0
        self.total_amplitude_zero_bits_total = 0
        self.total_amplitude_bit_count = 0

        # Per-phase counters.
        self.phase_sparsity = {
            phase: self._new_sparsity_counter()
            for phase in self._PHASES
        }

        # Unit/block sparsity config: a bits x b dims per unit.
        self.enable_unit_sparsity = True
        self.unit_bit_group_size = 2
        self.unit_dim_group_size = 2
        # phase -> layer_key -> {"zero_units": int, "total_units": int}
        self.unit_sparsity: Dict[str, Dict[str, Dict[str, int]]] = {
            phase: {} for phase in self._PHASES
        }

        # Collected layer bookkeeping per phase (debug/audit).
        self.collected_layer_names_by_phase = {
            phase: set() for phase in self._PHASES
        }
        self.collected_layer_call_count_by_phase = {
            phase: {} for phase in self._PHASES
        }

        # Per-layer sparsity records: key = f"{layer_name}_{layer_idx}".
        # Legacy records (module-level aggregate); structured per-role /
        # per-phase records live in per_role_sparsity (key includes role).
        self.per_layer_sparsity: Dict[str, Dict[str, Any]] = {}

        # Structured records: key = (layer_key, phase, tensor_role).
        # Accumulated counters per manual §21-§22 (sum numerators and
        # denominators first, ratios computed at export time).
        self.per_role_sparsity: Dict[tuple, Dict[str, Any]] = {}

        # Denoise flow-step call counts: flow_step -> layer_key -> calls.
        self.flow_step_sparsity: Dict[int, Dict[str, int]] = {}

        # Per-layer tensor-role call counts (audit: Linear in/out, MatMul A/B/O).
        self.tensor_role_calls: Dict[str, Dict[str, int]] = {}

        # Legacy last-observed tensor dims (kept for backward compatibility).
        # NOTE: this is role-agnostic and therefore MUST NOT be used to infer
        # MatMul physical MACs: A/B/O have different shapes and O arrives last.
        self.module_last_dims: Dict[str, tuple] = {}

        # Exact MatMul workload accounting keyed by
        # (module_id, phase, flow_step, attention_kind). A is observed before
        # O in every quantized MatMul forward; on O collection we accumulate
        # physical MACs = O.numel() * K where K = A.shape[-1].
        self.matmul_workload: Dict[tuple, Dict[str, Any]] = {}
        self._pending_matmul_k: Dict[tuple, int] = {}

        # Outlier side-path accounting (manual §26-§27): per module_id,
        # counts of calls with dynamic outlier protection active and the
        # protected element fraction (activation/weight sides).
        self.outlier_sidepath: Dict[str, Dict[str, float]] = {}

        # Structured outlier FP side-path partition (Phase H, manual §3):
        # key = (module_id, phase, flow_step, tensor_role, attention_kind),
        # value accumulates protected/normal element & bit counters so the
        # "native quant path" sparsity can exclude protected positions that
        # the outlier forward artificially zeroes in the normal path.
        self.outlier_partition: Dict[tuple, Dict[str, Any]] = {}

        # Structured unit/block sparsity (Phase H, manual §5):
        # key matches the main structured record (module_id, phase,
        # flow_step, tensor_role, attention_kind).
        self.per_role_unit_sparsity: Dict[tuple, Dict[str, int]] = {}

        # Per-layer WEIGHT sparsity records (static, collected once).
        self.per_layer_weight_sparsity: Dict[str, Dict[str, Any]] = {}

        # ------------------------------------------------------------------
        # Phase H FP-code audit (H1-Audit)
        # Debug-only: inspect the actual post-quant FP8 code distribution.
        # Independent of _extract_sm_from_raw(); never modifies tensors/RNG.
        # ------------------------------------------------------------------
        self.fp_code_audit_enabled = False
        # Deterministic max elements sampled per call (0/None = all).
        self.fp_code_audit_max_elements_per_call = 262_144
        self.fp_code_audit_roles = {
            "activation", "output", "A", "B", "O",
        }
        # key: (module_id, phase, flow_step, tensor_role, attention_kind)
        self.fp_code_audit: Dict[tuple, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # Sparsity: configuration & counters
    # -------------------------------------------------------------------------

    def enable_sparsity(
        self,
        enable: bool = True,
        chunk_size: Optional[int] = None,
    ):
        """Turn sparsity collection on/off (off by default)."""
        self.sparsity_enabled = bool(enable)
        if chunk_size is not None:
            self.sparse_stat_chunk_size = int(chunk_size)

    def configure_unit_sparsity(
        self,
        enable: bool = True,
        bit_group_size: int = 2,
        dim_group_size: int = 2,
    ):
        """Configure unit/block sparsity (a bits x b dims per unit)."""
        self.enable_unit_sparsity = enable
        self.unit_bit_group_size = int(bit_group_size)
        self.unit_dim_group_size = int(dim_group_size)

        if self.unit_bit_group_size <= 0:
            raise ValueError("unit_bit_group_size must be > 0")
        if self.unit_dim_group_size <= 0:
            raise ValueError("unit_dim_group_size must be > 0")

    def configure_fp_code_audit(
        self,
        enable: bool = False,
        max_elements_per_call: int = 262_144,
        roles=None,
    ):
        """Configure the independent FP-code audit (H1-Audit, debug-only)."""
        self.fp_code_audit_enabled = bool(enable)

        if max_elements_per_call is None:
            self.fp_code_audit_max_elements_per_call = 0
        else:
            self.fp_code_audit_max_elements_per_call = int(
                max_elements_per_call
            )

        if roles is not None:
            self.fp_code_audit_roles = set(roles)

    def set_phase(self, phase: str):
        """Tag subsequent collect calls with a phase (prefill/decode/...)."""
        if phase not in self.phase_sparsity:
            raise ValueError(f"Unknown sparsity phase: {phase}")
        self.current_phase = phase

    @staticmethod
    def _new_sparsity_counter() -> Dict[str, int]:
        return {
            "total_zero_count": 0,
            "total_element_count": 0,
            "total_bit_count": 0,
            "total_0bit_count": 0,
            "total_sparsebit_count": 0,
            "total_amplitude_zero_bits_total": 0,
            "total_amplitude_bit_count": 0,
        }

    def reset_sparsity(self):
        """Reset all sparsity counters (global, per-phase, per-layer, unit)."""
        self.total_zero_count = 0
        self.total_element_count = 0
        self.total_bit_count = 0
        self.total_0bit_count = 0
        self.total_sparsebit_count = 0
        self.total_amplitude_zero_bits_total = 0
        self.total_amplitude_bit_count = 0

        self.phase_sparsity = {
            phase: self._new_sparsity_counter()
            for phase in self._PHASES
        }
        self.unit_sparsity = {phase: {} for phase in self._PHASES}
        self.per_layer_sparsity = {}
        self.per_role_sparsity = {}
        self.flow_step_sparsity = {}
        self.tensor_role_calls = {}
        self.module_last_dims = {}
        self.matmul_workload = {}
        self._pending_matmul_k = {}
        self.outlier_sidepath = {}
        self.outlier_partition = {}
        self.per_role_unit_sparsity = {}
        self.per_layer_weight_sparsity = {}
        self.fp_code_audit = {}
        self.current_phase = "full_forward"

    def _accumulate_phase_sparsity(
        self,
        phase: str,
        total_num: int,
        abs_less_th: int,
        total_bits: int,
        zero_bits_total: int,
        sparse_bits_total: int,
        amplitude_zero_bits_total: int,
    ):
        # Canonical label is "denoise" (manual §5); the legacy counters use
        # "decode" — normalize so both labels hit the same bucket.
        if phase == "denoise":
            phase = "decode"
        if phase not in self.phase_sparsity:
            self.phase_sparsity[phase] = self._new_sparsity_counter()

        counter = self.phase_sparsity[phase]
        counter["total_zero_count"] += abs_less_th
        counter["total_element_count"] += total_num
        counter["total_bit_count"] += total_bits
        counter["total_0bit_count"] += zero_bits_total
        counter["total_sparsebit_count"] += sparse_bits_total
        counter["total_amplitude_zero_bits_total"] += amplitude_zero_bits_total
        counter["total_amplitude_bit_count"] += total_bits

    # -------------------------------------------------------------------------
    # Registration & collection
    # -------------------------------------------------------------------------

    def register_layer(self, layer_name: str, layer_idx: int):
        key = f"{layer_name}_{layer_idx}"
        if key not in self.stats:
            self.stats[key] = QuantStatistics(layer_name, layer_idx)

    def collect_linear_stats(
        self,
        layer_name: str,
        layer_idx: int,
        w_scale: float,
        a_scale: float,
        o_scale: float,
    ):
        """Collect statistics for linear layer (called by QuantizedLinear)."""
        key = f"{layer_name}_{layer_idx}"
        if key not in self.stats:
            self.register_layer(layer_name, layer_idx)
        self.stats[key].collect_linear_stats(w_scale, a_scale, o_scale)

    def collect_matmul_stats(
        self,
        layer_name: str,
        layer_idx: int,
        A_scale: float,
        B_scale: float,
        O_scale: float,
    ):
        """Collect statistics for matmul (called by QuantizedMatMul)."""
        key = f"{layer_name}_{layer_idx}"
        if key not in self.stats:
            self.register_layer(layer_name, layer_idx)
        self.stats[key].collect_matmul_stats(A_scale, B_scale, O_scale)

    def collect_quant_activation(
        self,
        layer_name: str,
        layer_idx: int,
        *args,
        **kwargs,
    ):
        """
        Collect quantized-activation sparsity statistics (opt-in).

        Signature is intentionally elastic to accept both call sites:

        - QuantizedLinear.quant_forward (7 positional args):
              (layer_name, layer_idx, x_code, x_fp16, a_spec,
               digit_size, parallelism, in_features, out_features)
        - QuantizedMatMul.quant_forward (9 positional args):
              (layer_name, layer_idx, A_sim, A, B_sim,
               B_spec, A_spec, digit_size, parallelism,
               in_features, out_features)

        Always records the call for auditing ONLY. Structured sparsity is
        collected exactly once by _collect_linear_runtime /
        _collect_matmul_runtime via collect_quant_tensor() (Phase H: keeps
        activation / A / B from being double-counted, which would corrupt
        the native sparsity correction whose outlier partition is recorded
        once per forward).
        """
        key = f"{layer_name}_{layer_idx}"
        self.quant_activation_calls[key] = (
            self.quant_activation_calls.get(key, 0) + 1
        )
        return

    @staticmethod
    def _layer_idx_from_module_id(module_id: str) -> int:
        """
        Parse the layer index from a module_id.

        Forms:
            vlm.layers.3.mlp.down_proj        -> 3
            expert.layer.7.qk                  -> 7
            model.model.action_head.proj       -> -1 (no layer index)
        """
        parts = str(module_id).split(".")
        for i, p in enumerate(parts):
            if p in ("layers", "layer") and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    break
        return -1

    @staticmethod
    def _parse_module_id(module_id: str) -> Tuple[str, str, int]:
        """
        Parse (component, operator, layer_idx) from a physical module_id.

        Forms:
            vlm.layers.3.mlp.down_proj       -> ("vlm", "down_proj", 3)
            expert.layers.7.self_attn.q_proj -> ("expert", "q_proj", 7)
            expert.layer.5.qk                 -> ("expert", "qk", 5)
            model.model.action_head.proj      -> ("head", "proj", -1)
        """
        s = str(module_id)
        component = s.split(".")[0] if s else "unknown"
        operator = s.split(".")[-1] if s else "unknown"
        layer_idx = QuantStatManager._layer_idx_from_module_id(s)
        return component, operator, layer_idx

    def collect_quant_tensor(
        self,
        *,
        module_id: str,
        tensor_role: str,
        tensor_code: torch.Tensor,
        spec: QuantSpec,
        attention_kind: Optional[str] = None,
        layer_idx: Optional[int] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Structured collector for ONE quantized tensor at a physical module
        (research manual §19). Keyed by module_id + runtime labels.

        Args:
            module_id:    physical operator identity, e.g.
                          "vlm.layers.3.mlp.down_proj" /
                          "expert.layer.7.qk" (NEVER scale_group).
            tensor_role:  "activation" | "output" (Linear) /
                          "A" | "B" | "O" (MatMul).
            tensor_code:  the quantized code tensor (post quant_awo).
            spec:         QuantSpec of this operand.
            attention_kind: "self" | "cross" (attention matmuls; None ok).
            runtime_context: snapshot from get_runtime_context(); when
                          omitted the current context is read.
            metadata:    free-form extras (e.g. operand_origin).
        """
        ctx = runtime_context or get_runtime_context()

        self.quant_activation_calls[module_id] = (
            self.quant_activation_calls.get(module_id, 0) + 1
        )

        if not self.sparsity_enabled:
            return
        if tensor_code is None or spec is None:
            return
        if spec.kind == "none" or not getattr(spec, "enabled", True):
            return

        # Record last-observed dims for the workload exporter.
        if tensor_code.dim() >= 1:
            self.module_last_dims[module_id] = (
                int(tensor_code.shape[-1]),  # K (inner dim)
                int(tensor_code.numel() // max(tensor_code.shape[-1], 1)),
            )

        # Runtime context drives the phase; fall back to the manual
        # set_phase() tag when no hook has ever set the context (keeps
        # legacy runners working).
        phase = ctx.get("phase", "unknown")
        if phase == "unknown" and self.current_phase != "full_forward":
            phase = self.current_phase
        flow_step = int(ctx.get("flow_step", -1))
        resolved_attention_kind = attention_kind or ctx.get(
            "attention_kind", "unknown"
        )

        # Exact physical MatMul workload accounting. The structured MatMul
        # collector emits A, then B, then O for each operator call.
        workload_key = (
            module_id,
            phase,
            flow_step,
            resolved_attention_kind or "unknown",
        )
        if tensor_role == "A" and tensor_code.dim() >= 1:
            self._pending_matmul_k[workload_key] = int(tensor_code.shape[-1])
        elif tensor_role == "O" and tensor_code.dim() >= 1:
            k_inner = self._pending_matmul_k.get(workload_key)
            if k_inner is not None:
                n_out = int(tensor_code.shape[-1])
                out_elems = int(tensor_code.numel())
                m_flat = out_elems // max(n_out, 1)
                workload = self.matmul_workload.setdefault(
                    workload_key,
                    {
                        "calls": 0,
                        "macs_total": 0,
                        "last_M": None,
                        "last_K": None,
                        "last_N": None,
                    },
                )
                workload["calls"] += 1
                workload["macs_total"] += out_elems * k_inner
                workload["last_M"] = m_flat
                workload["last_K"] = k_inner
                workload["last_N"] = n_out

        self._collect_one_tensor_sparsity(
            module_id,
            layer_idx,
            tensor_code,
            spec,
            tensor_role=tensor_role,
            phase=phase,
            flow_step=flow_step,
            attention_kind=resolved_attention_kind,
        )

        # H1-Audit: independent raw FP8-code inspection (does NOT touch
        # forward numerics, scale, mask, or RNG).
        self._collect_fp_code_audit(
            module_id=module_id,
            tensor_role=tensor_role,
            tensor_code=tensor_code,
            spec=spec,
            phase=phase,
            flow_step=flow_step,
            attention_kind=resolved_attention_kind,
        )

    # -------------------------------------------------------------------------
    # Sparsity: collection internals
    # -------------------------------------------------------------------------

    def _collect_fp_code_audit(
        self,
        *,
        module_id: str,
        tensor_role: str,
        tensor_code: torch.Tensor,
        spec: QuantSpec,
        phase: str,
        flow_step: int,
        attention_kind: str,
    ):
        """
        Debug-only independent E4M3 raw-code audit (H1-Audit).

        IMPORTANT:
        - Does NOT reuse _extract_sm_from_raw().
        - Does NOT modify tensor_code.
        - Does NOT use RNG.
        - Deterministically subsamples large tensors.
        """
        if not self.fp_code_audit_enabled:
            return
        if tensor_role not in self.fp_code_audit_roles:
            return
        if tensor_code is None or spec is None:
            return
        if spec.kind != "fp":
            return

        fmt = (spec.fmt or "").lower().strip()
        if fmt not in {"e4m3", "e4m3fn", "fp8_e4m3", "fp8_e4m3fn"}:
            return

        x = tensor_code.detach().reshape(-1)
        if x.numel() == 0:
            return

        # Deterministic strided subsampling (no RNG).
        max_n = self.fp_code_audit_max_elements_per_call
        if max_n and max_n > 0 and x.numel() > max_n:
            stride = (x.numel() + max_n - 1) // max_n
            x = x[::stride][:max_n]

        q8 = x.to(torch.float8_e4m3fn)
        raw = q8.view(torch.uint8).reshape(-1).to(torch.int64)
        q32 = q8.to(torch.float32)

        # Independent E4M3 parsing (bit7 sign, bit6..3 exp, bit2..0 mant).
        sign = (raw >> 7) & 0x1
        exp = (raw >> 3) & 0xF
        mant = raw & 0x7

        nan_mask = torch.isnan(q32)
        zero_mask = (exp == 0) & (mant == 0)
        subnormal_mask = (exp == 0) & (mant != 0)
        valid_mask = ~nan_mask

        # Magnitude significand (normal 1MMM, subnormal 0MMM, zero 0000).
        hidden = (exp != 0).to(torch.int64) << 3
        sig = mant | hidden
        sig = torch.where(zero_mask, torch.zeros_like(sig), sig)

        sig_valid = sig[valid_mask]
        nonzero_valid_mask = valid_mask & (~zero_mask)
        sig_nonzero = sig[nonzero_valid_mask]

        mant_valid = mant[valid_mask]
        mant_nonzero = mant[nonzero_valid_mask]
        exp_valid = exp[valid_mask]

        mant_hist = torch.bincount(mant_valid, minlength=8)
        mant_nonzero_hist = torch.bincount(mant_nonzero, minlength=8)
        exp_hist = torch.bincount(exp_valid, minlength=16)
        sig_hist = torch.bincount(sig_valid, minlength=16)

        sig_zero_bits = 0
        for bit_idx in range(4):
            bit = (sig_valid >> bit_idx) & 1
            sig_zero_bits += int((1 - bit).sum().item())

        sig_nonzero_zero_bits = 0
        for bit_idx in range(4):
            bit = (sig_nonzero >> bit_idx) & 1
            sig_nonzero_zero_bits += int((1 - bit).sum().item())

        max_val = float(torch.finfo(torch.float8_e4m3fn).max)
        saturation_count = int(
            (torch.abs(q32[valid_mask]) >= max_val).sum().item()
        )

        key = (
            module_id,
            phase,
            int(flow_step),
            tensor_role,
            attention_kind or "unknown",
        )

        entry = self.fp_code_audit.setdefault(
            key,
            {
                "module_id": module_id,
                "phase": phase,
                "flow_step": int(flow_step),
                "tensor_role": tensor_role,
                "attention_kind": attention_kind or "unknown",
                "calls": 0,
                "sampled_elements": 0,
                "valid_elements": 0,
                "nonzero_elements": 0,
                "zero_code_count": 0,
                "subnormal_count": 0,
                "nan_count": 0,
                "negative_count": 0,
                "saturation_count": 0,
                "sig_zero_bits": 0,
                "sig_total_bits": 0,
                "sig_nonzero_zero_bits": 0,
                "sig_nonzero_total_bits": 0,
                "mant_hist": [0] * 8,
                "mant_nonzero_hist": [0] * 8,
                "exp_hist": [0] * 16,
                "sig_hist": [0] * 16,
            },
        )

        n_sampled = int(raw.numel())
        n_valid = int(valid_mask.sum().item())
        n_nonzero = int(nonzero_valid_mask.sum().item())

        entry["calls"] += 1
        entry["sampled_elements"] += n_sampled
        entry["valid_elements"] += n_valid
        entry["nonzero_elements"] += n_nonzero
        entry["zero_code_count"] += int(zero_mask.sum().item())
        entry["subnormal_count"] += int(subnormal_mask.sum().item())
        entry["nan_count"] += int(nan_mask.sum().item())
        entry["negative_count"] += int((sign[valid_mask] != 0).sum().item())
        entry["saturation_count"] += saturation_count
        entry["sig_zero_bits"] += sig_zero_bits
        entry["sig_total_bits"] += 4 * n_valid
        entry["sig_nonzero_zero_bits"] += sig_nonzero_zero_bits
        entry["sig_nonzero_total_bits"] += 4 * n_nonzero

        for i in range(8):
            entry["mant_hist"][i] += int(mant_hist[i].item())
            entry["mant_nonzero_hist"][i] += int(
                mant_nonzero_hist[i].item()
            )
        for i in range(16):
            entry["exp_hist"][i] += int(exp_hist[i].item())
            entry["sig_hist"][i] += int(sig_hist[i].item())

    def _collect_one_tensor_sparsity(
        self,
        layer_name: str,
        layer_idx,
        activation: torch.Tensor,
        spec: QuantSpec,
        tensor_role: str = "activation",
        phase: Optional[str] = None,
        flow_step: int = -1,
        attention_kind: str = "unknown",
    ):
        """Compute and accumulate element/bit-level + unit sparsity.

        `layer_idx` may be None when the caller only has a module_id
        (structured path); it is parsed from the module_id when possible.
        `tensor_role`/`flow_step`/`attention_kind` extend the record key
        so Linear input/output and MatMul A/B/O stay separate, and denoise
        statistics can be split per flow step (manual §6/§8/§21).
        """
        phase = phase or self.current_phase
        if layer_idx is None:
            layer_idx = self._layer_idx_from_module_id(layer_name)
        layer_key = f"{layer_name}_{layer_idx}"

        # Flow-step split bookkeeping (denoise only; manual §21-§22).
        if phase == "denoise":
            fs_counter = self.flow_step_sparsity.setdefault(flow_step, {})
            fs_counter[layer_key] = fs_counter.get(layer_key, 0) + 1

        role_counts = self.tensor_role_calls.setdefault(layer_key, {})
        role_counts[tensor_role] = role_counts.get(tensor_role, 0) + 1

        self.collected_layer_names_by_phase.setdefault(phase, set()).add(layer_key)
        call_counts = self.collected_layer_call_count_by_phase.setdefault(
            phase, {}
        )
        call_counts[layer_key] = call_counts.get(layer_key, 0) + 1

        if spec.kind == "int":
            sparse = self.compute_sparse_stats(
                spec.bits,
                activation,
                0,
                chunk_size=self.sparse_stat_chunk_size,
            )
        elif spec.kind in {"fp", "bf"}:
            sparse = self.compute_sparse_stats_fp(
                spec.fmt,
                activation,
                0.0,
                chunk_size=self.sparse_stat_chunk_size,
            )
        else:
            return

        (
            total_num,
            abs_less_th,
            total_bits,
            zero_bits_total,
            sparse_bits_total,
            amplitude_zero_bits_total,
        ) = sparse

        # Global counters.
        self.total_zero_count += abs_less_th
        self.total_element_count += total_num
        self.total_bit_count += total_bits
        self.total_0bit_count += zero_bits_total
        self.total_sparsebit_count += sparse_bits_total
        self.total_amplitude_zero_bits_total += amplitude_zero_bits_total
        self.total_amplitude_bit_count += total_bits

        # Per-phase counters.
        self._accumulate_phase_sparsity(
            phase,
            total_num,
            abs_less_th,
            total_bits,
            zero_bits_total,
            sparse_bits_total,
            amplitude_zero_bits_total,
        )

        # Per-layer records (legacy aggregate).
        entry = self.per_layer_sparsity.setdefault(
            layer_key,
            {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
            },
        )
        self._accumulate_sparsity_entry(entry, total_num, abs_less_th, total_bits,
                                        zero_bits_total, sparse_bits_total,
                                        amplitude_zero_bits_total)

        # Structured per-role / per-phase / per-flow-step record.
        # Phase H (manual §4): flow_step and attention_kind are first-class
        # key dimensions so denoise steps 0..9 stay separate and attention
        # self/cross matmuls are not merged.
        component, operator, _ = self._parse_module_id(layer_name)
        role_key = (
            layer_name,
            phase,
            int(flow_step),
            tensor_role,
            attention_kind or "unknown",
        )
        role_entry = self.per_role_sparsity.setdefault(
            role_key,
            {
                "module_id": layer_name,
                "component": component,
                "operator": operator,
                "layer_idx": layer_idx,
                "phase": phase,
                "flow_step": int(flow_step),
                "tensor_role": tensor_role,
                "attention_kind": attention_kind or "unknown",
            },
        )
        self._accumulate_sparsity_entry(role_entry, total_num, abs_less_th,
                                        total_bits, zero_bits_total,
                                        sparse_bits_total,
                                        amplitude_zero_bits_total)
        role_entry["calls"] = role_entry.get("calls", 0) + 1

        # Unit/block sparsity.
        if self.enable_unit_sparsity:
            self.collect_unit_sparsity_structured(
                module_id=layer_name,
                layer_idx=layer_idx,
                tensor=activation,
                spec=spec,
                phase=phase,
                flow_step=int(flow_step),
                tensor_role=tensor_role,
                attention_kind=attention_kind or "unknown",
            )

    @staticmethod
    def _accumulate_sparsity_entry(
        entry: Dict[str, Any],
        total_num: int,
        abs_less_th: int,
        total_bits: int,
        zero_bits_total: int,
        sparse_bits_total: int,
        amplitude_zero_bits_total: int,
    ):
        """Accumulate counters into a record and refresh derived ratios."""
        entry["total_elements"] = entry.get("total_elements", 0) + int(total_num)
        entry["zero_elements"] = entry.get("zero_elements", 0) + int(abs_less_th)
        entry["total_bits"] = entry.get("total_bits", 0) + int(total_bits)
        entry["zero_bits"] = entry.get("zero_bits", 0) + int(zero_bits_total)
        entry["sparse_bits"] = entry.get("sparse_bits", 0) + int(sparse_bits_total)
        entry["amplitude_zero_bits"] = (
            entry.get("amplitude_zero_bits", 0) + int(amplitude_zero_bits_total)
        )
        entry["zero_rate"] = (
            entry["zero_elements"] / entry["total_elements"]
            if entry["total_elements"] > 0
            else 0.0
        )
        entry["sparse_bit_rate"] = (
            entry["sparse_bits"] / entry["total_bits"]
            if entry["total_bits"] > 0
            else 0.0
        )
        entry["amplitude_zero_bit_rate"] = (
            entry["amplitude_zero_bits"] / entry["total_bits"]
            if entry["total_bits"] > 0
            else 0.0
        )
        # Phase H (manual §7): 1/(1-S) is an ideal upper bound, NOT a
        # measured hardware speedup. Keep the legacy name for back-compat.
        upper_bound = (
            1 / (1 - entry["sparse_bit_rate"])
            if entry["sparse_bit_rate"] < 1
            else float("inf")
        )
        entry["ideal_sparse_upper_bound"] = upper_bound
        entry["ideal_speed_up"] = upper_bound
        entry["ideal_speed_up_legacy"] = upper_bound
        entry["1_bits"] = entry["total_bits"] - entry["amplitude_zero_bits"]

    def collect_weight_sparsity(
        self,
        layer_name: str,
        layer_idx: int,
        weight: torch.Tensor,
        spec: QuantSpec,
    ):
        """
        Collect sparsity statistics for a (static) quantized weight tensor.

        Weights do not change between forwards, so this is meant to be
        called ONCE per layer (e.g. right after scales are loaded /
        calibration finishes), not from the forward path. Results go to
        a dedicated per_layer_weight_sparsity record; global / phase
        counters are NOT touched (they track activations only).

        `weight` may be either the raw FP weight (quantized here via
        quant_awo with the layer's interval) or an already-quantized
        code tensor (pass interval=None).
        """
        if weight is None or spec is None:
            return
        if spec.kind == "none" or not getattr(spec, "enabled", True):
            return

        layer_key = f"{layer_name}_{layer_idx}"

        if spec.kind == "int":
            sparse = self.compute_sparse_stats(
                spec.bits, weight, 0,
                chunk_size=self.sparse_stat_chunk_size,
            )
        elif spec.kind in {"fp", "bf"}:
            sparse = self.compute_sparse_stats_fp(
                spec.fmt, weight, 0.0,
                chunk_size=self.sparse_stat_chunk_size,
            )
        else:
            return

        (
            total_num,
            abs_less_th,
            total_bits,
            zero_bits_total,
            sparse_bits_total,
            amplitude_zero_bits_total,
        ) = sparse

        component, operator, _ = self._parse_module_id(layer_name)
        entry = self.per_layer_weight_sparsity.setdefault(
            layer_key,
            {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "module_id": layer_name,
                "component": component,
                "operator": operator,
                "weight_spec": spec.name(),
            },
        )
        entry["total_elements"] = int(total_num)
        entry["zero_elements"] = int(abs_less_th)
        entry["total_bits"] = int(total_bits)
        entry["zero_bits"] = int(zero_bits_total)
        entry["sparse_bits"] = int(sparse_bits_total)
        entry["amplitude_zero_bits"] = int(amplitude_zero_bits_total)
        entry["zero_rate"] = (
            entry["zero_elements"] / entry["total_elements"]
            if entry["total_elements"] > 0 else 0.0
        )
        entry["sparse_bit_rate"] = (
            entry["sparse_bits"] / entry["total_bits"]
            if entry["total_bits"] > 0 else 0.0
        )
        entry["amplitude_zero_bit_rate"] = (
            entry["amplitude_zero_bits"] / entry["total_bits"]
            if entry["total_bits"] > 0 else 0.0
        )
        upper_bound = (
            1 / (1 - entry["sparse_bit_rate"])
            if entry["sparse_bit_rate"] < 1 else float("inf")
        )
        entry["ideal_sparse_upper_bound"] = upper_bound
        entry["ideal_speed_up"] = upper_bound
        entry["1_bits"] = entry["total_bits"] - entry["amplitude_zero_bits"]

    def export_per_layer_weight_sparsity_csv(
        self,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
    ):
        """Export per-layer WEIGHT bit sparsity records to CSV."""
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            weight_writer = csv.writer(f)
            weight_writer.writerow([
                "config", "model_path", "module_id", "component",
                "operator", "layer_idx", "weight_spec", "stat_semantics",
                "total_elements", "zero_elements", "zero_rate",
                "total_bits", "sparse_bits", "sparse_bit_rate",
                "amplitude_zero_bits", "amplitude_zero_bit_rate",
                "ideal_sparse_upper_bound", "1_bits",
            ])

            for layer_key in sorted(self.per_layer_weight_sparsity.keys()):
                entry = self.per_layer_weight_sparsity[layer_key]
                layer_idx = entry.get("layer_idx", -1)

                weight_writer.writerow([
                    config_name, model_path,
                    entry.get("module_id", entry.get("layer_name", layer_key)),
                    entry.get("component", ""),
                    entry.get("operator", ""),
                    layer_idx,
                    entry.get("weight_spec", ""),
                    "full_weight_quantized_no_dynamic_outlier_mask",
                    entry.get("total_elements", 0),
                    entry.get("zero_elements", 0),
                    entry.get("zero_rate", 0.0),
                    entry.get("total_bits", 0),
                    entry.get("sparse_bits", 0),
                    entry.get("sparse_bit_rate", 0.0),
                    entry.get("amplitude_zero_bits", 0),
                    entry.get("amplitude_zero_bit_rate", 0.0),
                    entry.get("ideal_sparse_upper_bound", 0.0),
                    entry.get("1_bits", 0),
                ])

    def print_weight_sparsity_summary(self):
        """Print aggregate weight sparsity across collected layers."""
        print("\n" + "-" * 80)
        print("WEIGHT SPARSITY SUMMARY (per layer, static)")
        print("-" * 80)

        if not self.per_layer_weight_sparsity:
            print("No weight sparsity statistics collected.")
            return

        tot_bits = sum(
            e["total_bits"] for e in self.per_layer_weight_sparsity.values()
        )
        tot_sparse = sum(
            e["sparse_bits"] for e in self.per_layer_weight_sparsity.values()
        )
        tot_amp = sum(
            e["amplitude_zero_bits"]
            for e in self.per_layer_weight_sparsity.values()
        )
        tot_elem = sum(
            e["total_elements"]
            for e in self.per_layer_weight_sparsity.values()
        )
        tot_zero = sum(
            e["zero_elements"]
            for e in self.per_layer_weight_sparsity.values()
        )

        print(f"  layers: {len(self.per_layer_weight_sparsity)}")
        if tot_elem > 0:
            print(f"  零值元素比例: {tot_zero / tot_elem:.4%}")
        if tot_bits > 0:
            print(f"  稀疏比特比例: {tot_sparse / tot_bits:.4%}")
            print(f"  SM编码比例:   {tot_amp / tot_bits:.4%}")

    def collect_model_weight_sparsity(self, model) -> int:
        """
        One-shot weight sparsity collection for a whole wrapped model.

        Iterates all QuantizedLinear / QuantizedMatMul modules, quantizes
        each weight once with the layer's interval + spec (via
        quant_awo), and records per-layer bit statistics into
        per_layer_weight_sparsity. Call AFTER calibration / scale
        loading (intervals must be set).

        Returns the number of layers collected.

        NOTE: import inside the function to avoid a circular import
        (quant_linear / quant_matmul import stat_manager types only
        via duck typing, but keep it safe anyway).
        """
        from .quant_spec import quant_awo

        n = 0
        for module in model.modules():
            cls_name = type(module).__name__
            if cls_name == "QuantizedLinear":
                if module.w_interval is None:
                    continue
                w_code = quant_awo(
                    module.weight,
                    module.w_interval,
                    module.w_spec,
                    out_dtype=torch.float32,
                    chunk_size=self.sparse_stat_chunk_size,
                )
                module_id = getattr(module, "module_id", "") or (
                    f"{module.layer_name}_{module.layer_idx}"
                )
                self.collect_weight_sparsity(
                    module_id, module.layer_idx, w_code, module.w_spec
                )
                n += 1
            elif cls_name == "QuantizedMatMul":
                # NOTE (manual §25): QuantizedMatMul has NO static weight —
                # B (K/V) arrives at runtime from attention projections /
                # KV cache. Only a scale exists (B_interval). Weight-side
                # sparsity for matmul B must therefore come from runtime
                # collection (tensor_role="B"), not this static path.
                continue
        return n

    def _spec_bitwidth(self, spec: QuantSpec) -> int:
        """
        Statistics bit-width of a quantized code tensor (matches the
        numerator/denominator width used by compute_sparse_stats / fp):
        INT -> spec.bits; FP/bf -> mantissa_int width (mant_bits + 1).
        """
        if spec.kind == "int":
            return int(spec.bits or 0)
        if spec.kind in {"fp", "bf"}:
            info = self._get_fp_format(spec.fmt)
            return info["mant_bits"] + 1
        return 0

    def collect_outlier_partition(
        self,
        *,
        module_id: str,
        tensor_role: str,
        total_elements: int,
        protected_elements: int,
        spec: QuantSpec,
        layer_idx: Optional[int] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        attention_kind: Optional[str] = None,
    ):
        """
        Phase H (manual §3): record the outlier FP side-path partition for
        one quantized tensor, so the "native quant path" sparsity can exclude
        positions that the outlier forward artificially zeroes in the normal
        code path.

        Key matches the main structured record. Accumulates numerator /
        denominator counters (never averages per-call ratios).
        """
        if not self.sparsity_enabled:
            return

        ctx = runtime_context or get_runtime_context()
        phase = ctx.get("phase", "unknown")
        # Mirror collect_quant_tensor's legacy fallback so the partition key
        # stays identical to the main structured record key.
        if phase == "unknown" and self.current_phase != "full_forward":
            phase = self.current_phase
        flow_step = int(ctx.get("flow_step", -1))
        attn = attention_kind or ctx.get("attention_kind", "unknown")

        total_elements = int(total_elements)
        protected_elements = int(protected_elements)
        bitwidth = self._spec_bitwidth(spec)

        key = (module_id, phase, flow_step, tensor_role, attn or "unknown")
        entry = self.outlier_partition.setdefault(
            key,
            {
                "module_id": module_id,
                "phase": phase,
                "flow_step": flow_step,
                "tensor_role": tensor_role,
                "attention_kind": attn or "unknown",
                "calls": 0,
                "total_elements": 0,
                "protected_elements": 0,
                "protected_bits": 0,
            },
        )
        entry["calls"] += 1
        entry["total_elements"] += total_elements
        entry["protected_elements"] += protected_elements
        entry["protected_bits"] += protected_elements * bitwidth

    # -------------------------------------------------------------------------
    # Sparsity: INT / FP bit statistics (chunked)
    # -------------------------------------------------------------------------

    def compute_sparse_stats(
        self,
        n: int,
        tensor: torch.Tensor,
        th: float = 0,
        chunk_size: int = 1_048_576,
    ) -> Tuple[int, int, int, int, int, int]:
        """
        Chunked bit statistics for INT quantized activations.

        Returns:
            (total_num, abs_less_th, total_bits,
             zero_bits_total, sparse_bits_total,
             amplitude_zero_bits_total)

        amplitude_zero_bits_total counts 0-bits in the sign-magnitude
        (original-code) representation; sparse_bits_total counts 0-bits
        with two's-complement sign semantics (positive: 0-bit saves,
        negative: 1-bit saves).
        """
        total_num = tensor.numel()
        total_bits = total_num * n

        if total_num == 0:
            return total_num, 0, total_bits, 0, 0, 0

        flat = tensor.detach().reshape(-1)

        abs_less_th = 0
        zero_bits_total = 0
        sparse_bits_total = 0
        amplitude_zero_bits_total = 0

        mask = (1 << n) - 1
        amp_mask = (1 << (n - 1)) - 1
        sign_bit = 1 << (n - 1)

        for start in range(0, total_num, chunk_size):
            end = min(start + chunk_size, total_num)
            chunk = flat[start:end]

            abs_less_th += int((torch.abs(chunk) <= th).sum().item())

            int_tensor = chunk.to(torch.int32)

            # Two's-complement low-n-bit pattern.
            us_tensor = int_tensor & mask

            # Positive mask (including 0).
            pos_mask = int_tensor >= 0

            # Sign-magnitude / original code.
            abs_val = torch.abs(int_tensor)
            neg_orig = sign_bit | (abs_val & amp_mask)
            orig_tensor = torch.where(pos_mask, us_tensor, neg_orig)

            for i in range(n):
                bit = (us_tensor >> i) & 1

                zero_bits_total += int((1 - bit).sum().item())

                sparse_contrib = torch.where(pos_mask, 1 - bit, bit)
                sparse_bits_total += int(sparse_contrib.sum().item())

                orig_bit = (orig_tensor >> i) & 1
                amplitude_zero_bits_total += int((1 - orig_bit).sum().item())

            del chunk, int_tensor, us_tensor, pos_mask, abs_val, neg_orig, orig_tensor

        return (
            total_num,
            abs_less_th,
            total_bits,
            zero_bits_total,
            sparse_bits_total,
            amplitude_zero_bits_total,
        )

    def _get_fp_format(self, fmt: str) -> Dict[str, int]:
        """
        FP format info: total_bits / exp_bits / mant_bits
        (mantissa fraction bits, hidden leading 1 NOT included).
        """
        fmt = str(fmt).lower().strip()

        if fmt in {"e4m3", "e4m3fn", "fp8_e4m3", "fp8_e4m3fn"}:
            return {"total_bits": 8, "exp_bits": 4, "mant_bits": 3}

        if fmt in {"e2m1", "fp4", "fp4_e2m1"}:
            return {"total_bits": 4, "exp_bits": 2, "mant_bits": 1}

        if fmt in {"e5m2", "fp8_e5m2"}:
            return {"total_bits": 8, "exp_bits": 5, "mant_bits": 2}

        if fmt in {"e5m10", "fp16", "float16"}:
            return {"total_bits": 16, "exp_bits": 5, "mant_bits": 10}

        if fmt in {"bf16", "bfloat16"}:
            return {"total_bits": 16, "exp_bits": 8, "mant_bits": 7}

        if fmt in {"fp32", "float32"}:
            return {"total_bits": 32, "exp_bits": 8, "mant_bits": 23}

        raise ValueError(f"Unsupported FP format: {fmt}")

    @staticmethod
    def _unpack_sm_exp(
        raw: torch.Tensor,
        sign_shift: int,
        exp_bits: int,
        mant_bits: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpack sign / exp / mantissa from a raw int bit pattern.

        Returns:
            sm:       mantissa(mant_bits)  (sign NOT included — the
                      significand metric is sign-agnostic)
            exp:      biased exponent
            normal:   exp != 0 (hidden leading 1 present)

        NOTE: `sign_shift` is accepted for caller compatibility but the
        sign bit is deliberately NOT folded into the significand; +0/-0
        must both map to the all-zero code, and subnormals (exp==0) must
        NOT receive the hidden leading 1.
        """
        raw = raw.to(torch.int64)

        mant = raw & ((1 << mant_bits) - 1)
        exp = (raw >> mant_bits) & ((1 << exp_bits) - 1)

        sm = mant
        normal = exp != 0

        return sm, exp, normal

    def _extract_sm_from_raw(
        self,
        raw_int: torch.Tensor,
        fmt: str,
    ) -> Tuple[torch.Tensor, int]:
        """
        Extract sign+mantissa (with hidden leading 1) from a raw bit
        pattern viewed as int, per format e5m10 / e4m3 / e2m1.

        Returns:
            sm_codes: same shape, low `width` bits valid
            width:    number of valid sm bits
        """
        fmt = (fmt or "").lower().strip()

        if fmt == "e5m10":
            sm, _exp, normal = self._unpack_sm_exp(
                raw_int, sign_shift=15, exp_bits=5, mant_bits=10,
            )
            width = 11
            mant_bits = 10
        elif fmt == "e4m3":
            sm, _exp, normal = self._unpack_sm_exp(
                raw_int, sign_shift=7, exp_bits=4, mant_bits=3,
            )
            width = 4
            mant_bits = 3
        elif fmt == "e2m1":
            sm, _exp, normal = self._unpack_sm_exp(
                raw_int, sign_shift=3, exp_bits=2, mant_bits=1,
            )
            width = 2
            mant_bits = 1
        else:
            raise ValueError(f"Unsupported fmt in _extract_sm_from_raw: {fmt}")

        # Add hidden leading 1 ONLY for normal numbers (exp != 0).
        # Subnormals (exp==0, mant!=0) and zeros stay without hidden bit.
        hidden = normal.to(torch.int64) << mant_bits
        sm_full = sm | hidden

        return sm_full, width

    def _fp_tensor_to_mantissa_fixed_chunk(
        self,
        tensor_chunk: torch.Tensor,
        fmt: str,
    ) -> torch.Tensor:
        """
        Convert an FP chunk to "mantissa fixed-point integers"
        (significand incl. hidden leading 1):

            E4M3:  1.xxx     -> 4-bit mantissa_int
            E5M10: 1.xxxxxxx -> 11-bit mantissa_int
            BF16:  1.xxxxxxx -> 8-bit mantissa_int
            E2M1:  1.x       -> 2-bit mantissa_int

        Zeros map to all-zero codes (no hidden leading 1).
        """
        info = self._get_fp_format(fmt)
        exp_bits = info["exp_bits"]
        mant_bits = info["mant_bits"]

        stat_total_bits = mant_bits + 1
        if stat_total_bits >= 63:
            raise ValueError(
                f"FP mantissa statistic width n={stat_total_bits} is too "
                f"large for int64 bit operations."
            )

        x = tensor_chunk.detach().to(torch.float32)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        abs_x = torch.abs(x)
        nonzero_mask = abs_x > 0

        safe_abs_x = torch.where(
            nonzero_mask,
            abs_x,
            torch.ones_like(abs_x),
        )

        # Exponent used only to extract significand = abs_x / 2^exp;
        # it does not participate in the fixed-point shift below.
        exp_unbiased = torch.floor(torch.log2(safe_abs_x)).to(torch.int64)

        exp_bias = (1 << (exp_bits - 1)) - 1
        min_normal_exp = 1 - exp_bias
        max_normal_exp = ((1 << exp_bits) - 2) - exp_bias

        exp_unbiased = torch.clamp(
            exp_unbiased,
            min=min_normal_exp,
            max=max_normal_exp,
        )

        base = torch.tensor(2.0, device=x.device, dtype=torch.float32)

        significand = safe_abs_x / torch.pow(
            base,
            exp_unbiased.to(torch.float32),
        )

        # mantissa_int includes the hidden leading 1:
        #   E4M3: 1.010 * 2^3 -> 1010
        mant_scale = 1 << mant_bits
        mantissa_int = torch.round(significand * mant_scale).to(torch.int64)

        # Renormalize 1.111... -> 10.000... carry after rounding.
        overflow = mantissa_int >= (1 << (mant_bits + 1))
        mantissa_int = torch.where(
            overflow,
            mantissa_int >> 1,
            mantissa_int,
        )

        # Zero has no hidden leading 1 -> force all-zero code.
        mantissa_int = torch.where(
            nonzero_mask,
            mantissa_int,
            torch.zeros_like(mantissa_int),
        )

        mask = (1 << stat_total_bits) - 1
        return mantissa_int & mask

    def compute_sparse_stats_fp(
        self,
        fmt: str,
        tensor: torch.Tensor,
        th: float = 0.0,
        chunk_size: int = 1_048_576,
    ) -> Tuple[int, int, int, int, int, int]:
        """
        Chunked sparse/zero-bit statistics for FP inputs.

        Statistics width = mantissa_int width (hidden leading 1
        included): E4M3 -> 4 bits, E5M10 -> 11 bits, BF16 -> 8 bits,
        E2M1 -> 2 bits. Raw-bit-pattern paths are used for
        e4m3/e2m1/e5m10 so the codes match the actual stored encoding.
        """
        info = self._get_fp_format(fmt)
        mant_bits = info["mant_bits"]

        total_num = tensor.numel()
        if total_num == 0:
            return total_num, 0, 0, 0, 0, 0

        n = mant_bits + 1
        total_bits = total_num * n

        if n >= 63:
            raise ValueError(
                f"FP mantissa statistic width n={n} is too large "
                f"for int64 bit operations."
            )

        flat = tensor.detach().reshape(-1)

        abs_less_th = 0
        zero_bits_total = 0
        sparse_bits_total = 0
        amplitude_zero_bits_total = 0

        for start in range(0, total_num, chunk_size):
            end = min(start + chunk_size, total_num)
            chunk = flat[start:end]

            chunk_f32 = chunk.detach().to(torch.float32)
            chunk_f32 = torch.nan_to_num(
                chunk_f32, nan=0.0, posinf=0.0, neginf=0.0
            )

            abs_less_th += int((torch.abs(chunk_f32) <= th).sum().item())

            fmt_lower = fmt.lower().strip()

            if fmt_lower in {"e4m3", "e4m3fn", "fp8_e4m3", "fp8_e4m3fn"}:
                raw = (
                    chunk.to(torch.float8_e4m3fn)
                    .view(torch.uint8)
                    .reshape(-1)
                    .to(torch.int64)
                )
                sm_flat, n_bits = self._extract_sm_from_raw(raw, "e4m3")
            elif fmt_lower in {"e2m1", "fp4", "fp4_e2m1"}:
                # FP4 fake-quant values mapped to E2M1 4-bit codes.
                abs_v = chunk_f32.abs()
                e2m1_code = torch.zeros_like(abs_v, dtype=torch.int64)
                e2m1_code[abs_v >= 5.0] = 7
                e2m1_code[(abs_v >= 3.5) & (abs_v < 5.0)] = 6
                e2m1_code[(abs_v >= 2.5) & (abs_v < 3.5)] = 5
                e2m1_code[(abs_v >= 1.75) & (abs_v < 2.5)] = 4
                e2m1_code[(abs_v >= 1.25) & (abs_v < 1.75)] = 3
                e2m1_code[(abs_v >= 0.75) & (abs_v < 1.25)] = 2
                e2m1_code[(abs_v >= 0.25) & (abs_v < 0.75)] = 1
                sign = (chunk_f32 < 0).to(torch.int64)
                raw = (sign << 3) | e2m1_code  # 4-bit S E E M
                sm_flat, n_bits = self._extract_sm_from_raw(raw, "e2m1")
            elif fmt_lower in {"e5m10", "fp16", "float16"}:
                raw = (
                    chunk.detach()
                    .to(torch.float16)
                    .view(torch.int16)
                    .reshape(-1)
                    .to(torch.int64)
                )
                sm_flat, n_bits = self._extract_sm_from_raw(raw, "e5m10")
            else:
                sm_flat = self._fp_tensor_to_mantissa_fixed_chunk(
                    chunk, fmt
                ).reshape(-1)
                n_bits = n

            for i in range(n_bits):
                bit = (sm_flat >> i) & 1
                zero_bit = 1 - bit
                bit_zero_count = int(zero_bit.sum().item())

                zero_bits_total += bit_zero_count
                sparse_bits_total += bit_zero_count
                amplitude_zero_bits_total += bit_zero_count

            del chunk, chunk_f32, sm_flat

        return (
            total_num,
            abs_less_th,
            total_bits,
            zero_bits_total,
            sparse_bits_total,
            amplitude_zero_bits_total,
        )

    # -------------------------------------------------------------------------
    # Sparsity: unit/block statistics (a-bit x b-dim groups)
    # -------------------------------------------------------------------------

    def _fp_stat_width(self, fmt: str) -> int:
        """Statistic width for FP formats (mantissa_int incl. hidden 1)."""
        info = self._get_fp_format(fmt)
        return info["mant_bits"] + 1

    def _split_layer_key(self, layer_key: str) -> Tuple[str, int]:
        """
        Parse layer key into (layer_type, layer_idx).

        Examples:
            q_proj_0        -> ("q_proj", 0)
            qk_matmul_A_0   -> ("qk_matmul_A", 0)
            pv_matmul_B_23  -> ("pv_matmul_B", 23)
        """
        parts = layer_key.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        return layer_key, -1

    def _get_unit_counter(self, phase: str, layer_key: str) -> Dict[str, int]:
        if phase not in self.unit_sparsity:
            self.unit_sparsity[phase] = {}
        if layer_key not in self.unit_sparsity[phase]:
            self.unit_sparsity[phase][layer_key] = {
                "zero_units": 0,
                "total_units": 0,
            }
        return self.unit_sparsity[phase][layer_key]

    def collect_unit_sparsity(
        self,
        layer_name: str,
        layer_idx: int,
        activation: torch.Tensor,
        spec: QuantSpec,
    ):
        """Accumulate unit sparsity for one layer's activation."""
        if activation is None or spec is None:
            return

        with torch.no_grad():
            phase = self.current_phase
            layer_key = f"{layer_name}_{layer_idx}"

            a = self.unit_bit_group_size
            b = self.unit_dim_group_size

            if spec.kind == "int":
                zero_units, total_units = self.compute_unit_sparsity_int(
                    activation,
                    bits=spec.bits,
                    bit_group_size=a,
                    dim_group_size=b,
                )
            elif spec.kind in {"fp", "bf"}:
                zero_units, total_units = self.compute_unit_sparsity_fp(
                    activation,
                    fmt=spec.fmt,
                    bit_group_size=a,
                    dim_group_size=b,
                )
            else:
                return

            counter = self._get_unit_counter(phase, layer_key)
            counter["zero_units"] += int(zero_units)
            counter["total_units"] += int(total_units)

    def collect_unit_sparsity_structured(
        self,
        *,
        module_id: str,
        layer_idx: int,
        tensor: torch.Tensor,
        spec: QuantSpec,
        phase: str,
        flow_step: int,
        tensor_role: str,
        attention_kind: str = "unknown",
    ):
        """
        Phase H (manual §5): accumulate unit sparsity into a record whose key
        matches the main structured record (module_id, phase, flow_step,
        tensor_role, attention_kind), so operator / flow-step / role unit
        analysis is possible (unlike the legacy phase->layer_key container).
        """
        if tensor is None or spec is None:
            return

        with torch.no_grad():
            a = self.unit_bit_group_size
            b = self.unit_dim_group_size

            if spec.kind == "int":
                zero_units, total_units = self.compute_unit_sparsity_int(
                    tensor, bits=spec.bits, bit_group_size=a, dim_group_size=b
                )
            elif spec.kind in {"fp", "bf"}:
                zero_units, total_units = self.compute_unit_sparsity_fp(
                    tensor, fmt=spec.fmt, bit_group_size=a, dim_group_size=b
                )
            else:
                return

            key = (
                module_id,
                phase,
                int(flow_step),
                tensor_role,
                attention_kind or "unknown",
            )
            counter = self.per_role_unit_sparsity.setdefault(
                key,
                {
                    "zero_units": 0,
                    "total_units": 0,
                    "unit_bit_group_size": int(a),
                    "unit_dim_group_size": int(b),
                },
            )
            counter["zero_units"] += int(zero_units)
            counter["total_units"] += int(total_units)

    def compute_unit_sparsity_int(
        self,
        tensor: torch.Tensor,
        bits: int,
        bit_group_size: int,
        dim_group_size: int,
        chunk_rows: int = 4096,
    ) -> Tuple[int, int]:
        """
        INT activation unit sparsity.

        Tensor is viewed as [token_like, dim] (trailing dim last).
        One unit = bit_group_size consecutive bits x dim_group_size
        consecutive dims; a unit is "zero" when all its bits are 0.
        Non-divisible tails are zero-padded (not dropped).
        """
        if tensor is None or tensor.numel() == 0:
            return 0, 0

        if tensor.dim() == 1:
            x2d = tensor.detach().reshape(1, -1)
        else:
            x2d = tensor.detach().reshape(-1, tensor.shape[-1])

        rows, dim = x2d.shape

        padded_dim = (
            (dim + dim_group_size - 1) // dim_group_size
        ) * dim_group_size
        padded_bits = (
            (bits + bit_group_size - 1) // bit_group_size
        ) * bit_group_size

        if padded_dim == 0 or padded_bits == 0:
            return 0, 0

        dim_pad = padded_dim - dim

        zero_units_total = 0
        total_units_total = 0

        mask = (1 << bits) - 1

        for row_start in range(0, rows, chunk_rows):
            row_end = min(row_start + chunk_rows, rows)
            chunk = x2d[row_start:row_end, :]

            if dim_pad > 0:
                pad_tensor = torch.zeros(
                    chunk.shape[0],
                    dim_pad,
                    dtype=chunk.dtype,
                    device=chunk.device,
                )
                chunk = torch.cat([chunk, pad_tensor], dim=-1)
                del pad_tensor

            # quant_awo outputs float32 codes; cast to int64 for bit ops.
            int_chunk = chunk.to(torch.int64) & mask

            grouped = int_chunk.reshape(
                int_chunk.shape[0],
                padded_dim // dim_group_size,
                dim_group_size,
            )

            for bit_start in range(0, padded_bits, bit_group_size):
                real_bit_end = min(bit_start + bit_group_size, bits)

                if bit_start >= bits:
                    # Entirely padding-zero bit group -> all-zero units.
                    unit_zero = torch.ones(
                        grouped.shape[0],
                        grouped.shape[1],
                        dtype=torch.bool,
                        device=grouped.device,
                    )
                else:
                    real_group_bits = real_bit_end - bit_start
                    bit_mask = ((1 << real_group_bits) - 1) << bit_start

                    selected_bits = grouped & bit_mask
                    unit_nonzero = selected_bits.ne(0).any(dim=-1)
                    unit_zero = ~unit_nonzero

                    del selected_bits, unit_nonzero

                zero_units_total += int(unit_zero.sum().item())
                total_units_total += int(unit_zero.numel())
                del unit_zero

            del chunk, int_chunk, grouped

        return zero_units_total, total_units_total

    def compute_unit_sparsity_fp(
        self,
        tensor: torch.Tensor,
        fmt: str,
        bit_group_size: int,
        dim_group_size: int,
        chunk_rows: int = 2048,
    ) -> Tuple[int, int]:
        """
        FP activation unit sparsity.

        Converts FP activations to mantissa fixed-point integers
        (E4M3 -> 4 bit, FP16 -> 11 bit, ...), then counts zero units
        over a-bit x b-dim groups. Non-divisible tails are zero-padded.
        """
        if tensor is None or tensor.numel() == 0:
            return 0, 0

        if tensor.dim() == 1:
            x2d = tensor.detach().reshape(1, -1)
        else:
            x2d = tensor.detach().reshape(-1, tensor.shape[-1])

        rows, dim = x2d.shape

        stat_width = self._fp_stat_width(fmt)

        padded_dim = (
            (dim + dim_group_size - 1) // dim_group_size
        ) * dim_group_size
        padded_bits = (
            (stat_width + bit_group_size - 1) // bit_group_size
        ) * bit_group_size

        if padded_dim == 0 or padded_bits == 0:
            return 0, 0

        dim_pad = padded_dim - dim

        zero_units_total = 0
        total_units_total = 0

        for row_start in range(0, rows, chunk_rows):
            row_end = min(row_start + chunk_rows, rows)
            chunk = x2d[row_start:row_end, :]

            fixed = self._fp_tensor_to_mantissa_fixed_chunk(chunk, fmt)

            if dim_pad > 0:
                pad_tensor = torch.zeros(
                    fixed.shape[0],
                    dim_pad,
                    dtype=fixed.dtype,
                    device=fixed.device,
                )
                fixed = torch.cat([fixed, pad_tensor], dim=-1)
                del pad_tensor

            grouped = fixed.reshape(
                fixed.shape[0],
                padded_dim // dim_group_size,
                dim_group_size,
            )

            for bit_start in range(0, padded_bits, bit_group_size):
                real_bit_end = min(bit_start + bit_group_size, stat_width)

                if bit_start >= stat_width:
                    unit_zero = torch.ones(
                        grouped.shape[0],
                        grouped.shape[1],
                        dtype=torch.bool,
                        device=grouped.device,
                    )
                else:
                    real_group_bits = real_bit_end - bit_start
                    bit_mask = ((1 << real_group_bits) - 1) << bit_start

                    selected_bits = grouped & bit_mask
                    unit_nonzero = selected_bits.ne(0).any(dim=-1)
                    unit_zero = ~unit_nonzero

                    del selected_bits, unit_nonzero

                zero_units_total += int(unit_zero.sum().item())
                total_units_total += int(unit_zero.numel())
                del unit_zero

            del chunk, fixed, grouped

        return zero_units_total, total_units_total

    # -------------------------------------------------------------------------
    # Sparsity: reporting & export
    # -------------------------------------------------------------------------

    def _print_one_sparsity_counter(self, title: str, counter: dict):
        elem = counter["total_element_count"]
        bits = counter["total_bit_count"]

        print(f"\n[{title}]")

        if elem > 0:
            print(
                f"  量化激活零值比例: "
                f"{counter['total_zero_count'] / elem:.4%}"
            )
            print(
                f"  零值元素: "
                f"{counter['total_zero_count']:,} / {elem:,}"
            )
        else:
            print("  未收集到元素级统计")

        if bits > 0:
            print(
                f"  稀疏比特比例: "
                f"{counter['total_sparsebit_count'] / bits:.4%}"
            )
            print(
                f"  Sign Magnitude编码比例: "
                f"{counter['total_amplitude_zero_bits_total'] / bits:.4%}"
            )
            print(
                f"  零比特比例: "
                f"{counter['total_0bit_count'] / bits:.4%}"
            )
        else:
            print("  未收集到bit级统计")

    def print_global_sparsity(self, title: str = "GLOBAL SPARSITY"):
        print("\n" + "-" * 80)
        print(title)
        print("-" * 80)

        counter = {
            "total_zero_count": self.total_zero_count,
            "total_element_count": self.total_element_count,
            "total_bit_count": self.total_bit_count,
            "total_0bit_count": self.total_0bit_count,
            "total_sparsebit_count": self.total_sparsebit_count,
            "total_amplitude_zero_bits_total": self.total_amplitude_zero_bits_total,
            "total_amplitude_bit_count": self.total_amplitude_bit_count,
        }

        self._print_one_sparsity_counter("TOTAL", counter)

    def print_prefill_decode_sparsity(self):
        print("\n" + "-" * 80)
        print("PREFILL / DECODE SPARSITY")
        print("-" * 80)

        self._print_one_sparsity_counter(
            "PREFILL", self.phase_sparsity["prefill"],
        )
        self._print_one_sparsity_counter(
            "DECODE", self.phase_sparsity["decode"],
        )

    def print_unit_sparsity_by_phase(self):
        print("\n" + "-" * 80)
        print("UNIT / BLOCK SPARSITY BY PHASE AND LAYER")
        print("-" * 80)

        print(
            f"unit config: "
            f"a(bit_group_size)={self.unit_bit_group_size}, "
            f"b(dim_group_size)={self.unit_dim_group_size}"
        )

        for phase in ["prefill", "decode"]:
            print(f"\n[{phase.upper()}]")

            phase_stats = self.unit_sparsity.get(phase, {})

            if not phase_stats:
                print("  No unit sparsity statistics collected.")
                continue

            for layer_key in sorted(phase_stats.keys()):
                counter = phase_stats[layer_key]
                total = counter["total_units"]
                zero = counter["zero_units"]

                if total > 0:
                    ratio = zero / total
                    print(
                        f"  {layer_key}: "
                        f"zero_units={zero:,}, "
                        f"total_units={total:,}, "
                        f"unit_zero_ratio={ratio:.4%}"
                    )
                else:
                    print(f"  {layer_key}: no valid units")

    def print_collected_layer_names(
        self,
        phase: str = "full_forward",
        title: Optional[str] = None,
    ):
        if title is None:
            title = f"{phase.upper()} COLLECTED LAYERS"

        print("\n" + "-" * 80)
        print(title)
        print("-" * 80)

        names = sorted(
            self.collected_layer_names_by_phase.get(phase, set())
        )
        call_counts = self.collected_layer_call_count_by_phase.get(phase, {})

        print(f"phase: {phase}")
        print(f"unique collected layers: {len(names)}")

        if len(names) == 0:
            print("No collected layers.")
            return

        prefix_counter = {}
        for layer_key in names:
            prefix, _ = self._split_layer_key(layer_key)
            prefix_counter[prefix] = prefix_counter.get(prefix, 0) + 1

        print("\nLayer type summary:")
        for prefix in sorted(prefix_counter.keys()):
            print(f"  {prefix}: {prefix_counter[prefix]} layers")

        print("\nCollected layer names:")
        for layer_key in names:
            print(
                f"  {layer_key} "
                f"(calls={call_counts.get(layer_key, 0)})"
            )

    def export_unit_sparsity_csv(
        self,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
    ):
        """Export per-layer unit sparsity to CSV."""
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "model_path", "phase", "layer_key",
                "layer_type", "layer_idx",
                "zero_units", "total_units", "unit_zero_ratio",
            ])

            for phase in ["prefill", "decode"]:
                phase_stats = self.unit_sparsity.get(phase, {})

                for layer_key in sorted(phase_stats.keys()):
                    counter = phase_stats[layer_key]
                    zero = int(counter.get("zero_units", 0))
                    total = int(counter.get("total_units", 0))
                    ratio = zero / total if total > 0 else 0.0

                    layer_type, layer_idx = self._split_layer_key(layer_key)

                    writer.writerow([
                        config_name, model_path, phase, layer_key,
                        layer_type, layer_idx,
                        zero, total, ratio,
                    ])

    def export_unit_sparsity_summary_csv(
        self,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
        append: bool = True,
    ):
        """Export per-layer-type aggregated unit sparsity to CSV."""
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        file_exists = os.path.exists(csv_path)
        mode = "a" if append else "w"

        agg = {}

        for phase in ["prefill", "decode"]:
            phase_stats = self.unit_sparsity.get(phase, {})

            for layer_key, counter in phase_stats.items():
                layer_type, _ = self._split_layer_key(layer_key)
                key = (phase, layer_type)

                if key not in agg:
                    agg[key] = {
                        "num_layers": 0,
                        "zero_units": 0,
                        "total_units": 0,
                    }

                agg[key]["num_layers"] += 1
                agg[key]["zero_units"] += int(counter.get("zero_units", 0))
                agg[key]["total_units"] += int(counter.get("total_units", 0))

        with open(csv_path, mode, newline="") as f:
            writer = csv.writer(f)

            if (not file_exists) or (not append):
                writer.writerow([
                    "config", "model_path", "phase", "layer_type",
                    "num_layers", "zero_units", "total_units",
                    "unit_zero_ratio",
                ])

            for (phase, layer_type) in sorted(agg.keys()):
                item = agg[(phase, layer_type)]
                zero = item["zero_units"]
                total = item["total_units"]
                ratio = zero / total if total > 0 else 0.0

                writer.writerow([
                    config_name, model_path, phase, layer_type,
                    item["num_layers"], zero, total, ratio,
                ])

    def export_collected_layers_csv(
        self,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
        phases=None,
        append: bool = False,
    ):
        """Export per-layer collection/call-count audit to CSV."""
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        if phases is None:
            phases = list(self._PHASES)

        file_exists = os.path.exists(csv_path)
        mode = "a" if append else "w"

        with open(csv_path, mode, newline="") as f:
            writer = csv.writer(f)

            if (not file_exists) or (not append):
                writer.writerow([
                    "config", "model_path", "phase", "layer_key",
                    "layer_type", "layer_idx", "call_count",
                ])

            for phase in phases:
                names = sorted(
                    self.collected_layer_names_by_phase.get(phase, set())
                )
                call_counts = self.collected_layer_call_count_by_phase.get(
                    phase, {}
                )

                for layer_key in names:
                    layer_type, layer_idx = self._split_layer_key(layer_key)
                    writer.writerow([
                        config_name, model_path, phase, layer_key,
                        layer_type, layer_idx,
                        call_counts.get(layer_key, 0),
                    ])

    def export_collected_layers_summary_csv(
        self,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
        phases=None,
        append: bool = True,
    ):
        """Export per-layer-type aggregated call-count audit to CSV."""
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        if phases is None:
            phases = list(self._PHASES)

        file_exists = os.path.exists(csv_path)
        mode = "a" if append else "w"

        rows = []

        for phase in phases:
            names = sorted(
                self.collected_layer_names_by_phase.get(phase, set())
            )
            call_counts = self.collected_layer_call_count_by_phase.get(
                phase, {}
            )

            grouped = {}

            for layer_key in names:
                layer_type, layer_idx = self._split_layer_key(layer_key)

                if layer_type not in grouped:
                    grouped[layer_type] = {
                        "indices": [],
                        "calls": [],
                    }

                if layer_idx >= 0:
                    grouped[layer_type]["indices"].append(layer_idx)

                grouped[layer_type]["calls"].append(
                    int(call_counts.get(layer_key, 0))
                )

            for layer_type in sorted(grouped.keys()):
                indices = sorted(set(grouped[layer_type]["indices"]))
                calls = grouped[layer_type]["calls"]

                if len(indices) > 0:
                    min_idx = min(indices)
                    max_idx = max(indices)
                    expected = set(range(min_idx, max_idx + 1))
                    missing = sorted(expected - set(indices))
                    missing_str = "|".join(str(x) for x in missing)
                else:
                    min_idx = -1
                    max_idx = -1
                    missing_str = ""

                total_calls = sum(calls) if calls else 0
                min_calls = min(calls) if calls else 0
                max_calls = max(calls) if calls else 0

                rows.append([
                    config_name, model_path, phase, layer_type,
                    len(indices), min_idx, max_idx,
                    total_calls, min_calls, max_calls, missing_str,
                ])

        with open(csv_path, mode, newline="") as f:
            writer = csv.writer(f)

            if (not file_exists) or (not append):
                writer.writerow([
                    "config", "model_path", "phase", "layer_type",
                    "num_layers", "min_layer_idx", "max_layer_idx",
                    "total_calls", "min_calls", "max_calls",
                    "missing_layer_indices",
                ])

            writer.writerows(rows)

    def export_module_sparsity_csv(
        self,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
    ):
        """
        Export structured sparsity records (module_id × phase × tensor_role)
        to CSV — the manual §46 module_sparsity.csv.

        Ratios are computed from accumulated numerator/denominator sums
        (manual §23: never average per-layer ratios).
        """
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "model_path",
                "module_id", "component", "layer_idx", "operator",
                "phase", "flow_step", "tensor_role", "attention_kind",
                "calls",
                "total_elements_reported", "zero_elements_reported",
                "zero_rate_reported",
                "protected_elements", "fp_sidepath_ratio",
                "total_elements_native", "zero_elements_native",
                "zero_rate_native",
                "total_bits_reported", "sparse_bits_reported",
                "sparse_bit_rate_reported",
                "protected_bits", "total_bits_native",
                "sparse_bits_native", "sparse_bit_rate_native",
                "amplitude_zero_bits_native",
                "amplitude_zero_bit_rate_native",
                "ideal_sparse_upper_bound",
            ])

            for key in sorted(
                self.per_role_sparsity.keys(),
                key=lambda k: (
                    str(k[0]), str(k[1]), k[2], str(k[3]), str(k[4])
                ),
            ):
                entry = self.per_role_sparsity[key]
                module_id = entry.get("module_id", key[0])
                phase = entry.get("phase", key[1])
                flow_step = entry.get("flow_step", key[2])
                tensor_role = entry.get("tensor_role", key[3])
                attention_kind = entry.get("attention_kind", key[4])

                total_elements = int(entry.get("total_elements", 0))
                zero_elements = int(entry.get("zero_elements", 0))
                total_bits = int(entry.get("total_bits", 0))
                sparse_bits = int(entry.get("sparse_bits", 0))
                amp_zero_bits = int(entry.get("amplitude_zero_bits", 0))

                partition = self.outlier_partition.get(key, {})
                protected_elements = int(
                    partition.get("protected_elements", 0)
                )

                # Native quant path = reported minus FP side-path positions
                # that the outlier forward forces to zero (manual §3.2).
                total_elements_native = total_elements - protected_elements
                zero_elements_native = zero_elements - protected_elements

                # Bit-level: protected positions each contribute `bitwidth`
                # zero bits to the reported numerator/denominator.
                bitwidth = (
                    total_bits // total_elements if total_elements > 0 else 0
                )
                protected_bits = protected_elements * bitwidth
                total_bits_native = total_bits - protected_bits
                sparse_bits_native = sparse_bits - protected_bits
                amp_zero_bits_native = amp_zero_bits - protected_bits

                zero_rate_reported = (
                    zero_elements / total_elements
                    if total_elements > 0 else 0.0
                )
                fp_sidepath_ratio = (
                    protected_elements / total_elements
                    if total_elements > 0 else 0.0
                )
                zero_rate_native = (
                    zero_elements_native / total_elements_native
                    if total_elements_native > 0 else 0.0
                )
                sparse_bit_rate_reported = (
                    sparse_bits / total_bits if total_bits > 0 else 0.0
                )
                sparse_bit_rate_native = (
                    sparse_bits_native / total_bits_native
                    if total_bits_native > 0 else 0.0
                )
                amp_zero_bit_rate_native = (
                    amp_zero_bits_native / total_bits_native
                    if total_bits_native > 0 else 0.0
                )
                # Phase H: the exported upper bound reflects NATIVE
                # quant-path sparsity (not the masked reported one).
                upper_bound = (
                    1.0 / (1.0 - sparse_bit_rate_native)
                    if sparse_bit_rate_native < 1.0
                    else float("inf")
                )

                writer.writerow([
                    config_name, model_path,
                    module_id,
                    entry.get("component", ""),
                    entry.get("layer_idx", -1),
                    entry.get("operator", ""),
                    phase, flow_step, tensor_role, attention_kind,
                    entry.get("calls", 0),
                    total_elements, zero_elements, zero_rate_reported,
                    protected_elements, fp_sidepath_ratio,
                    total_elements_native, zero_elements_native,
                    zero_rate_native,
                    total_bits, sparse_bits, sparse_bit_rate_reported,
                    protected_bits, total_bits_native,
                    sparse_bits_native, sparse_bit_rate_native,
                    amp_zero_bits_native, amp_zero_bit_rate_native,
                    upper_bound,
                ])

    def export_workload_csv(
        self,
        model,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
    ):
        """
        Export the hardware-facing workload trace (manual §49).

        One row per (module_id, phase, tensor_role) with call counts from
        the runtime collection, GEMM shapes from the wrapped modules, and
        first-order active-bit-op proxies:

            BOP_dense  = MACs * B_A * B_B        (full bitwidths)
            BOP_active = MACs * b_A  * b_B       (1 - bit sparsity scaled)

        These are hardware PROXIES (manual §55), not measured latency.

        Linear shapes: x[B,T,K] @ W[N,K]^T  -> M=B*T, K, N.
        MatMul (QK/PV) physical MACs are accumulated from the actual A/O
        runtime shapes: MACs = O.numel() * A.shape[-1].
        """
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        def _macs_linear(module) -> tuple:
            M = None  # batch*token count is runtime-dependent; use per-call
            K = module.in_features
            N = module.out_features
            return K, N

        rows = []

        for module in model.modules():
            cls_name = type(module).__name__
            # Strict class check first; duck-typing fallback ONLY for test
            # fakes (a real nn.Linear must never be picked up: it lacks
            # a_spec / A_spec).
            if cls_name == "QuantizedLinear":
                is_linear, is_matmul = True, False
            elif cls_name == "QuantizedMatMul":
                is_linear, is_matmul = False, True
            elif hasattr(module, "A_spec") and hasattr(module, "B_spec"):
                is_linear, is_matmul = False, True
            elif (
                hasattr(module, "a_spec")
                and hasattr(module, "w_spec")
                and hasattr(module, "in_features")
            ):
                is_linear, is_matmul = True, False
            else:
                continue
            module_id = getattr(module, "module_id", "") or (
                f"{getattr(module, 'layer_name', '')}_"
                f"{getattr(module, 'layer_idx', 0)}"
            )

            if is_linear:
                op_type = "linear"
                K, N = _macs_linear(module)
                a_bits = module.a_bit if module.a_spec.kind == "int" else 8
                b_bits = module.w_bit if module.w_spec.kind == "int" else 8
            else:
                op_type = "matmul"
                # Physical MatMul dims/MACs are resolved below from the
                # exact A/O runtime-shape accumulator.
                K = N = None
                a_bits = module.A_bit if module.A_spec.kind == "int" else 8
                b_bits = module.B_bit if module.B_spec.kind == "int" else 8

            for (mid, phase, flow_step, role, attn), entry in (
                self.per_role_sparsity.items()
            ):
                if mid != module_id:
                    continue
                calls = entry.get("calls", 0)
                elems = entry.get("total_elements", 0)
                bits = entry.get("total_bits", 0)
                bit_sparsity = entry.get("sparse_bit_rate", 0.0)

                if is_linear:
                    if calls > 0 and K:
                        M = elems // (calls * K)
                        macs = M * K * (N or 0)
                    else:
                        M = None
                        macs = None
                else:
                    wk = self.matmul_workload.get(
                        (module_id, phase, flow_step, attn)
                    )
                    if wk and wk.get("calls", 0) > 0:
                        M = wk.get("last_M")
                        K = wk.get("last_K")
                        N = wk.get("last_N")
                        macs = wk["macs_total"] / wk["calls"]
                    else:
                        M = K = N = None
                        macs = None

                if macs is not None:
                    b_a = a_bits * (1 - bit_sparsity)
                    b_b = b_bits * (1 - bit_sparsity)
                    bop_dense = macs * a_bits * b_bits
                    bop_active = macs * b_a * b_b
                else:
                    bop_dense = bop_active = None

                mac_semantics = (
                    "linear_role_derived_v1"
                    if is_linear
                    else (
                        "matmul_physical_exact_v2"
                        if macs is not None else "matmul_missing"
                    )
                )
                rows.append([
                    config_name, model_path,
                    module_id, phase, flow_step, role, attn, op_type, calls,
                    M, K, N, elems, bits,
                    a_bits, b_bits,
                    entry.get("sparse_bit_rate", 0.0),
                    macs, mac_semantics, bop_dense, bop_active,
                ])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "model_path",
                "module_id", "phase", "flow_step", "tensor_role",
                "attention_kind", "op_type", "calls",
                "M", "K", "N", "elements", "bits",
                "A_bitwidth", "B_bitwidth",
                "bit_sparsity",
                "MACs", "MAC_semantics",
                "BOP_dense_proxy", "BOP_active_proxy",
            ])
            writer.writerows(rows)

        return len(rows)

    def export_quantization_manifest_csv(
        self,
        model,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
    ):
        """
        Phase H (manual §6): export a quantization / deployment coverage
        manifest (quantization_manifest.csv). One row per wrapped physical
        operator (QuantizedLinear / QuantizedMatMul) describing its actual
        quantized method + operand formats, so per-component sparsity is
        never mistaken for whole-model sparsity.
        """
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        rows = []
        for module in model.modules():
            cls_name = type(module).__name__
            if cls_name == "QuantizedLinear":
                is_linear, is_matmul = True, False
            elif cls_name == "QuantizedMatMul":
                is_linear, is_matmul = False, True
            elif hasattr(module, "A_spec") and hasattr(module, "B_spec"):
                is_linear, is_matmul = False, True
            elif (
                hasattr(module, "a_spec")
                and hasattr(module, "w_spec")
                and hasattr(module, "in_features")
            ):
                is_linear, is_matmul = True, False
            else:
                continue

            module_id = getattr(module, "module_id", "") or getattr(
                module, "layer_name", ""
            )
            component, operator, layer_idx = self._parse_module_id(module_id)
            method = getattr(module, "method", "")

            if is_linear:
                rows.append([
                    config_name, model_path, module_id, component, layer_idx,
                    operator, "linear", "true", method,
                    module.a_spec.name() if module.a_spec else "",
                    module.w_spec.name() if module.w_spec else "",
                    module.o_spec.name() if module.o_spec else "",
                    "per_output_channel"
                    if method == "pot_ao_outlier_channel" else "per_tensor",
                    getattr(module, "outlier_ratio", 0.0),
                    getattr(module, "in_features", None),
                    getattr(module, "out_features", None),
                    "",
                ])
            else:
                rows.append([
                    config_name, model_path, module_id, component, layer_idx,
                    operator, "matmul", "true", method,
                    module.A_spec.name() if module.A_spec else "",
                    module.B_spec.name() if module.B_spec else "",
                    module.O_spec.name() if module.O_spec else "",
                    "",
                    getattr(module, "outlier_ratio", 0.0),
                    None, None,
                    "",
                ])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "model_path",
                "module_id", "component", "layer_idx", "operator",
                "op_type", "quantized", "method",
                "a_or_A_kind", "w_or_B_kind", "o_or_O_kind",
                "weight_quant_granularity", "outlier_ratio",
                "in_features", "out_features", "attention_kind",
            ])
            writer.writerows(rows)

        return len(rows)

    def export_outlier_sidepath_csv(
        self,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
    ):
        """
        Phase H (manual §8.3): export the outlier FP side-path partition
        (outlier_sidepath.csv). One row per (module_id, phase, flow_step,
        tensor_role, attention_kind); ratios from accumulated counters.
        """
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "model_path",
                "module_id", "phase", "flow_step", "tensor_role",
                "attention_kind", "calls",
                "total_elements", "protected_elements", "fp_sidepath_ratio",
            ])

            for key in sorted(
                self.outlier_partition.keys(),
                key=lambda k: (
                    str(k[0]), str(k[1]), k[2], str(k[3]), str(k[4])
                ),
            ):
                e = self.outlier_partition[key]
                total = int(e.get("total_elements", 0))
                protected = int(e.get("protected_elements", 0))
                ratio = protected / total if total > 0 else 0.0

                writer.writerow([
                    config_name, model_path,
                    e.get("module_id", key[0]),
                    e.get("phase", key[1]),
                    e.get("flow_step", key[2]),
                    e.get("tensor_role", key[3]),
                    e.get("attention_kind", key[4]),
                    e.get("calls", 0),
                    total, protected, ratio,
                ])

    def export_unit_sparsity_structured_csv(
        self,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
    ):
        """
        Phase H (manual §8.4): export structured unit sparsity aligned with
        the main record key (module_id, phase, flow_step, tensor_role,
        attention_kind).
        """
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "model_path",
                "module_id", "phase", "flow_step", "tensor_role",
                "attention_kind",
                "unit_bit_group_size", "unit_dim_group_size",
                "zero_units", "total_units", "unit_zero_rate",
            ])

            for key in sorted(
                self.per_role_unit_sparsity.keys(),
                key=lambda k: (
                    str(k[0]), str(k[1]), k[2], str(k[3]), str(k[4])
                ),
            ):
                c = self.per_role_unit_sparsity[key]
                zero = int(c.get("zero_units", 0))
                total = int(c.get("total_units", 0))
                ratio = zero / total if total > 0 else 0.0

                writer.writerow([
                    config_name, model_path,
                    key[0], key[1], key[2], key[3], key[4],
                    c.get("unit_bit_group_size", 0),
                    c.get("unit_dim_group_size", 0),
                    zero, total, ratio,
                ])

    def export_fp_code_audit_csv(
        self,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
    ):
        """Export the independent FP-code audit records (H1-Audit) to CSV."""
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        fieldnames = [
            "config", "model_path",
            "module_id",
            "phase",
            "flow_step",
            "tensor_role",
            "attention_kind",

            "calls",
            "sampled_elements",
            "valid_elements",
            "nonzero_elements",

            "zero_code_count",
            "zero_code_rate",

            "subnormal_count",
            "subnormal_rate",

            "nan_count",
            "nan_rate",

            "negative_count",
            "negative_rate",

            "saturation_count",
            "saturation_rate",

            "sig_zero_bits",
            "sig_total_bits",
            "sig_zero_rate",

            "sig_nonzero_zero_bits",
            "sig_nonzero_total_bits",
            "sig_nonzero_zero_rate",

            "mant_111_nonzero_rate",
        ]

        fieldnames += [f"mant_{i:03b}" for i in range(8)]
        fieldnames += [f"mant_nonzero_{i:03b}" for i in range(8)]
        fieldnames += [f"exp_{i:04b}" for i in range(16)]
        fieldnames += [f"sig_{i:04b}" for i in range(16)]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for key in sorted(self.fp_code_audit.keys()):
                e = self.fp_code_audit[key]
                sampled = e["sampled_elements"]
                valid = e["valid_elements"]
                nonzero = e["nonzero_elements"]
                sig_total = e["sig_total_bits"]
                sig_nz_total = e["sig_nonzero_total_bits"]
                mant_nz_total = sum(e["mant_nonzero_hist"])

                row = {
                    "config": config_name,
                    "model_path": model_path,
                    "module_id": e["module_id"],
                    "phase": e["phase"],
                    "flow_step": e["flow_step"],
                    "tensor_role": e["tensor_role"],
                    "attention_kind": e["attention_kind"],

                    "calls": e["calls"],
                    "sampled_elements": sampled,
                    "valid_elements": valid,
                    "nonzero_elements": nonzero,

                    "zero_code_count": e["zero_code_count"],
                    "zero_code_rate": (
                        e["zero_code_count"] / valid if valid else 0.0
                    ),

                    "subnormal_count": e["subnormal_count"],
                    "subnormal_rate": (
                        e["subnormal_count"] / valid if valid else 0.0
                    ),

                    "nan_count": e["nan_count"],
                    "nan_rate": (
                        e["nan_count"] / sampled if sampled else 0.0
                    ),

                    "negative_count": e["negative_count"],
                    "negative_rate": (
                        e["negative_count"] / valid if valid else 0.0
                    ),

                    "saturation_count": e["saturation_count"],
                    "saturation_rate": (
                        e["saturation_count"] / valid if valid else 0.0
                    ),

                    "sig_zero_bits": e["sig_zero_bits"],
                    "sig_total_bits": sig_total,
                    "sig_zero_rate": (
                        e["sig_zero_bits"] / sig_total
                        if sig_total else 0.0
                    ),

                    "sig_nonzero_zero_bits": e[
                        "sig_nonzero_zero_bits"
                    ],
                    "sig_nonzero_total_bits": sig_nz_total,
                    "sig_nonzero_zero_rate": (
                        e["sig_nonzero_zero_bits"] / sig_nz_total
                        if sig_nz_total else 0.0
                    ),

                    "mant_111_nonzero_rate": (
                        e["mant_nonzero_hist"][7] / mant_nz_total
                        if mant_nz_total else 0.0
                    ),
                }

                for i in range(8):
                    row[f"mant_{i:03b}"] = e["mant_hist"][i]
                    row[f"mant_nonzero_{i:03b}"] = e[
                        "mant_nonzero_hist"
                    ][i]

                for i in range(16):
                    row[f"exp_{i:04b}"] = e["exp_hist"][i]
                    row[f"sig_{i:04b}"] = e["sig_hist"][i]

                writer.writerow(row)

    def export_per_layer_sparsity_csv(
        self,
        csv_path: str,
        config_name: str = "",
        model_path: str = "",
    ):
        """Export per-layer element/bit sparsity records to CSV."""
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "model_path", "layer_key", "layer_type",
                "layer_idx",
                "total_elements", "zero_elements", "zero_rate",
                "total_bits", "zero_bits", "sparse_bits",
                "amplitude_zero_bits", "sparse_bit_rate",
                "amplitude_zero_bit_rate", "ideal_sparse_upper_bound", "1_bits",
            ])

            for layer_key in sorted(self.per_layer_sparsity.keys()):
                entry = self.per_layer_sparsity[layer_key]
                layer_type, layer_idx = self._split_layer_key(layer_key)

                writer.writerow([
                    config_name, model_path, layer_key, layer_type,
                    layer_idx,
                    entry.get("total_elements", 0),
                    entry.get("zero_elements", 0),
                    entry.get("zero_rate", 0.0),
                    entry.get("total_bits", 0),
                    entry.get("zero_bits", 0),
                    entry.get("sparse_bits", 0),
                    entry.get("amplitude_zero_bits", 0),
                    entry.get("sparse_bit_rate", 0.0),
                    entry.get("amplitude_zero_bit_rate", 0.0),
                    entry.get("ideal_sparse_upper_bound", 0.0),
                    entry.get("1_bits", 0),
                ])

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save_all_scales(self):
        """Save all collected scales to pickle files."""
        print(f"Saving scales to {self.scale_dir}...")

        for key, stat in self.stats.items():
            scales = stat.get_final_scales()

            for scale_type, scale_value in scales.items():
                filename = f"{stat.layer_name}_{scale_type}_{stat.layer_idx}.p"
                filepath = os.path.join(self.scale_dir, filename)

                with open(filepath, 'wb') as f:
                    pickle.dump(scale_value, f)

                # Per-channel scales are [N_out] tensors (G2-B); print a
                # compact summary instead of a scalar format string.
                if isinstance(scale_value, torch.Tensor):
                    print(
                        f"  Saved {filename}: tensor{tuple(scale_value.shape)}"
                        f" min={scale_value.min().item():.3e}"
                        f" max={scale_value.max().item():.3e}"
                    )
                else:
                    print(f"  Saved {filename}: {scale_value:.6f}")

        print(f"Total scales saved: {len(self.stats)} layers")

    def load_all_scales(self) -> Dict[str, Dict[str, float]]:
        """Load all scales from files."""
        loaded_scales = {}

        for key, stat in self.stats.items():
            scales = {}

            for scale_type in [
                'w_scale', 'a_scale', 'o_scale',
                'A_scale', 'B_scale', 'O_scale',
            ]:
                filename = f"{stat.layer_name}_{scale_type}_{stat.layer_idx}.p"
                filepath = os.path.join(self.scale_dir, filename)

                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        scales[scale_type] = pickle.load(f)

            if scales:
                loaded_scales[key] = scales

        return loaded_scales

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            'total_layers': len(self.stats),
            'total_samples': sum(
                stat.sample_count for stat in self.stats.values()
            ),
            'layers': {},
        }

        for key, stat in self.stats.items():
            summary['layers'][key] = {
                'layer_name': stat.layer_name,
                'layer_idx': stat.layer_idx,
                'sample_count': stat.sample_count,
                'scales': stat.get_final_scales(),
            }

        return summary

    def print_summary(self):
        """Print summary of collected statistics."""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("Quantization Statistics Summary")
        print("=" * 80)
        print(f"Total layers: {summary['total_layers']}")
        print(f"Total samples: {summary['total_samples']}")
        print("\nPer-layer statistics:")
        print("-" * 80)

        for key, info in summary['layers'].items():
            print(f"\n{key}:")
            print(f"  Samples: {info['sample_count']}")
            print(f"  Scales:")
            for scale_name, scale_value in info['scales'].items():
                if isinstance(scale_value, torch.Tensor):
                    # Per-channel [N_out] scales (G2-B): compact summary.
                    print(
                        f"    {scale_name}: tensor{tuple(scale_value.shape)}"
                        f" min={scale_value.min().item():.3e}"
                        f" max={scale_value.max().item():.3e}"
                    )
                else:
                    print(f"    {scale_name}: {scale_value:.6f}")

        print("=" * 80 + "\n")

    def reset_all(self):
        """Reset all statistics."""
        for stat in self.stats.values():
            stat.reset()
        self.quant_activation_calls.clear()
        self.reset_sparsity()
