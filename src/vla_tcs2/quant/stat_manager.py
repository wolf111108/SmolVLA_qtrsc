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
        """
        scales = {}

        if self.w_scales:
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
        self.per_layer_sparsity: Dict[str, Dict[str, Any]] = {}

        # Per-layer WEIGHT sparsity records (static, collected once).
        self.per_layer_weight_sparsity: Dict[str, Dict[str, Any]] = {}

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
        self.per_layer_weight_sparsity = {}
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

        Always records the call for auditing; computes bit/unit sparsity
        only when enable_sparsity() has been called.
        """
        key = f"{layer_name}_{layer_idx}"
        self.quant_activation_calls[key] = (
            self.quant_activation_calls.get(key, 0) + 1
        )

        if not self.sparsity_enabled:
            return

        n_extra = len(args)
        if n_extra == 7:
            # Linear call site:
            #   args = (x_code, x_fp16, a_spec, digit_size,
            #           parallelism, in_features, out_features)
            x_code, _x_fp16, a_spec = args[0], args[1], args[2]
            tensors = [(x_code, a_spec)]
        elif n_extra == 9:
            # MatMul call site:
            #   args = (A_sim, A, B_sim, B_spec, A_spec, digit_size,
            #           parallelism, in_features, out_features)
            A_sim, _A, B_sim, B_spec, A_spec = (
                args[0], args[1], args[2], args[3], args[4]
            )
            tensors = [(A_sim, A_spec), (B_sim, B_spec)]
        else:
            raise ValueError(
                f"collect_quant_activation: unsupported positional-arg "
                f"count {n_extra} (expected 7 for linear or 9 for matmul); "
                f"layer={key}"
            )

        for tensor, spec in tensors:
            if tensor is None or spec is None:
                continue
            if spec.kind == "none" or not getattr(spec, "enabled", True):
                # FP16/BF16 pass-through档不参与稀疏统计
                continue
            self._collect_one_tensor_sparsity(layer_name, layer_idx, tensor, spec)

    # -------------------------------------------------------------------------
    # Sparsity: collection internals
    # -------------------------------------------------------------------------

    def _collect_one_tensor_sparsity(
        self,
        layer_name: str,
        layer_idx: int,
        activation: torch.Tensor,
        spec: QuantSpec,
    ):
        """Compute and accumulate element/bit-level + unit sparsity."""
        phase = self.current_phase
        layer_key = f"{layer_name}_{layer_idx}"

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

        # Per-layer records.
        entry = self.per_layer_sparsity.setdefault(
            layer_key,
            {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
            },
        )
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
        entry["ideal_speed_up"] = (
            1 / (1 - entry["sparse_bit_rate"])
            if entry["sparse_bit_rate"] < 1
            else float("inf")
        )
        entry["1_bits"] = (
            entry["total_bits"] - entry["amplitude_zero_bits"]
        )

        # Unit/block sparsity.
        if self.enable_unit_sparsity:
            self.collect_unit_sparsity(layer_name, layer_idx, activation, spec)

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

        entry = self.per_layer_weight_sparsity.setdefault(
            layer_key,
            {"layer_name": layer_name, "layer_idx": layer_idx},
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
        entry["ideal_speed_up"] = (
            1 / (1 - entry["sparse_bit_rate"])
            if entry["sparse_bit_rate"] < 1 else float("inf")
        )
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
                "config", "model_path", "layer_key", "layer_type",
                "layer_idx",
                "total_elements", "zero_elements", "zero_rate",
                "total_bits", "zero_bits", "sparse_bits",
                "amplitude_zero_bits", "sparse_bit_rate",
                "amplitude_zero_bit_rate", "ideal_speed_up", "1_bits",
            ])

            for layer_key in sorted(self.per_layer_weight_sparsity.keys()):
                entry = self.per_layer_weight_sparsity[layer_key]
                layer_type, layer_idx = self._split_layer_key(layer_key)

                weight_writer.writerow([
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
                    entry.get("ideal_speed_up", 0.0),
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
                self.collect_weight_sparsity(
                    module.layer_name, module.layer_idx, w_code, module.w_spec
                )
                n += 1
            elif cls_name == "QuantizedMatMul":
                if module.B_interval is None:
                    continue
                # MatMul "weight" side is B (K/V or V^T); A is dynamic.
                B_code = quant_awo(
                    module.B,
                    module.B_interval,
                    module.B_spec,
                    out_dtype=torch.float32,
                    chunk_size=self.sparse_stat_chunk_size,
                )
                self.collect_weight_sparsity(
                    f"{module.layer_name}_B",
                    module.layer_idx,
                    B_code,
                    module.B_spec,
                )
                n += 1
        return n

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
            sm:       sign(1) << mant_bits | mantissa(mant_bits)
            exp:      biased exponent
            nonzero:  exp != 0 or mant != 0
        """
        raw = raw.to(torch.int64)

        sign = (raw >> sign_shift) & 0x1
        mant = raw & ((1 << mant_bits) - 1)
        exp = (raw >> mant_bits) & ((1 << exp_bits) - 1)

        sm = (sign << mant_bits) | mant
        nonzero = (exp != 0) | (mant != 0)

        return sm, exp, nonzero

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
            sm, _exp, nonzero = self._unpack_sm_exp(
                raw_int, sign_shift=15, exp_bits=5, mant_bits=10,
            )
            width = 11
            mant_bits = 10
        elif fmt == "e4m3":
            sm, _exp, nonzero = self._unpack_sm_exp(
                raw_int, sign_shift=7, exp_bits=4, mant_bits=3,
            )
            width = 4
            mant_bits = 3
        elif fmt == "e2m1":
            sm, _exp, nonzero = self._unpack_sm_exp(
                raw_int, sign_shift=3, exp_bits=2, mant_bits=1,
            )
            width = 2
            mant_bits = 1
        else:
            raise ValueError(f"Unsupported fmt in _extract_sm_from_raw: {fmt}")

        # Add hidden leading 1 for normal numbers (exp != 0).
        hidden = nonzero.to(torch.int64) << mant_bits
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
                "amplitude_zero_bit_rate", "ideal_speed_up", "1_bits",
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
                    entry.get("ideal_speed_up", 0.0),
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
                print(f"    {scale_name}: {scale_value:.6f}")

        print("=" * 80 + "\n")

    def reset_all(self):
        """Reset all statistics."""
        for stat in self.stats.values():
            stat.reset()
        self.quant_activation_calls.clear()
        self.reset_sparsity()
