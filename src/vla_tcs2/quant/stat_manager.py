"""
Statistics Manager for Quantization Calibration.

Ported (simplified) from opt-qt/quant/stat_manager.py.

Kept:
    - QuantStatistics: per-layer scale collection (w/a/o, A/B for matmul)
    - QuantStatManager: registry + pickle save/load + summary printing
      (file naming matches QuantizedLinear._load_scales /
       QuantizedLinear.save_scales: {layer_name}_{type}_{layer_idx}.p)

Dropped (not needed on the VLA-TCS2 path):
    - HW mapping / latency statistics (Mapping_stat, CIM_sys, ...)
    - prefill/decode phase sparsity accounting
    - CSV export helpers
    - unit sparsity counters

NOTE on collect_quant_activation:
    The original implementation runs bit-level HW utilization statistics
    (PAFCIM / Systolic proxies). On the VLA path it is currently a
    bookkeeping-only stub: it records which layers were seen so that a
    run can be audited, but performs no HW modelling. Re-enable later by
    porting HW_proxy if utilization numbers are needed.
"""

import os
import pickle
from typing import Dict, List, Any


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

    def __init__(self, scale_dir: str):
        self.scale_dir = scale_dir
        self.stats: Dict[str, QuantStatistics] = {}

        os.makedirs(scale_dir, exist_ok=True)

        # Bookkeeping for collect_quant_activation auditing (VLA path).
        self.quant_activation_calls: Dict[str, int] = {}

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
        Bookkeeping-only stub (see module docstring).

        Signature is intentionally elastic to accept both call sites:

        - QuantizedLinear.quant_forward (9 positional args):
              (layer_name, layer_idx, x_code, x_code, a_spec,
               digit_size, parallelism, in_features, out_features)
        - QuantizedMatMul.quant_forward (11 positional args):
              (layer_name, layer_idx, A_sim, A, B_sim,
               B_spec, A_spec, digit_size, parallelism,
               in_features, out_features)
        """
        key = f"{layer_name}_{layer_idx}"
        self.quant_activation_calls[key] = (
            self.quant_activation_calls.get(key, 0) + 1
        )

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
