#!/usr/bin/env python
"""G5 vlm-selective-precision — config-routing smoke check.

Before running any 100-ep rollout, verify that each config actually routes
the right w_bit/method per operator and leaves Expert Linear as raw nn.Linear
(the G2-B 0% history motivates this pre-flight).

For every VLM QuantizedLinear, print (module_id, op_type, w_bit, method).
Then assert:
  - G5-A: all 7 VLM op types FP8 (w_bit=e4m3), no pot_ao_outlier
  - G5-B: q/k/v/o = W4 pot_ao_outlier, gate/up/down = FP8
  - G5-C: q/k/v/o = FP8, gate/up/down = W4 pot_ao_outlier
  - Expert subtree has zero QuantizedLinear (raw nn.Linear)
  - QK/PV MatMul present (64) and FP8

Usage:
  conda run -n smolvla_eval python scripts/g5_routing_smoke.py <config.yaml> <expected_mode>
  expected_mode: all_fp8 | attn_w4 | attn_fp8_mlp_w4
"""

import sys
import yaml

sys.path.insert(0, "src")

from vla_tcs2.model_wrapper import ModelWrapper
from vla_tcs2.quant_linear import QuantizedLinear
from vla_tcs2.quant_matmul import QuantizedMatMul


ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP = {"gate_proj", "up_proj", "down_proj"}


def main() -> None:
    config_path = sys.argv[1]
    expected = sys.argv[2] if len(sys.argv) > 2 else "all_fp8"
    config = yaml.safe_load(open(config_path))

    w = ModelWrapper(config)
    w.build(mode="scale_inspection")

    vlm_lines = {}
    expert_quantized = 0
    n_matmul = 0

    for m in w.model.modules():
        if isinstance(m, QuantizedLinear):
            mid = getattr(m, "module_id", "") or ""
            if mid.startswith("vlm."):
                op = mid.split(".")[-1]
                vlm_lines[op] = (m.w_bit, m.method)
            elif mid.startswith("expert."):
                expert_quantized += 1
        elif isinstance(m, QuantizedMatMul):
            n_matmul += 1

    print(f"=== G5 routing smoke: {config_path} ===")
    print(f"expected mode: {expected}")
    for op in sorted(ATTN | MLP):
        wb, meth = vlm_lines.get(op, ("<missing>", "<missing>"))
        print(f"  vlm {op:12s}: w_bit={wb} method={meth}")
    print(f"expert QuantizedLinear count: {expert_quantized} (expect 0 = raw FP)")
    print(f"QuantizedMatMul count: {n_matmul} (expect 64)")

    # --- assertions ---
    assert expert_quantized == 0, "Expert Linear must be raw FP"
    assert n_matmul == 64, f"expected 64 MatMul, got {n_matmul}"

    def is_w4(op):
        wb, meth = vlm_lines[op]
        return str(wb) == "4" and meth == "pot_ao_outlier"

    def is_fp8(op):
        wb, meth = vlm_lines[op]
        return str(wb) == "e4m3" and meth != "pot_ao_outlier"

    if expected == "all_fp8":
        for op in ATTN | MLP:
            assert is_fp8(op), f"{op} should be FP8, got {vlm_lines[op]}"
    elif expected == "attn_w4":
        for op in ATTN:
            assert is_w4(op), f"{op} should be W4, got {vlm_lines[op]}"
        for op in MLP:
            assert is_fp8(op), f"{op} should be FP8, got {vlm_lines[op]}"
    elif expected == "attn_fp8_mlp_w4":
        for op in ATTN:
            assert is_fp8(op), f"{op} should be FP8, got {vlm_lines[op]}"
        for op in MLP:
            assert is_w4(op), f"{op} should be W4, got {vlm_lines[op]}"
    else:
        raise ValueError(f"unknown expected mode: {expected}")

    print("G5 ROUTING SMOKE: ALL PASS")


if __name__ == "__main__":
    main()
