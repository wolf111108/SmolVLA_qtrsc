# Quick smoke test for the sparsity accounting ported into
# src/vla_tcs2/quant/stat_manager.py (from opt-qt stat_manager_old.py).
#
# Run inside the smolvla_eval env (torch >= 2.1 with float8 dtypes):
#   conda run -n smolvla_eval python scripts/test_sparsity_stat_smoke.py

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from vla_tcs2.quant.quant_spec import QuantSpec  # noqa: E402
from vla_tcs2.quant.stat_manager import QuantStatManager  # noqa: E402


def main() -> None:
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()

    # --- INT bit stats: known-value check ---
    t = torch.tensor([1.0, -1.0, 2.0, 0.0])
    num, less, bits, zb, sb, azb = m.compute_sparse_stats(4, t, 0)
    print("INT:", num, less, bits, zb, sb, azb)
    # codes: 1=0001, -1(two's compl. low4)=1111, 2=0010, 0=0000
    assert num == 4 and bits == 16
    assert zb == 10, f"zb={zb}"        # 0-bits: 3+0+3+4
    assert azb == 12, f"azb={azb}"     # sign-mag: 3+2+3+4
    assert sb == 14, f"sb={sb}"        # sparse: 3+4+3+4
    print("INT bit stats OK")

    # --- FP bit stats: E4M3 ---
    f = torch.tensor([1.0, 0.5, 0.0])
    num, less, bits, zb, sb, azb = m.compute_sparse_stats_fp("e4m3", f, 0.0)
    print("FP:", num, less, bits, zb, sb, azb)
    # 1.0 -> sm(with hidden 1)=1000 (3 zero bits); 0.5 -> 1000 (3);
    # 0.0 -> 0000 (4). total_bits = 3*4 = 12
    assert num == 3 and bits == 12
    assert zb == 10 and sb == 10 and azb == 10, f"zb={zb}"
    print("FP bit stats OK")

    # --- linear (7 positional args) dispatch ---
    m.reset_sparsity()
    m.collect_quant_activation(
        "q_proj", 0, t, t, QuantSpec(kind="int", bits=4), 4, 16, 16, 32
    )
    assert m.total_element_count == 4 and m.total_bit_count == 16
    assert "q_proj_0" in m.per_layer_sparsity
    print("linear dispatch OK")

    # --- matmul (9 positional args) dispatch: A(e4m3) + B(int8) ---
    m.reset_sparsity()
    A = torch.tensor([[1.0, 0.5], [0.0, -2.0]])
    B = torch.tensor([[1.0, -1.0], [0.0, 2.0]])
    m.collect_quant_activation(
        "qk_matmul", 0, A, A, B,
        QuantSpec(kind="int", bits=8),
        QuantSpec(kind="fp", fmt="e4m3"),
        4, 16, 2, 2,
    )
    # A: 4 elems * 4 bits = 16; B: 4 elems * 8 bits = 32
    assert m.total_element_count == 8, m.total_element_count
    assert m.total_bit_count == 48, m.total_bit_count
    print("matmul dispatch OK")

    # --- phase dispatch ---
    m.reset_sparsity()
    m.set_phase("prefill")
    m.collect_quant_activation(
        "q_proj", 1, t, t, QuantSpec(kind="int", bits=4), 4, 16, 16, 32
    )
    m.set_phase("decode")
    m.collect_quant_activation(
        "q_proj", 1, t, t, QuantSpec(kind="int", bits=4), 4, 16, 16, 32
    )
    assert m.phase_sparsity["prefill"]["total_element_count"] == 4
    assert m.phase_sparsity["decode"]["total_element_count"] == 4
    assert m.phase_sparsity["full_forward"]["total_element_count"] == 0
    print("phase dispatch OK")

    # --- passthrough tiers skipped ---
    m.reset_sparsity()
    m.collect_quant_activation(
        "x_proj", 0, t, t,
        QuantSpec(kind="fp", fmt="e5m10", enabled=False),
        4, 16, 16, 32,
    )
    assert m.total_element_count == 0
    print("passthrough skip OK")

    # --- unit sparsity ---
    # rows: [0,0] -> both bit-groups (b[0:2], b[2:4]) all zero -> 2 zero units
    #       [1,0] -> code 1 = 0001: low group b[0:2] nonzero, high group
    #                b[2:4] = 00|00 all zero -> 1 zero unit
    zu, tu = m.compute_unit_sparsity_int(
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        bits=4, bit_group_size=2, dim_group_size=2,
    )
    print("unit int:", zu, tu)
    assert zu == 3 and tu == 4, f"zu={zu}, tu={tu}"

    # 1.0 -> sm(with hidden 1) = 1000; 0.0 -> 0000.
    # dim group [1.0, 0.0]: low group b[0:2] = 00|00 all zero -> 1 zero unit;
    # high group b[2:4] contains the hidden 1 -> nonzero.
    zu, tu = m.compute_unit_sparsity_fp(
        torch.tensor([[1.0, 0.0]]),
        fmt="e4m3", bit_group_size=2, dim_group_size=2,
    )
    print("unit fp:", zu, tu)
    assert zu == 1 and tu == 2, f"zu={zu}, tu={tu}"
    print("unit sparsity OK")

    # --- default off: bookkeeping only, no sparsity ---
    m2 = QuantStatManager(tempfile.mkdtemp())
    m2.collect_quant_activation(
        "q_proj", 0, t, t, QuantSpec(kind="int", bits=4), 4, 16, 16, 32
    )
    assert m2.total_element_count == 0
    assert m2.quant_activation_calls["q_proj_0"] == 1
    print("default-off OK")

    # --- CSV export ---
    out = tempfile.mkdtemp()
    m.export_per_layer_sparsity_csv(os.path.join(out, "per_layer.csv"))
    m.export_unit_sparsity_csv(os.path.join(out, "unit.csv"), "cfg", "model")
    m.export_collected_layers_csv(os.path.join(out, "layers.csv"), "cfg", "model")
    for fn in ["per_layer.csv", "unit.csv", "layers.csv"]:
        p = os.path.join(out, fn)
        assert os.path.exists(p) and os.path.getsize(p) > 0
        print(f"{fn}: {sum(1 for _ in open(p))} lines")

    # --- reporting prints run without error ---
    m.print_global_sparsity()
    m.print_prefill_decode_sparsity()
    m.print_unit_sparsity_by_phase()
    m.print_collected_layer_names()

    print()
    print("=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main()
