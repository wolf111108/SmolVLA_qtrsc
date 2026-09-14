# Phase H sparsity accounting tests (CODE_MODIFICATION_PLAN.md §10, T1-T6).
#
# Run (offline, no model needed):
#   conda run -n smolvla_eval python -m pytest tests/test_sparsity_accounting.py -v
# or standalone:
#   conda run -n smolvla_eval python tests/test_sparsity_accounting.py

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from vla_tcs2.quant.quant_spec import QuantSpec  # noqa: E402
from vla_tcs2.quant.stat_manager import QuantStatManager  # noqa: E402
from vla_tcs2.runtime_context import (  # noqa: E402
    CURRENT_PHASE,
    CURRENT_FLOW_STEP,
    CURRENT_ATTN_KIND,
)


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _ctx(phase="prefill", flow_step=-1, attention_kind="unknown"):
    return {
        "phase": phase,
        "flow_step": flow_step,
        "attention_kind": attention_kind,
        "generation_id": -1,
    }


def _module_rows(rows, module_id, tensor_role):
    return [
        r
        for r in rows
        if r["module_id"] == module_id and r["tensor_role"] == tensor_role
    ]


# ---------------------------------------------------------------------------
# T1: no-outlier backward compatibility (reported == native)
# ---------------------------------------------------------------------------
def test_t1_no_outlier_reported_equals_native():
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    spec = QuantSpec(kind="int", bits=4)
    # 4 elements, one zero (reported zero = 1). No outlier partition -> 0.
    t = torch.tensor([1.0, -1.0, 2.0, 0.0])
    m.collect_quant_tensor(
        module_id="vlm.layers.0.mlp.down_proj",
        tensor_role="activation",
        tensor_code=t,
        spec=spec,
        runtime_context=_ctx(),
    )

    path = os.path.join(tempfile.mkdtemp(), "module_sparsity.csv")
    m.export_module_sparsity_csv(path)
    rows = _read_csv(path)
    row = rows[0]

    assert int(row["total_elements_reported"]) == 4
    assert int(row["zero_elements_reported"]) == 1
    assert int(row["protected_elements"]) == 0
    assert int(row["total_elements_native"]) == 4
    assert int(row["zero_elements_native"]) == 1
    assert row["zero_rate_reported"] == row["zero_rate_native"]


# ---------------------------------------------------------------------------
# T2: artificial outlier mask correction (native excludes protected zeros)
# ---------------------------------------------------------------------------
def test_t2_outlier_mask_native_correction():
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    spec = QuantSpec(kind="int", bits=4)
    # 8 codes: 1 real zero (index 3) + 2 protected positions forced to 0
    # (indices 6,7) => reported zero = 3, protected = 2.
    t = torch.tensor([1.0, -1.0, 2.0, 0.0, 3.0, -3.0, 0.0, 0.0])
    m.collect_quant_tensor(
        module_id="expert.layers.1.mlp.up_proj",
        tensor_role="output",
        tensor_code=t,
        spec=spec,
        runtime_context=_ctx(),
    )
    m.collect_outlier_partition(
        module_id="expert.layers.1.mlp.up_proj",
        tensor_role="output",
        total_elements=8,
        protected_elements=2,
        spec=spec,
        runtime_context=_ctx(),
    )

    path = os.path.join(tempfile.mkdtemp(), "module_sparsity.csv")
    m.export_module_sparsity_csv(path)
    rows = _read_csv(path)
    row = rows[0]

    assert int(row["total_elements_reported"]) == 8
    assert int(row["zero_elements_reported"]) == 3
    assert int(row["protected_elements"]) == 2
    assert int(row["total_elements_native"]) == 6
    assert int(row["zero_elements_native"]) == 1  # real_zero only
    # bit-level: 8*4 = 32 reported bits; 2*4 = 8 protected bits -> 24 native.
    assert int(row["total_bits_reported"]) == 32
    assert int(row["protected_bits"]) == 8
    assert int(row["total_bits_native"]) == 24


# ---------------------------------------------------------------------------
# T3: weighted aggregation (sum numerators/denominators, never average ratios)
# ---------------------------------------------------------------------------
def test_t3_weighted_aggregation():
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    spec = QuantSpec(kind="int", bits=4)
    mid = "vlm.layers.0.mlp.gate_proj"

    # call 1: 10 elements, 1 zero
    m.collect_quant_tensor(
        module_id=mid, tensor_role="activation",
        tensor_code=torch.tensor([1.0] * 9 + [0.0]), spec=spec,
        runtime_context=_ctx(),
    )
    # call 2: 90 elements, 45 zeros
    m.collect_quant_tensor(
        module_id=mid, tensor_role="activation",
        tensor_code=torch.tensor([0.0] * 45 + [1.0] * 45), spec=spec,
        runtime_context=_ctx(),
    )

    path = os.path.join(tempfile.mkdtemp(), "module_sparsity.csv")
    m.export_module_sparsity_csv(path)
    row = _module_rows(_read_csv(path), mid, "activation")[0]

    assert int(row["total_elements_reported"]) == 100
    assert int(row["zero_elements_reported"]) == 46
    # 46/100 = 0.46, NOT (10% + 50%)/2 = 0.30
    assert abs(float(row["zero_rate_reported"]) - 0.46) < 1e-9


# ---------------------------------------------------------------------------
# T4: flow-step separation
# ---------------------------------------------------------------------------
def test_t4_flow_step_separation():
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    spec = QuantSpec(kind="int", bits=4)
    mid = "expert.layer.2.self_attn.q_proj"

    m.collect_quant_tensor(
        module_id=mid, tensor_role="activation",
        tensor_code=torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        spec=spec, runtime_context=_ctx(phase="denoise", flow_step=0),
    )
    m.collect_quant_tensor(
        module_id=mid, tensor_role="activation",
        tensor_code=torch.tensor([0.0] * 9 + [1.0]),
        spec=spec, runtime_context=_ctx(phase="denoise", flow_step=1),
    )

    path = os.path.join(tempfile.mkdtemp(), "module_sparsity.csv")
    m.export_module_sparsity_csv(path)
    rows = _module_rows(_read_csv(path), mid, "activation")

    flow_steps = sorted(int(r["flow_step"]) for r in rows)
    assert flow_steps == [0, 1], flow_steps
    by_step = {int(r["flow_step"]): r for r in rows}
    # step 0: 2/10 = 0.2 ; step 1: 9/10 = 0.9 (NOT merged to 50%).
    assert abs(float(by_step[0]["zero_rate_reported"]) - 0.2) < 1e-9
    assert abs(float(by_step[1]["zero_rate_reported"]) - 0.9) < 1e-9


# ---------------------------------------------------------------------------
# T5: unit role separation (MatMul A/B/O distinct unit rows)
# ---------------------------------------------------------------------------
def test_t5_unit_role_separation():
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    spec = QuantSpec(kind="fp", fmt="e4m3")
    mid = "expert.layer.3.qk"

    d = CURRENT_PHASE.set("denoise")
    s = CURRENT_FLOW_STEP.set(2)
    k = CURRENT_ATTN_KIND.set("cross")
    for role in ("A", "B", "O"):
        m.collect_quant_tensor(
            module_id=mid, tensor_role=role,
            tensor_code=torch.eye(2), spec=spec,
        )
    CURRENT_ATTN_KIND.reset(k)
    CURRENT_FLOW_STEP.reset(s)
    CURRENT_PHASE.reset(d)

    path = os.path.join(tempfile.mkdtemp(), "unit_sparsity.csv")
    m.export_unit_sparsity_structured_csv(path)
    rows = _read_csv(path)

    roles = sorted(r["tensor_role"] for r in rows)
    assert roles == ["A", "B", "O"], roles
    for r in rows:
        assert r["attention_kind"] == "cross"
        assert int(r["total_units"]) > 0


# ---------------------------------------------------------------------------
# T6: collector must not mutate the input tensor
# ---------------------------------------------------------------------------
def test_t6_collector_does_not_mutate_input():
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    spec = QuantSpec(kind="int", bits=4)
    t = torch.tensor([1.0, -1.0, 2.0, 0.0])
    t_before = t.clone()

    m.collect_quant_tensor(
        module_id="vlm.layers.0.mlp.up_proj",
        tensor_role="activation",
        tensor_code=t,
        spec=spec,
        runtime_context=_ctx(),
    )

    assert torch.equal(t, t_before), "collector mutated the input tensor"


# ---------------------------------------------------------------------------
# T7: export methods (manifest + workload) do not crash on a wrapped model
# ---------------------------------------------------------------------------
def test_t7_export_manifest_and_workload():
    import torch.nn as nn

    class FakeLinear(nn.Module):
        def __init__(self, module_id):
            super().__init__()
            self.module_id = module_id
            self.layer_name = module_id.split(".")[-1]
            self.layer_idx = 0
            self.a_spec = QuantSpec(kind="int", bits=8)
            self.w_spec = QuantSpec(kind="int", bits=4)
            self.o_spec = QuantSpec(kind="fp", fmt="e4m3")
            self.a_bit = 8
            self.w_bit = 4
            self.o_bit = "e4m3"
            self.method = "pot_ao_outlier"
            self.outlier_ratio = 0.01
            self.in_features = 64
            self.out_features = 64

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = FakeLinear("vlm.layers.0.mlp.down_proj")

    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    model = FakeModel()

    tmp = tempfile.mkdtemp()
    n = m.export_quantization_manifest_csv(
        model, os.path.join(tmp, "quantization_manifest.csv")
    )
    assert n == 1, n

    rows = _read_csv(os.path.join(tmp, "quantization_manifest.csv"))
    assert rows[0]["component"] == "vlm"
    assert rows[0]["operator"] == "down_proj"
    assert rows[0]["method"] == "pot_ao_outlier"

    n2 = m.export_workload_csv(model, os.path.join(tmp, "workload.csv"))
    assert n2 == 0  # no collected per-role records -> no workload rows


# ---------------------------------------------------------------------------
# T8: collect_quant_activation is audit-only (no double collection)
# ---------------------------------------------------------------------------
def test_t8_no_double_collection_between_legacy_and_structured():
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    spec = QuantSpec(kind="int", bits=4)
    x = torch.tensor([1.0, 0.0, 1.0, 1.0])
    mid = "expert.layers.0.mlp.up_proj"

    # legacy audit call (must NOT feed structured sparsity anymore)
    m.collect_quant_activation(
        "expert.layers.0.mlp.up_proj",
        0,
        x,          # x_code
        x,          # x_fp16
        spec,       # a_spec
        4,          # digit_size
        1,          # parallelism
        4,          # in_features
        4,          # out_features
    )

    # the single real structured collection
    m.collect_quant_tensor(
        module_id=mid,
        tensor_role="activation",
        tensor_code=x,
        spec=spec,
        runtime_context=_ctx(),
    )

    path = os.path.join(tempfile.mkdtemp(), "module_sparsity.csv")
    m.export_module_sparsity_csv(path)
    rows = _module_rows(_read_csv(path), mid, "activation")
    assert len(rows) == 1, rows

    row = rows[0]
    # one zero element (index 1) out of 4, collected exactly once.
    assert int(row["calls"]) == 1, row["calls"]
    assert int(row["total_elements_reported"]) == 4
    assert int(row["zero_elements_reported"]) == 1


# ---------------------------------------------------------------------------
# T9: runtime hook must tag denoise steps 0..9 (not 0,0,...,0)
# ---------------------------------------------------------------------------
def test_t9_runtime_hook_flow_step_sequencing():
    from vla_tcs2.model_wrapper import install_runtime_hooks
    from vla_tcs2.runtime_context import get_runtime_context

    observed = []

    class FlowModel:
        def __init__(self):
            self._runtime_hooks_installed = False

        def sample_actions(self, *args, **kwargs):
            for _ in range(10):
                self.denoise_step()

        def denoise_step(self, *args, **kwargs):
            observed.append(get_runtime_context()["flow_step"])

    class Model:
        def __init__(self):
            self.model = FlowModel()

    model = Model()
    installed = install_runtime_hooks(model)
    assert installed is True

    model.model.sample_actions()
    model.model.sample_actions()

    # Two sample_actions calls -> two full 0..9 sequences.
    assert observed == list(range(10)) + list(range(10)), observed

    # Re-install is a no-op (marker present).
    assert install_runtime_hooks(model) is False


# ---------------------------------------------------------------------------
# T10: E4M3 significand encoding (+0/-0/subnormal hidden-bit semantics)
# ---------------------------------------------------------------------------
def test_t10_e4m3_significand_encoding():
    m = QuantStatManager(tempfile.mkdtemp())

    raw = torch.tensor([
        0x00,  # +0
        0x80,  # -0
        0x38,  # +1.0
        0xB8,  # -1.0
        0x01,  # +smallest subnormal
        0x81,  # -smallest subnormal
    ], dtype=torch.int64)

    sig, width = m._extract_sm_from_raw(raw, "e4m3")

    assert width == 4
    assert sig.tolist() == [
        0b0000,  # +0
        0b0000,  # -0 (sign must NOT leak)
        0b1000,  # +1.0 (hidden 1)
        0b1000,  # -1.0 (sign-agnostic)
        0b0001,  # +subnormal (NO hidden 1)
        0b0001,  # -subnormal (NO hidden 1, sign-agnostic)
    ]


# ---------------------------------------------------------------------------
# T11: quant_forward_with_outlier must NOT mutate the output code in place
# (H1-Audit regression). The dequant step used `.mul_(M_q)` which
# in-place rewrote out_normal_quant from quantized code ([-448,448]) into
# the dequantized magnitude, so the collected output_code had ~98% values
# > 448 that round to E4M3 NaN and diluted sparse_bit_rate to ~0.5%.
# ---------------------------------------------------------------------------
def test_t11_output_code_not_mutated_by_dequant():
    from vla_tcs2.quant_linear import QuantizedLinear
    from vla_tcs2.quant.quant_methods import quant_forward_with_outlier

    torch.manual_seed(0)

    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()

    layer = QuantizedLinear(
        in_features=16,
        out_features=16,
        a_bit="e4m3",
        w_bit="e4m3",
        o_bit="e4m3",
        outlier_ratio=0.01,
    )
    layer.set_layer_info(
        "q_proj", 0, module_id="vlm.layers.0.self_attn.q_proj"
    )
    # Power-of-two scales (pot_fp8_outlier requirement).
    layer.a_interval = 0.5
    layer.w_interval = 0.5
    layer.o_interval = 0.5
    layer.weight.data = torch.randn(16, 16) * 0.1
    layer.bias.data = torch.randn(16) * 0.1

    x = torch.randn(32, 16)

    # Spy on the collector to capture the exact `output` code tensor that
    # gets passed to collect_quant_tensor (the post-quant FP8 code).
    captured_codes = []
    _orig_collect = m.collect_quant_tensor

    def _spy_collect(**kwargs):
        if kwargs.get("tensor_role") == "output":
            captured_codes.append(kwargs["tensor_code"].detach().clone())
        return _orig_collect(**kwargs)

    m.collect_quant_tensor = _spy_collect

    out = quant_forward_with_outlier(layer, x, stat_collector=m)

    # Forward output must stay finite (dequant is unchanged).
    assert not torch.isnan(out).any(), "forward output has NaN"

    # Direct invariant: the collected output code must be the quantized
    # FP8 code — all finite and within E4M3 representable range [-448, 448].
    # If `mul_` had mutated it in place, it would be dequant magnitudes
    # (e.g. ±294912) that round to NaN.
    assert captured_codes, "no output code captured"
    captured = captured_codes[0].to(torch.float32)
    assert torch.isfinite(captured).all(), "collected output code has NaN/Inf"
    assert captured.abs().max() <= 448.0, (
        f"collected output code max abs={captured.abs().max().item()} > 448; "
        "out_normal_quant was mutated in place"
    )

    # Second-layer sanity: significand sparse-bit rate must be a real
    # ~10-80% value, NOT the ~1% dilution from NaN.
    out_records = [
        v
        for k, v in m.per_role_sparsity.items()
        if k[0] == "vlm.layers.0.self_attn.q_proj" and k[3] == "output"
    ]
    assert out_records, "output role record missing"
    role = out_records[0]
    total_bits = role["total_bits"]
    sparse_bits = role["sparse_bits"]
    rate = sparse_bits / total_bits if total_bits else 0.0
    assert 0.10 < rate < 0.80, (
        f"output sparse_bit_rate={rate:.3f} out of expected range; "
        "out_normal_quant was mutated in place"
    )


def _main():
    test_t1_no_outlier_reported_equals_native()
    test_t2_outlier_mask_native_correction()
    test_t3_weighted_aggregation()
    test_t4_flow_step_separation()
    test_t5_unit_role_separation()
    test_t6_collector_does_not_mutate_input()
    test_t7_export_manifest_and_workload()
    test_t8_no_double_collection_between_legacy_and_structured()
    test_t9_runtime_hook_flow_step_sequencing()
    test_t10_e4m3_significand_encoding()
    test_t11_output_code_not_mutated_by_dequant()
    print("all sparsity accounting tests passed")


if __name__ == "__main__":
    _main()
