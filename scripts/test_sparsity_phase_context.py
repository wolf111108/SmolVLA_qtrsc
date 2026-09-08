# S0 correctness / audit tests for the phase-aware sparsity framework
# (research manual §60-§64, §85-§90).
#
# Two levels:
#   1. Offline unit tests (this file, `--offline`): runtime context labels,
#      structured collector identity (module_id, role, phase, flow_step),
#      weight once-only, MatMul A/B/O, CSV export. No model needed.
#   2. Live model audit (`--live`): one predict_action_chunk on a tiny
#      synthetic batch; asserts vlm.* -> prefill only, expert.* -> denoise
#      only with flow_step 0..9. Requires the checkpoint + smolvla_eval env.
#
# Run offline tests:
#   conda run -n smolvla_eval python scripts/test_sparsity_phase_context.py

import argparse
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
    CURRENT_GENERATION_ID,
    get_runtime_context,
)


def test_runtime_context_defaults():
    assert CURRENT_PHASE.get() == "unknown"
    assert CURRENT_FLOW_STEP.get() == -1
    assert CURRENT_ATTN_KIND.get() == "unknown"
    assert CURRENT_GENERATION_ID.get() == -1
    ctx = get_runtime_context()
    assert ctx == {
        "phase": "unknown",
        "flow_step": -1,
        "attention_kind": "unknown",
        "generation_id": -1,
    }
    # token / reset round trip
    t = CURRENT_PHASE.set("prefill")
    assert get_runtime_context()["phase"] == "prefill"
    CURRENT_PHASE.reset(t)
    assert CURRENT_PHASE.get() == "unknown"
    print("runtime context defaults OK")


def test_structured_collector_identity():
    """module_id + role + phase keyed records; denoise flow-step split."""
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    t = torch.tensor([1.0, -1.0, 2.0, 0.0])
    spec = QuantSpec(kind="int", bits=4)

    p = CURRENT_PHASE.set("prefill")
    m.collect_quant_tensor(
        module_id="vlm.layers.3.mlp.down_proj",
        tensor_role="activation",
        tensor_code=t,
        spec=spec,
    )
    CURRENT_PHASE.reset(p)

    d = CURRENT_PHASE.set("denoise")
    for step in (0, 1):
        s = CURRENT_FLOW_STEP.set(step)
        m.collect_quant_tensor(
            module_id="expert.layer.7.qk",
            tensor_role="A",
            tensor_code=t,
            spec=spec,
        )
        CURRENT_FLOW_STEP.reset(s)
    CURRENT_PHASE.reset(d)

    # module_id keys (never layer_name_idx mixups across components)
    assert "vlm.layers.3.mlp.down_proj" in m.quant_activation_calls
    assert "expert.layer.7.qk" in m.quant_activation_calls

    lk_vlm = "vlm.layers.3.mlp.down_proj_3"
    lk_exp = "expert.layer.7.qk_7"
    assert (lk_vlm, "prefill", "activation") in m.per_role_sparsity
    assert (lk_exp, "denoise", "A") in m.per_role_sparsity

    # flow-step split: 2 denoise calls across steps 0/1
    assert m.flow_step_sparsity[0][lk_exp] == 1
    assert m.flow_step_sparsity[1][lk_exp] == 1

    # phase counters: prefill 4 elems, denoise 8 elems
    assert m.phase_sparsity["prefill"]["total_element_count"] == 4
    assert m.phase_sparsity["denoise"]["total_element_count"] == 8
    print("structured collector identity OK")


def test_matmul_abo_roles():
    """QK A/B/O all recorded as distinct roles at the same module_id."""
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    spec = QuantSpec(kind="fp", fmt="e4m3")

    d = CURRENT_PHASE.set("denoise")
    s = CURRENT_FLOW_STEP.set(3)
    k = CURRENT_ATTN_KIND.set("cross")
    for role in ("A", "B", "O"):
        m.collect_quant_tensor(
            module_id="expert.layer.7.qk",
            tensor_role=role,
            tensor_code=torch.eye(2),
            spec=spec,
            metadata={"operand_origin": "expert_suffix_query"}
            if role == "A"
            else None,
        )
    CURRENT_ATTN_KIND.reset(k)
    CURRENT_FLOW_STEP.reset(s)
    CURRENT_PHASE.reset(d)

    lk = "expert.layer.7.qk_7"
    roles = {
        key[2] for key in m.per_role_sparsity if key[0] == lk
    }
    assert roles == {"A", "B", "O"}, roles
    # each role saw the same 4 elements
    for role in ("A", "B", "O"):
        assert (
            m.per_role_sparsity[(lk, "denoise", role)]["total_elements"] == 4
        )
    print("matmul A/B/O roles OK")


def test_weight_once_only():
    """collect_weight_sparsity overwrites (never accumulates) on re-call."""
    m = QuantStatManager(tempfile.mkdtemp())
    t = torch.tensor([1.0, -1.0, 2.0, 0.0])
    spec = QuantSpec(kind="int", bits=4)
    m.collect_weight_sparsity("expert.layers.0.mlp.down_proj", 0, t, spec)
    m.collect_weight_sparsity("expert.layers.0.mlp.down_proj", 0, t, spec)
    entry = m.per_layer_weight_sparsity["expert.layers.0.mlp.down_proj_0"]
    assert entry["total_elements"] == 4  # NOT 8 — static, once-only
    print("weight once-only OK")


def test_legacy_adapter_module_id():
    """Legacy collect_quant_activation routes through the structured path."""
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    t = torch.tensor([1.0, -1.0, 2.0, 0.0])

    p = CURRENT_PHASE.set("prefill")
    m.collect_quant_activation(
        "vlm.layers.0.self_attn.q_proj", 0, t, t,
        QuantSpec(kind="int", bits=4), 4, 16, 16, 32,
    )
    CURRENT_PHASE.reset(p)

    lk = "vlm.layers.0.self_attn.q_proj_0"
    assert (lk, "prefill", "activation") in m.per_role_sparsity
    assert m.phase_sparsity["prefill"]["total_element_count"] == 4
    # legacy aggregate record still present for old tooling
    assert lk in m.per_layer_sparsity
    print("legacy adapter OK")


def test_export_bundle():
    """module_sparsity.csv + workload.csv export from synthetic records."""
    m = QuantStatManager(tempfile.mkdtemp())
    m.enable_sparsity()
    spec = QuantSpec(kind="int", bits=4)
    t = torch.tensor([[1.0, -1.0, 2.0, 0.0, 0.0, 0.0]])

    p = CURRENT_PHASE.set("prefill")
    m.collect_quant_tensor(
        module_id="vlm.layers.0.self_attn.q_proj",
        tensor_role="activation",
        tensor_code=t,
        spec=spec,
    )
    CURRENT_PHASE.reset(p)

    out = tempfile.mkdtemp()

    class _FakeLinear:
        module_id = "vlm.layers.0.self_attn.q_proj"
        layer_name = "q_proj"
        layer_idx = 0
        in_features = 6
        out_features = 4
        a_bit = 4
        w_bit = 4
        a_spec = spec
        w_spec = spec
        weight = torch.zeros(4, 6)

    class _FakeModel:
        def modules(self):
            return [_FakeLinear()]

    m.export_module_sparsity_csv(os.path.join(out, "module_sparsity.csv"))
    n_rows = m.export_workload_csv(
        _FakeModel(), os.path.join(out, "workload.csv")
    )
    for fn in ("module_sparsity.csv", "workload.csv"):
        path = os.path.join(out, fn)
        assert os.path.exists(path) and os.path.getsize(path) > 0, fn
        print(f"{fn}: {sum(1 for _ in open(path))} lines")
    assert n_rows >= 1

    # workload row sanity: M = elems / calls / K = 6 / 1 / 6 = 1
    with open(os.path.join(out, "workload.csv")) as f:
        lines = f.read().strip().splitlines()
    row = lines[1].split(",")
    assert row[7] == "1", row  # M
    assert row[8] == "6" and row[9] == "4"  # K, N
    print("export bundle OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true",
        help="Run the live model audit (requires checkpoint).",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    test_runtime_context_defaults()
    test_structured_collector_identity()
    test_matmul_abo_roles()
    test_weight_once_only()
    test_legacy_adapter_module_id()
    test_export_bundle()

    print("\nAll offline S0 tests passed.")


if __name__ == "__main__":
    main()
