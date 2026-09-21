# Vision/Connector quantization routing tests (Phase I vision quantization).
#
# Run:
#   conda run -n smolvla_eval python -m pytest tests/test_vision_quant_routing.py -v

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from vla_tcs2.model_wrapper import (  # noqa: E402
    VISION_ATTN_LINEAR_NAMES,
    VISION_MLP_LINEAR_NAMES,
    _wrap_smolvlm_vision_linear_layers,
)
from vla_tcs2.quant_linear import QuantizedLinear  # noqa: E402
from vla_tcs2.quant.stat_manager import QuantStatManager  # noqa: E402


# ---------------------------------------------------------------------------
# Lightweight mock of the transformers SmolVLM vision/connector structure so
# tests stay unit-level (no model download / GPU).
# ---------------------------------------------------------------------------

N_VISION_LAYERS = 12
HIDDEN = 768


class _FakeVisionAttn(nn.Module):
    def __init__(self):
        super().__init__()
        for name in VISION_ATTN_LINEAR_NAMES:
            setattr(self, name, nn.Linear(HIDDEN, HIDDEN))


class _FakeVisionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        for name in VISION_MLP_LINEAR_NAMES:
            setattr(self, name, nn.Linear(HIDDEN, HIDDEN))


class _FakeVisionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeVisionAttn()
        self.mlp = _FakeVisionMLP()


class _FakeVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeVisionLayer() for _ in range(N_VISION_LAYERS)]
        )


class _FakeVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeVisionEncoder()


class _FakeModalityProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)


class _FakeConnector(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality_projection = _FakeModalityProj()


class _FakeVLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = _FakeVisionModel()
        self.connector = _FakeConnector()


class _FakeVLMExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_model = _FakeVLMModel()

    def get_vlm_model(self):
        return self.vlm_model


class _FakeFlowMatching(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_with_expert = _FakeVLMExpert()


class _FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeFlowMatching()


def _make_policy():
    return _FakePolicy()


def _cfg(
    vision_enabled=False,
    vision_mlp=False,
    vision_attn_proj=False,
    connector_enabled=False,
    overrides=None,
    linear_enabled=True,
    linear_include=None,
    linear_exclude=None,
):
    return {
        "method": "pot_fp8_outlier",
        "scale_dir": tempfile.mkdtemp(),
        "outlier_ratio": 0.01,
        "linear_scale_granularity": "per_site",
        "linear": {
            "enabled": linear_enabled,
            "include": linear_include or ["*"],
            "exclude": linear_exclude or [],
            "overrides": overrides or [],
        },
        "vision": {
            "enabled": vision_enabled,
            "linear": {
                "mlp": vision_mlp,
                "attn_proj": vision_attn_proj,
            },
        },
        "connector": {"enabled": connector_enabled},
    }


def _quantized_count(policy):
    return sum(
        1
        for m in policy.model.modules()
        if isinstance(m, QuantizedLinear)
    )


# ---------------------------------------------------------------------------
# Test 1 — default off (backward-compat gate)
# ---------------------------------------------------------------------------
def test_t1_default_off():
    policy = _make_policy()
    n = _wrap_smolvlm_vision_linear_layers(policy, _cfg(), "scale_inspection", None)
    assert n == 0
    assert _quantized_count(policy) == 0

    # Vision / connector Linear must still be plain nn.Linear.
    vlm = policy.model.vlm_with_expert.get_vlm_model()
    assert isinstance(vlm.vision_model.encoder.layers[0].self_attn.q_proj, nn.Linear)
    assert isinstance(vlm.vision_model.encoder.layers[0].mlp.fc1, nn.Linear)
    assert isinstance(vlm.connector.modality_projection.proj, nn.Linear)


# ---------------------------------------------------------------------------
# Test 2 — connector only (+1)
# ---------------------------------------------------------------------------
def test_t2_connector_only():
    policy = _make_policy()
    n = _wrap_smolvlm_vision_linear_layers(
        policy, _cfg(connector_enabled=True), "scale_inspection", None
    )
    assert n == 1
    assert _quantized_count(policy) == 1

    vlm = policy.model.vlm_with_expert.get_vlm_model()
    proj = vlm.connector.modality_projection.proj
    assert isinstance(proj, QuantizedLinear)
    assert proj.module_id == "connector.layer.0.connector_proj"
    # Vision stays plain.
    assert isinstance(
        vlm.vision_model.encoder.layers[0].self_attn.q_proj, nn.Linear
    )


# ---------------------------------------------------------------------------
# Test 3 — vision.enabled alone wraps NOTHING (explicit group gate, P0)
# ---------------------------------------------------------------------------
def test_t3_enabled_alone_wraps_nothing():
    policy = _make_policy()
    n = _wrap_smolvlm_vision_linear_layers(
        policy, _cfg(vision_enabled=True), "scale_inspection", None
    )
    assert n == 0
    assert _quantized_count(policy) == 0

    vlm = policy.model.vlm_with_expert.get_vlm_model()
    assert isinstance(
        vlm.vision_model.encoder.layers[0].self_attn.q_proj, nn.Linear
    )
    assert isinstance(vlm.vision_model.encoder.layers[0].mlp.fc1, nn.Linear)


# ---------------------------------------------------------------------------
# Test 4 — vision MLP only (12×2 = 24)
# ---------------------------------------------------------------------------
def test_t4_vision_mlp_only():
    policy = _make_policy()
    n = _wrap_smolvlm_vision_linear_layers(
        policy,
        _cfg(vision_enabled=True, vision_mlp=True),
        "scale_inspection",
        None,
    )
    assert n == 24
    assert _quantized_count(policy) == 24

    vlm = policy.model.vlm_with_expert.get_vlm_model()
    layer0 = vlm.vision_model.encoder.layers[0]
    assert isinstance(layer0.mlp.fc1, QuantizedLinear)
    assert isinstance(layer0.mlp.fc2, QuantizedLinear)
    # Attention projection stays plain.
    assert isinstance(layer0.self_attn.q_proj, nn.Linear)
    assert isinstance(layer0.self_attn.out_proj, nn.Linear)


# ---------------------------------------------------------------------------
# Test 5 — vision attention projection only (12×4 = 48)
# ---------------------------------------------------------------------------
def test_t5_vision_attn_proj_only():
    policy = _make_policy()
    n = _wrap_smolvlm_vision_linear_layers(
        policy,
        _cfg(vision_enabled=True, vision_attn_proj=True),
        "scale_inspection",
        None,
    )
    assert n == 48
    assert _quantized_count(policy) == 48

    vlm = policy.model.vlm_with_expert.get_vlm_model()
    layer0 = vlm.vision_model.encoder.layers[0]
    assert isinstance(layer0.self_attn.q_proj, QuantizedLinear)
    assert isinstance(layer0.self_attn.k_proj, QuantizedLinear)
    assert isinstance(layer0.self_attn.v_proj, QuantizedLinear)
    assert isinstance(layer0.self_attn.out_proj, QuantizedLinear)
    # MLP stays plain.
    assert isinstance(layer0.mlp.fc1, nn.Linear)
    assert isinstance(layer0.mlp.fc2, nn.Linear)


# ---------------------------------------------------------------------------
# Test 6 — full vision linear (mlp + attn_proj = 72)
# ---------------------------------------------------------------------------
def test_t6_full_vision_linear():
    policy = _make_policy()
    n = _wrap_smolvlm_vision_linear_layers(
        policy,
        _cfg(vision_enabled=True, vision_mlp=True, vision_attn_proj=True),
        "scale_inspection",
        None,
    )
    assert n == 72
    assert _quantized_count(policy) == 72

    vlm = policy.model.vlm_with_expert.get_vlm_model()
    layer0 = vlm.vision_model.encoder.layers[0]
    assert isinstance(layer0.self_attn.q_proj, QuantizedLinear)
    assert isinstance(layer0.self_attn.out_proj, QuantizedLinear)
    assert isinstance(layer0.mlp.fc1, QuantizedLinear)
    assert isinstance(layer0.mlp.fc2, QuantizedLinear)
    # Connector stays plain.
    assert isinstance(vlm.connector.modality_projection.proj, nn.Linear)


# ---------------------------------------------------------------------------
# Test 7 — vision + connector (72 + 1 = 73)
# ---------------------------------------------------------------------------
def test_t7_vision_plus_connector():
    policy = _make_policy()
    n = _wrap_smolvlm_vision_linear_layers(
        policy,
        _cfg(
            vision_enabled=True,
            vision_mlp=True,
            vision_attn_proj=True,
            connector_enabled=True,
        ),
        "scale_inspection",
        None,
    )
    assert n == 73
    assert _quantized_count(policy) == 73


# ---------------------------------------------------------------------------
# Test 8 — V2 cumulative: connector + vision MLP = 1 + 24 = 25
# ---------------------------------------------------------------------------
def test_t8_v2_connector_plus_mlp():
    policy = _make_policy()
    n = _wrap_smolvlm_vision_linear_layers(
        policy,
        _cfg(
            vision_enabled=True,
            vision_mlp=True,
            connector_enabled=True,
        ),
        "scale_inspection",
        None,
    )
    assert n == 25


# ---------------------------------------------------------------------------
# Test 9 — module IDs are unique and well-formed
# ---------------------------------------------------------------------------
def test_t9_module_ids_unique():
    policy = _make_policy()
    _wrap_smolvlm_vision_linear_layers(
        policy,
        _cfg(
            vision_enabled=True,
            vision_mlp=True,
            vision_attn_proj=True,
            connector_enabled=True,
        ),
        "scale_inspection",
        None,
    )

    ids = [
        m.module_id
        for m in policy.model.modules()
        if isinstance(m, QuantizedLinear)
    ]
    assert len(ids) == 73
    assert len(ids) == len(set(ids)), "duplicate module_id"

    # Spot-check the canonical forms.
    assert "vision.layers.0.self_attn.q_proj" in ids
    assert "vision.layers.0.self_attn.out_proj" in ids
    assert "vision.layers.0.mlp.fc1" in ids
    assert "vision.layers.11.mlp.fc2" in ids
    assert "connector.layer.0.connector_proj" in ids


# ---------------------------------------------------------------------------
# Test 10 — _should_wrap honored: linear.enabled=false disables vision wrap
# ---------------------------------------------------------------------------
def test_t10_linear_disabled_disables_vision():
    policy = _make_policy()
    n = _wrap_smolvlm_vision_linear_layers(
        policy,
        _cfg(
            vision_enabled=True,
            vision_mlp=True,
            vision_attn_proj=True,
            linear_enabled=False,
        ),
        "scale_inspection",
        None,
    )
    assert n == 0
    assert _quantized_count(policy) == 0


# ---------------------------------------------------------------------------
# Test 11 — _should_wrap honored: linear.exclude isolates vision attention
# ---------------------------------------------------------------------------
def test_t11_linear_exclude_isolates_attn():
    policy = _make_policy()
    n = _wrap_smolvlm_vision_linear_layers(
        policy,
        _cfg(
            vision_enabled=True,
            vision_mlp=True,
            vision_attn_proj=True,
            linear_exclude=["vision.layers.*.self_attn.*"],
        ),
        "scale_inspection",
        None,
    )
    # Only MLP (24) wraps; attention (48) is excluded.
    assert n == 24

    vlm = policy.model.vlm_with_expert.get_vlm_model()
    assert isinstance(vlm.vision_model.encoder.layers[0].mlp.fc1, QuantizedLinear)
    assert isinstance(
        vlm.vision_model.encoder.layers[0].self_attn.q_proj, nn.Linear
    )


# ---------------------------------------------------------------------------
# Test 12 — _parse_module_id resolves vision/connector component/operator/idx
# ---------------------------------------------------------------------------
def test_t12_parse_module_id_vision_connector():
    parse = QuantStatManager._parse_module_id

    assert parse("vision.layers.3.mlp.fc1") == ("vision", "fc1", 3)
    assert parse("vision.layers.7.self_attn.out_proj") == (
        "vision", "out_proj", 7,
    )
    assert parse("connector.layer.0.connector_proj") == (
        "connector", "connector_proj", 0,
    )
    assert parse("vlm.layers.5.mlp.down_proj") == ("vlm", "down_proj", 5)
    assert parse("expert.layers.1.self_attn.q_proj") == ("expert", "q_proj", 1)


# ---------------------------------------------------------------------------
# Test 13 — linear.overrides resolve vision/connector precision via module_id
# ---------------------------------------------------------------------------
def test_t13_overrides_target_vision():
    from vla_tcs2.model_wrapper import resolve_linear_quant_config

    cfg = _cfg(overrides=[{
        "name": "vision_mlp_fp8",
        "target": {"module_id": "vision.layers.*.mlp.*"},
        "config": {"w_bit": "e4m3", "method": "pot_fp8_outlier"},
    }])

    # Vision MLP fc1 -> overridden.
    resolved, names = resolve_linear_quant_config(
        cfg, "fc1", "vision.layers.4.mlp.fc1"
    )
    assert resolved["w_bit"] == "e4m3"
    assert names == ["vision_mlp_fp8"]

    # Vision attention q_proj -> not overridden.
    resolved2, names2 = resolve_linear_quant_config(
        cfg, "q_proj", "vision.layers.4.self_attn.q_proj"
    )
    assert names2 == []

    # VLM q_proj -> not overridden (different component).
    resolved3, names3 = resolve_linear_quant_config(
        cfg, "q_proj", "vlm.layers.4.self_attn.q_proj"
    )
    assert names3 == []


# ---------------------------------------------------------------------------
# Test 14 — vision/connector calibration policy defaults to recalibrate
# ---------------------------------------------------------------------------
def test_t14_calibration_policy_recalibrate():
    policy = _make_policy()
    _wrap_smolvlm_vision_linear_layers(
        policy,
        _cfg(
            vision_enabled=True,
            vision_mlp=True,
            connector_enabled=True,
        ),
        "scale_inspection",
        None,
    )

    vlm = policy.model.vlm_with_expert.get_vlm_model()
    fc1 = vlm.vision_model.encoder.layers[0].mlp.fc1
    proj = vlm.connector.modality_projection.proj
    assert fc1.calibration_policy == "recalibrate"
    assert proj.calibration_policy == "recalibrate"


# ---------------------------------------------------------------------------
# Test 15 — full Vision Linear path: all 72 sites force recalibration
# ---------------------------------------------------------------------------
def test_t15_full_vision_linear_all_recalibrate():
    policy = _make_policy()
    _wrap_smolvlm_vision_linear_layers(
        policy,
        _cfg(
            vision_enabled=True,
            vision_mlp=True,
            vision_attn_proj=True,
        ),
        "scale_inspection",
        None,
    )

    vision = [
        m
        for m in policy.model.modules()
        if isinstance(m, QuantizedLinear)
        and m.module_id.startswith("vision.")
    ]
    assert len(vision) == 72
    assert all(m.calibration_policy == "recalibrate" for m in vision)


if __name__ == "__main__":
    test_t1_default_off()
    test_t2_connector_only()
    test_t3_enabled_alone_wraps_nothing()
    test_t4_vision_mlp_only()
    test_t5_vision_attn_proj_only()
    test_t6_full_vision_linear()
    test_t7_vision_plus_connector()
    test_t8_v2_connector_plus_mlp()
    test_t9_module_ids_unique()
    test_t10_linear_disabled_disables_vision()
    test_t11_linear_exclude_isolates_attn()
    test_t12_parse_module_id_vision_connector()
    test_t13_overrides_target_vision()
    test_t14_calibration_policy_recalibrate()
    test_t15_full_vision_linear_all_recalibrate()
    print("all vision quant routing tests passed")
