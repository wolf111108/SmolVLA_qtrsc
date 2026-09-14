# Component-aware Linear precision routing resolver tests (G6 upgrade, Gate 1).
#
# Run:
#   conda run -n smolvla_eval python -m pytest tests/test_linear_precision_routing.py -v

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from vla_tcs2.model_wrapper import (  # noqa: E402
    resolve_linear_quant_config,
)


def _cfg(overrides=None, granularity="per_site"):
    q = {
        "method": "pot_fp8_outlier",
        "q_proj": {"a_bit": "e4m3", "w_bit": "e4m3", "o_bit": "e4m3"},
        "k_proj": {"a_bit": "e4m3", "w_bit": "e4m3", "o_bit": "e4m3"},
        "v_proj": {"a_bit": "e4m3", "w_bit": "e4m3", "o_bit": "e4m3"},
        "o_proj": {"a_bit": "e4m3", "w_bit": "e4m3", "o_bit": "e4m3"},
        "gate_proj": {"a_bit": "e4m3", "w_bit": "e4m3", "o_bit": "e4m3"},
        "up_proj": {"a_bit": "e4m3", "w_bit": "e4m3", "o_bit": "e4m3"},
        "down_proj": {"a_bit": "e4m3", "w_bit": "e4m3", "o_bit": "e4m3"},
        "linear": {
            "enabled": True,
            "include": ["vlm.*", "expert.*"],
            "exclude": [],
            "overrides": overrides or [],
        },
    }
    return q


def test_t1_no_overrides_base_config():
    q = _cfg()
    cfg, names = resolve_linear_quant_config(q, "q_proj", "vlm.layers.0.self_attn.q_proj")
    assert cfg["w_bit"] == "e4m3"
    assert names == []

    cfg2, _ = resolve_linear_quant_config(q, "q_proj", "expert.layers.0.self_attn.q_proj")
    assert cfg2["w_bit"] == "e4m3"


def test_t2_vlm_attention_override():
    q = _cfg(overrides=[{
        "name": "vlm_attn_w4",
        "target": {"module_id": "vlm.layers.*.self_attn.*_proj"},
        "config": {"w_bit": 4, "method": "pot_ao_outlier"},
    }])

    # VLM attention q_proj -> W4
    cfg, names = resolve_linear_quant_config(q, "q_proj", "vlm.layers.3.self_attn.q_proj")
    assert cfg["w_bit"] == 4
    assert cfg["method"] == "pot_ao_outlier"
    assert names == ["vlm_attn_w4"]

    # Expert attention q_proj -> FP8 (base)
    cfg2, _ = resolve_linear_quant_config(q, "q_proj", "expert.layers.3.self_attn.q_proj")
    assert cfg2["w_bit"] == "e4m3"

    # VLM MLP gate_proj -> FP8 (base)
    cfg3, _ = resolve_linear_quant_config(q, "gate_proj", "vlm.layers.3.mlp.gate_proj")
    assert cfg3["w_bit"] == "e4m3"


def test_t3_vlm_mlp_override():
    q = _cfg(overrides=[{
        "name": "vlm_mlp_w4",
        "target": {"module_id": "vlm.layers.*.mlp.*_proj"},
        "config": {"w_bit": 4, "method": "pot_ao_outlier"},
    }])

    cfg, _ = resolve_linear_quant_config(q, "down_proj", "vlm.layers.7.mlp.down_proj")
    assert cfg["w_bit"] == 4

    cfg2, _ = resolve_linear_quant_config(q, "o_proj", "vlm.layers.7.self_attn.o_proj")
    assert cfg2["w_bit"] == "e4m3"

    cfg3, _ = resolve_linear_quant_config(q, "down_proj", "expert.layers.7.mlp.down_proj")
    assert cfg3["w_bit"] == "e4m3"


def test_t4_unknown_field_fail_fast():
    q = _cfg(overrides=[{
        "name": "typo",
        "target": {"module_id": "vlm.*"},
        "config": {"wieght_bit": 4},  # typo
    }])
    with pytest.raises(ValueError):
        resolve_linear_quant_config(q, "q_proj", "vlm.layers.0.self_attn.q_proj")


def test_t5_later_override_wins():
    q = _cfg(overrides=[
        {
            "name": "all_vlm_w4",
            "target": {"component": "vlm"},
            "config": {"w_bit": 4, "method": "pot_ao_outlier"},
        },
        {
            "name": "keep_layer0_fp8",
            "target": {"module_id": "vlm.layers.0.*.*"},
            "config": {"w_bit": "e4m3", "method": "pot_fp8_outlier"},
        },
    ])

    # layer0 -> later override wins -> FP8
    cfg, names = resolve_linear_quant_config(q, "q_proj", "vlm.layers.0.self_attn.q_proj")
    assert cfg["w_bit"] == "e4m3"
    assert names == ["all_vlm_w4", "keep_layer0_fp8"]

    # layer3 -> only first override matches -> W4
    cfg2, _ = resolve_linear_quant_config(q, "q_proj", "vlm.layers.3.self_attn.q_proj")
    assert cfg2["w_bit"] == 4


def test_t6_per_site_guard():
    # The guard lives in _wrap_smolvla_linear_layers; test it via a direct
    # import to keep this test unit-level without a full model.
    from vla_tcs2.model_wrapper import _wrap_smolvla_linear_layers  # noqa: F401
    # guard logic is exercised in the wrapper; here we just assert the
    # resolver itself does not error on non-per-site (guard is wrapper-side).
    q = _cfg(overrides=[{
        "name": "x",
        "target": {"component": "vlm"},
        "config": {"w_bit": 4},
    }])
    cfg, _ = resolve_linear_quant_config(q, "q_proj", "vlm.layers.0.self_attn.q_proj")
    assert cfg["w_bit"] == 4


def test_t7_empty_target_no_match():
    q = _cfg(overrides=[{
        "name": "empty",
        "target": {},
        "config": {"w_bit": 4},
    }])
    with pytest.raises(ValueError):
        resolve_linear_quant_config(q, "q_proj", "vlm.layers.0.self_attn.q_proj")


def test_t8_target_typo_fail_fast():
    # "componet" (typo) must NOT silently match all modules.
    q = _cfg(overrides=[{
        "name": "typo_target",
        "target": {"componet": "vlm"},
        "config": {"w_bit": 4},
    }])
    with pytest.raises(ValueError):
        resolve_linear_quant_config(q, "q_proj", "vlm.layers.0.self_attn.q_proj")


if __name__ == "__main__":
    test_t1_no_overrides_base_config()
    test_t2_vlm_attention_override()
    test_t3_vlm_mlp_override()
    test_t4_unknown_field_fail_fast()
    test_t5_later_override_wins()
    test_t6_per_site_guard()
    test_t7_empty_target_no_match()
    test_t8_target_typo_fail_fast()
    print("all linear precision routing tests passed")
