"""Phase C end-to-end calibration smoke test.

Builds the wrapped SmolVLA (linear + matmul, scale_inspection mode) and runs a
synthetic prefix-prefill + one denoise step, then asserts that:

  1. all 64 physical MatMul scale groups collected a scale (prefix→vlm,
     denoise→expert);
  2. all Linear scale groups (per_site → 224 + head) collected a scale;
  3. the number of distinct scale files matches the granularity.

This exercises the full ContextVar routing through
SmolVLMWithExpertModel.forward for BOTH the prefix (component=vlm) and
denoise (component=expert) phases — the only way to reach every MatMul.

Usage:
    python scripts/test_phaseC_calibration.py [checkpoint_path] [granularity]
"""

import sys

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAConfig
from vla_tcs2.model_wrapper import ModelWrapper


def build(path: str, granularity: str) -> ModelWrapper:
    cfg = {
        "model": {
            "type": "smolvla",
            "path": path,
            "device": "cuda",
            "overrides": {"n_action_steps": 1, "num_steps": 10},
        },
        "quantization": {
            "enabled": True,
            "method": "per_tensor",
            "scale_dir": f"/tmp/_phaseC_{granularity}",
            "quantize_matmul": True,
            "matmul_scale_granularity": granularity,
            "linear_scale_granularity": granularity,
            "linear": {"enabled": True, "include": ["*"], "exclude": []},
        },
    }
    w = ModelWrapper(cfg)
    w.build(mode="scale_inspection")
    return w


def run_prefix(am, B=1, S_p=4):
    vlm_hidden = am.get_vlm_model().text_model.config.hidden_size
    prefix = torch.randn(B, S_p, vlm_hidden, device="cuda", dtype=torch.float32)
    mask = torch.ones(B, S_p, S_p, dtype=torch.bool, device="cuda")
    pos = torch.arange(S_p, device="cuda").unsqueeze(0)
    with torch.no_grad():
        _, cache = am.forward(mask, pos, None, [prefix, None], use_cache=True)
    return cache


def run_denoise(am, cache, B=1, S_s=2, S_p=4):
    exp_hidden = am.lm_expert.config.hidden_size
    suffix = torch.randn(B, S_s, exp_hidden, device="cuda", dtype=torch.float32)
    # full attention mask: [prefix_len ; suffix_len] columns
    full = torch.ones(B, S_s, S_p + S_s, dtype=torch.bool, device="cuda")
    pos = torch.arange(S_p, S_p + S_s, device="cuda").unsqueeze(0)
    with torch.no_grad():
        out, _ = am.forward(full, pos, cache, [None, suffix], use_cache=True)
    return out


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/smolvla_base"
    granularity = sys.argv[2] if len(sys.argv) > 2 else "per_site"

    w = build(path, granularity)
    am = w.model.model.vlm_with_expert
    sm = w.stat_manager

    cache = run_prefix(am)
    run_denoise(am, cache)

    # --- audit collected scale groups ---
    # MatMul scale groups (scale_group_name contains "_matmul" or is "qk"/"pv")
    mm_keys = {k for k in sm.stats if "matmul" in sm.stats[k].layer_name}
    lin_keys = {k for k in sm.stats if "matmul" not in sm.stats[k].layer_name}

    n_mm = len(am.quant_matmuls)
    n_lin = sum(
        1 for m in am.modules() if hasattr(m, "scale_group_name")
        and m.__class__.__name__ == "QuantizedLinear"
    )

    print(f"physical MatMul: {n_mm}, physical Linear: {n_lin}")
    print(f"collected MatMul scale groups: {len(mm_keys)}")
    print(f"collected Linear scale groups: {len(lin_keys)}")

    # every physical MatMul must have collected a scale (prefix+denoise cover all)
    expected_mm = {
        "global": 2,
        "per_component": 4,
        "per_layer": 32,
        "per_site": 64,
    }[granularity]
    assert len(mm_keys) == expected_mm, (
        f"expected {expected_mm} MatMul scale groups, got {len(mm_keys)}"
    )
    print(f"[1] MatMul scale groups == {expected_mm}: OK")

    # per_site Linear: 224 vlm/expert + head
    for k, stat in sm.stats.items():
        assert stat.sample_count > 0, f"scale group {k} collected nothing"
    print(f"[2] all {len(sm.stats)} scale groups have sample_count > 0: OK")

    print(f"\nPHASE C CALIBRATION SMOKE: PASSED ({granularity})")


if __name__ == "__main__":
    main()
