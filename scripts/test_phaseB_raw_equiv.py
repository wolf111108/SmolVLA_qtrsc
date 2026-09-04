"""Phase B raw-equivalence + 64-MatMul routing smoke test.

Verifies, without a full LIBERO rollout:
  1. 64 physical QuantizedMatMul objects are created (vlm/expert x 16 x qk/pv).
  2. module_id is unique per MatMul; scale_group follows the granularity.
  3. The ContextVar-routed eager attention interface is bit-exact with the
     original (unquantized) eager_attention_forward under mode="raw".

Usage:
    python scripts/test_phaseB_raw_equiv.py [checkpoint_path] [granularity]
"""

import sys

import torch

from vla_tcs2.model_wrapper import (
    ModelWrapper,
    _CURRENT_ATTN_SITE,
)


def build_wrapper(path: str, granularity: str) -> ModelWrapper:
    cfg = {
        "model": {
            "type": "smolvla",
            "path": path,
            "device": "cuda",
            "overrides": {"n_action_steps": 1, "num_steps": 10},
        },
        "quantization": {
            "enabled": True,
            "quantize_matmul": True,
            "matmul_scale_granularity": granularity,
            "scale_dir": f"/tmp/_phaseB_{granularity}",
            "linear": {"enabled": False, "include": [], "exclude": []},
        },
    }
    wrapper = ModelWrapper(cfg)
    wrapper.build(mode="raw")
    return wrapper


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/smolvla_base"
    granularity = sys.argv[2] if len(sys.argv) > 2 else "per_site"

    wrapper = build_wrapper(path, granularity)
    am = wrapper.model.model.vlm_with_expert

    n = len(am.quant_matmuls)
    assert n == 64, f"expected 64 MatMul, got {n}"
    print(f"[1] 64 physical MatMul objects: OK ({n})")

    ids = [m.module_id for m in am.quant_matmuls.values()]
    assert len(set(ids)) == 64, "module_id must be unique"
    print(f"[2] unique module_id: OK (e.g. {ids[0]} .. {ids[-1]})")

    if granularity == "per_site":
        sgs = [m._scale_identity() for m in am.quant_matmuls.values()]
        assert len(set(sgs)) == 64, "per_site should give 64 unique scale groups"
        print("[3] per_site -> 64 unique scale groups: OK")

    # --- raw-equivalence check on a synthetic attention input ---
    orig = am._original_attention_forward
    quant = am.get_attention_interface()

    torch.manual_seed(0)
    B, S = 1, 8
    H = am.num_attention_heads
    H_kv = am.num_key_value_heads
    D = am.vlm.config.text_config.head_dim
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, H_kv, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, H_kv, D, device="cuda", dtype=torch.float32)
    attn_mask = torch.ones(B, S, S, dtype=torch.bool, device="cuda")

    token = _CURRENT_ATTN_SITE.set(("vlm", 3))
    try:
        out_quant = quant(attn_mask, B, D, q, k, v)
    finally:
        _CURRENT_ATTN_SITE.reset(token)

    out_orig = orig(attn_mask, B, D, q, k, v)

    diff = (out_quant - out_orig).abs().max().item()
    print(f"[4] raw-equivalence (vlm layer 3): max|diff| = {diff:.3e}")
    assert diff < 1e-5, f"raw mode must match original: {diff}"

    # expert routing
    token = _CURRENT_ATTN_SITE.set(("expert", 15))
    try:
        out_quant_e = quant(attn_mask, B, D, q, k, v)
    finally:
        _CURRENT_ATTN_SITE.reset(token)
    diff_e = (out_quant_e - out_orig).abs().max().item()
    print(f"[5] raw-equivalence (expert layer 15): max|diff| = {diff_e:.3e}")
    assert diff_e < 1e-5

    print("\nPHASE B RAW-EQUIVALENCE: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
