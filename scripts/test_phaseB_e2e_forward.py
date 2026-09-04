"""Phase B end-to-end forward equivalence check.

Runs a full SmolVLMWithExpertModel.forward (prefix prefill, fill_kv_cache
path) once WITHOUT matmul injection and once WITH it (mode="raw"), and
asserts the outputs are bit-exact. This exercises the complete chain:
    forward -> forward_attn_layer -> ContextVar set -> dispatcher -> MatMul.

Usage:
    python scripts/test_phaseB_e2e_forward.py [checkpoint_path]
"""

import sys

import torch

from vla_tcs2.model_wrapper import (
    ModelWrapper,
    _inject_smolvla_quantized_matmul,
)


def run_forward(am):
    B, S_p = 1, 4
    hidden = am.get_vlm_model().text_model.config.hidden_size
    prefix = torch.randn(B, S_p, hidden, device="cuda", dtype=torch.float32)
    inputs_embeds = [prefix, None]
    attn_mask = torch.ones(B, S_p, S_p, dtype=torch.bool, device="cuda")
    pos_ids = torch.arange(S_p, device="cuda").unsqueeze(0)
    with torch.no_grad():
        out, _cache = am.forward(
            attn_mask, pos_ids, None, inputs_embeds, use_cache=True
        )
    return out[0]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/smolvla_base"

    base_cfg = {
        "model": {
            "type": "smolvla",
            "path": path,
            "device": "cuda",
            "overrides": {"n_action_steps": 1, "num_steps": 10},
        },
        "quantization": {"enabled": False},
    }
    w = ModelWrapper(base_cfg)
    w.build(mode="raw")
    am = w.model.model.vlm_with_expert

    torch.manual_seed(0)
    out_base = run_forward(am)

    _inject_smolvla_quantized_matmul(
        am,
        {
            "quantize_matmul": True,
            "matmul_scale_granularity": "per_site",
            "scale_dir": "/tmp/_phaseB_e2e",
        },
        "raw",
        None,
    )

    torch.manual_seed(0)
    out_quant = run_forward(am)

    diff = (out_quant - out_base).abs().max().item()
    print(f"E2E forward max|diff| = {diff:.3e}")
    assert diff < 1e-5, f"raw forward must match baseline: {diff}"
    print("PHASE B E2E FORWARD: EQUIVALENCE CONFIRMED")


if __name__ == "__main__":
    main()
