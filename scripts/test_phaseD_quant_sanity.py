"""Phase D — Quantization sanity test (plan §27).

4-step progressive verification of the quantization chain, using int8
per_tensor (no outlier) for simplicity; FP8 specifics are exercised by the
formal ablation configs.

    Step 1: Linear raw + MatMul raw           -> routing intact (finite output)
    Step 2: Linear raw + MatMul quant_forward -> MatMul scales load, output close
    Step 3: Linear quant_forward + MatMul quant_forward -> full quant loads
    Step 4: granularity sweep                 -> scale-file count correct

Usage:
    python scripts/test_phaseD_quant_sanity.py [checkpoint_path]
"""

import sys

import torch

from vla_tcs2.model_wrapper import ModelWrapper, switch_quantization_mode_all
from vla_tcs2.quant_linear import QuantizedLinear
from vla_tcs2.quant_matmul import QuantizedMatMul

GRANULARITIES = ["global", "per_component", "per_layer", "per_site"]
EXPECTED_MM = {"global": 2, "per_component": 4, "per_layer": 32, "per_site": 64}


def build(path: str, granularity: str, mode: str) -> ModelWrapper:
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
            "quantize_matmul": True,
            "matmul_scale_granularity": granularity,
            "linear_scale_granularity": granularity,
            "scale_dir": f"/tmp/_phaseD_sanity_{granularity}",
            "linear": {"enabled": True, "include": ["*"], "exclude": []},
        },
    }
    w = ModelWrapper(cfg)
    w.build(mode=mode)
    return w


def run_full(w: ModelWrapper):
    am = w.model.model.vlm_with_expert
    vlm_h = am.get_vlm_model().text_model.config.hidden_size
    exp_h = am.lm_expert.config.hidden_size
    torch.manual_seed(0)
    B, S_p, S_s = 1, 4, 2
    prefix = torch.randn(B, S_p, vlm_h, device="cuda", dtype=torch.float32)
    mask = torch.ones(B, S_p, S_p, dtype=torch.bool, device="cuda")
    pos = torch.arange(S_p, device="cuda").unsqueeze(0)
    with torch.no_grad():
        _, cache = am.forward(mask, pos, None, [prefix, None], use_cache=True)
        suffix = torch.randn(B, S_s, exp_h, device="cuda", dtype=torch.float32)
        full = torch.ones(B, S_s, S_p + S_s, dtype=torch.bool, device="cuda")
        pos2 = torch.arange(S_p, S_p + S_s, device="cuda").unsqueeze(0)
        out, _ = am.forward(full, pos2, cache, [None, suffix], use_cache=True)
    return out[1]  # suffix hidden states (denoise: inputs_embeds[0] is None)


def calibrate(w: ModelWrapper):
    switch_quantization_mode_all(w.model, "scale_inspection")
    run_full(w)
    w.stat_manager.save_all_scales()


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/smolvla_base"

    # ------------------------------------------------------------------ Step 1
    print("=== Step 1: Linear raw + MatMul raw ===")
    w = build(path, "per_site", "raw")
    out = run_full(w)
    assert torch.isfinite(out).all(), "raw forward produced non-finite output"
    print(f"  raw forward OK, output shape={tuple(out.shape)}")

    # ------------------------------------------------------------------ Step 2
    print("\n=== Step 2: Linear raw + MatMul quant_forward ===")
    w = build(path, "per_site", "scale_inspection")
    calibrate(w)
    switch_quantization_mode_all(w.model, "raw")
    out_raw = run_full(w)

    # quantize only MatMul
    for m in w.model.modules():
        if isinstance(m, QuantizedLinear):
            m.mode = "raw"
        elif isinstance(m, QuantizedMatMul):
            m.mode = "quant_forward"
    out_q = run_full(w)
    diff = (out_q - out_raw).abs().max().item()
    print(f"  MatMul-only quant max|diff| vs raw = {diff:.3e}")
    assert torch.isfinite(out_q).all()

    # ------------------------------------------------------------------ Step 3
    print("\n=== Step 3: Linear + MatMul quant_forward ===")
    switch_quantization_mode_all(w.model, "quant_forward")
    out_full = run_full(w)
    diff_full = (out_full - out_raw).abs().max().item()
    print(f"  full quant max|diff| vs raw = {diff_full:.3e}")
    assert torch.isfinite(out_full).all()

    # ------------------------------------------------------------------ Step 4
    print("\n=== Step 4: granularity sweep (scale-file counts) ===")
    for g in GRANULARITIES:
        wg = build(path, g, "scale_inspection")
        calibrate(wg)
        import glob
        files = glob.glob(f"/tmp/_phaseD_sanity_{g}/*.p")
        mm = len({m._scale_identity() for m in wg.model.modules()
                  if isinstance(m, QuantizedMatMul)})
        ln = len({m._scale_identity() for m in wg.model.modules()
                  if isinstance(m, QuantizedLinear)})
        assert mm == EXPECTED_MM[g], f"{g}: MatMul groups={mm}"
        exp_files = mm * 3 + ln * 3
        assert len(files) == exp_files, (
            f"{g}: files={len(files)}, expected {exp_files}"
        )
        print(f"  {g:14s}: MatMul groups={mm:3d}, files={len(files):4d}  OK")

    print("\nPHASE D QUANT SANITY: ALL STEPS PASSED")


if __name__ == "__main__":
    main()
