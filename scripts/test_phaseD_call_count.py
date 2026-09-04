"""Phase D — Call-count audit (plan §25).

Proves that all 64 physical MatMul objects are actually *executed* during a
real rollout (prefix prefill -> vlm component, denoise -> expert component),
not merely created. Registers a forward hook on every QuantizedMatMul and
reports `module_id -> call_count`.

Expected after one prefix prefill + one denoise step:
    vlm.layer.0.qk    > 0
    vlm.layer.0.pv    > 0
    ...
    expert.layer.15.qk > 0
    expert.layer.15.pv > 0

If any entry is 0, the ContextVar routing is not covering that site.

Usage:
    python scripts/test_phaseD_call_count.py [checkpoint_path]
"""

import sys
from collections import Counter

import torch

from vla_tcs2.model_wrapper import ModelWrapper
from vla_tcs2.quant_matmul import QuantizedMatMul


def build(path: str) -> ModelWrapper:
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
            "matmul_scale_granularity": "per_site",
            "linear_scale_granularity": "per_site",
            "scale_dir": "/tmp/_phaseD_callcount",
            "linear": {"enabled": True, "include": ["*"], "exclude": []},
        },
    }
    w = ModelWrapper(cfg)
    w.build(mode="scale_inspection")
    return w


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/smolvla_base"
    w = build(path)
    am = w.model.model.vlm_with_expert

    # Forward hooks record each MatMul call.
    counts: Counter[str] = Counter()
    handles = []
    for m in am.modules():
        if isinstance(m, QuantizedMatMul):
            def make_hook(mm):
                def hook(module, args, kwargs):
                    counts[mm.module_id] += 1
                return hook
            handles.append(m.register_forward_hook(make_hook(m)))

    vlm_h = am.get_vlm_model().text_model.config.hidden_size
    exp_h = am.lm_expert.config.hidden_size

    torch.manual_seed(0)
    B, S_p, S_s = 1, 4, 2
    prefix = torch.randn(B, S_p, vlm_h, device="cuda", dtype=torch.float32)
    mask = torch.ones(B, S_p, S_p, dtype=torch.bool, device="cuda")
    pos = torch.arange(S_p, device="cuda").unsqueeze(0)

    with torch.no_grad():
        _, cache = am.forward(mask, pos, None, [prefix, None], use_cache=True)

        # denoise step (expert component)
        suffix = torch.randn(B, S_s, exp_h, device="cuda", dtype=torch.float32)
        full = torch.ones(B, S_s, S_p + S_s, dtype=torch.bool, device="cuda")
        pos2 = torch.arange(S_p, S_p + S_s, device="cuda").unsqueeze(0)
        am.forward(full, pos2, cache, [None, suffix], use_cache=True)

    for h in handles:
        h.remove()

    # --- audit ---
    n_matmuls = len([m for m in am.modules() if isinstance(m, QuantizedMatMul)])
    assert n_matmuls == 64, f"expected 64 MatMul, got {n_matmuls}"

    never = [
        m.module_id for m in am.modules()
        if isinstance(m, QuantizedMatMul) and m.module_id not in counts
    ]

    print(f"physical MatMul: {n_matmuls}")
    print(f"called MatMul: {len(counts)}")
    print(f"zero-count (created but never called): {len(never)}")
    if never:
        print("  NEVER CALLED:")
        for mid in sorted(never):
            print(f"    {mid}")

    # A sample of the call map
    print("\ncall map (sorted, first 12):")
    for mid, c in sorted(counts.items())[:12]:
        print(f"  {mid:24s} {c}")

    assert len(never) == 0, (
        f"{len(never)} MatMul were never called: {sorted(never)}"
    )
    print("\nPHASE D CALL-COUNT: ALL 64 MATMUL EXECUTED")


if __name__ == "__main__":
    main()
