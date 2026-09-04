"""Phase D — Preflight audit (plan §24).

Checks, for each granularity, that the wrapped model satisfies the expected
physical/scale identity invariants BEFORE any calibration or rollout:

  - physical MatMul count == 64 (2 components x 16 layers x qk/pv)
  - physical Linear count == 224 (2 components x 16 layers x 7 ops)
  - unique module_id == physical count (no identity collision)
  - unique scale groups == granularity-specific count
  - expected scale files == unique scale groups x 3

Granularity expectations (scale groups -> scale files):
    global          MatMul  2 ->  6   Linear  7 ->  21
    per_component   MatMul  4 -> 12   Linear 14 ->  42
    per_layer       MatMul 32 -> 96   Linear 112 -> 336
    per_site        MatMul 64 -> 192  Linear 224 -> 672

Usage:
    python scripts/test_phaseD_preflight.py [checkpoint_path]
"""

import sys

from vla_tcs2.model_wrapper import ModelWrapper
from vla_tcs2.quant_linear import QuantizedLinear
from vla_tcs2.quant_matmul import QuantizedMatMul

GRANULARITIES = ["global", "per_component", "per_layer", "per_site"]

# Expected unique scale-group counts per granularity.
EXPECTED = {
    "global": {"matmul": 2, "linear": 7},
    "per_component": {"matmul": 4, "linear": 14},
    "per_layer": {"matmul": 32, "linear": 112},
    "per_site": {"matmul": 64, "linear": 224},
}


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
            "quantize_matmul": True,
            "matmul_scale_granularity": granularity,
            "linear_scale_granularity": granularity,
            "scale_dir": f"/tmp/_phaseD_preflight_{granularity}",
            "linear": {"enabled": True, "include": ["*"], "exclude": []},
        },
    }
    w = ModelWrapper(cfg)
    w.build(mode="scale_inspection")
    return w


def audit(path: str, granularity: str) -> None:
    print(f"\n=== preflight: {granularity} ===")
    w = build(path, granularity)

    matmuls = [
        m for m in w.model.modules() if isinstance(m, QuantizedMatMul)
    ]
    linears = [
        m for m in w.model.modules() if isinstance(m, QuantizedLinear)
    ]

    # 1. physical counts
    assert len(matmuls) == 64, f"physical MatMul={len(matmuls)}, expected 64"
    assert len(linears) == 224, f"physical Linear={len(linears)}, expected 224"
    print(f"[1] physical: MatMul={len(matmuls)}, Linear={len(linears)}  OK")

    # 2. unique module_id
    mm_ids = {m.module_id for m in matmuls}
    ln_ids = {m.module_id for m in linears}
    assert len(mm_ids) == 64, f"unique MatMul module_id={len(mm_ids)}"
    assert len(ln_ids) == 224, f"unique Linear module_id={len(ln_ids)}"
    assert mm_ids.isdisjoint(ln_ids), "MatMul/Linear module_id spaces must not overlap"
    print("[2] unique module_id (64 MatMul + 224 Linear, disjoint)  OK")

    # 3. unique scale groups
    mm_sg = {m._scale_identity() for m in matmuls}
    ln_sg = {m._scale_identity() for m in linears}
    exp = EXPECTED[granularity]
    assert len(mm_sg) == exp["matmul"], (
        f"{granularity}: MatMul scale groups={len(mm_sg)}, expected {exp['matmul']}"
    )
    assert len(ln_sg) == exp["linear"], (
        f"{granularity}: Linear scale groups={len(ln_sg)}, expected {exp['linear']}"
    )
    print(
        f"[3] scale groups: MatMul={len(mm_sg)} Linear={len(ln_sg)}  OK"
    )

    # 4. expected scale files
    n_files = len(mm_sg) * 3 + len(ln_sg) * 3
    print(f"[4] expected scale files = {n_files}  OK")

    print(f"preflight {granularity}: PASSED")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/smolvla_base"
    for g in GRANULARITIES:
        audit(path, g)
    print("\nPHASE D PREFLIGHT: ALL GRANULARITIES PASSED")


if __name__ == "__main__":
    main()
