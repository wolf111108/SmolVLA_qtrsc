# Full Vision Encoder Linear Quantization — local experiment

Branch: `phaseI/vision-linear-full-integration`

## Scope

This experiment adds **all 72 Vision Encoder Linear operators** to the existing
quantization path while keeping Vision attention QK/PV inside the original
SDPA implementation.

Quantized Vision Linear operators:

- 12 × `q_proj`
- 12 × `k_proj`
- 12 × `v_proj`
- 12 × `out_proj`
- 12 × `fc1`
- 12 × `fc2`

Total: **72 QuantizedLinear**.

The connector is deliberately disabled. Existing VLM/Expert quantization stays
on the canonical FP8 background.

Expected model counts:

| component | QuantizedLinear | QuantizedMatMul |
|---|---:|---:|
| VLM + Expert | 224 | 64 |
| Vision | 72 | 0 |
| Connector | 0 | 0 |
| **Total** | **296** | **64** |

The Vision Linear portion corresponds to about **347.9 GFLOPs/sample_actions**
in the V0 audit: 231.9 G MLP + 116.0 G attention projections, i.e. roughly
81.2% of Vision compute and about 58% of whole-model dense inference FLOPs.

## Calibration isolation

The experiment uses a fresh scale directory:

`scales/2026-09-15_phaseI_vision-quantization/vlin_full_vision_linear_fp8`

Before calibration, the runner copies the canonical G6-A FP8 VLM/Expert scale
files into the new directory. The config sets legacy modules to `reuse`, while
the Vision wrapper forces the new 72 Vision Linear sites to `recalibrate`.

Expected action split:

- reuse = **288** = 224 legacy Linear + 64 legacy MatMul
- recalibrate = **72** = Vision Linear only

With per-site A/W/O scales, Vision calibration must add exactly:

`72 × 3 = 216` new scale files.

## Local run

From the repository root:

```bash
git checkout phaseI/vision-linear-full-integration
bash experiments/2026-09-15_phaseI_vision-quantization/scripts/run_full_vision_linear.sh
```

The runner performs:

1. `py_compile` and `tests/test_vision_quant_routing.py`
2. isolated canonical-scale copy
3. real-model routing/count audit
4. calibration-only
5. calibration coverage audit
6. LIBERO Goal task0 × 1 smoke

It stops immediately on any failed gate.

### If the canonical G6-A scale path differs locally

```bash
BASE_SCALE_DIR=/your/path/to/g6a_all_fp8_control \
bash experiments/2026-09-15_phaseI_vision-quantization/scripts/run_full_vision_linear.sh
```

### Formal Goal ×100 after smoke passes

Run directly without repeating calibration:

```bash
bash experiments/2026-09-15_phaseI_vision-quantization/scripts/run_full_vision_linear_goal.sh
```

or run the all-in-one path:

```bash
RUN_GOAL=1 bash experiments/2026-09-15_phaseI_vision-quantization/scripts/run_full_vision_linear.sh
```

## Gate expectations

### Routing gate

Must print:

```text
QuantizedLinear total         : 296
legacy VLM/Expert Linear      : 224
Vision Linear                 : 72
Vision attention projection   : 48
Vision MLP                    : 24
Connector Linear              : 0
QuantizedMatMul total         : 64
Vision QuantizedMatMul        : 0

Calibration action plan:
reuse       = 288
recalibrate = 72

ROUTING GATE: PASS
```

### Calibration coverage gate

Must print:

```text
Vision Linear sites : 72 / 72
complete sites      : 72 / 72
Vision scale files  : 216 / 216

q_proj   : 12 / 12
k_proj   : 12 / 12
v_proj   : 12 / 12
out_proj : 12 / 12
fc1      : 12 / 12
fc2      : 12 / 12

CALIBRATION COVERAGE GATE: PASS
```

Every persisted scale must be finite and strictly positive.

## Configs

Smoke / calibration:

`configs/vlin_full_vision_linear_fp8.yaml`

Formal Goal ×100:

`configs/vlin_full_vision_linear_fp8_goal.yaml`

Both use:

- PoT-FP8 + outlier protection
- A/W/O = E4M3 for Vision Linear
- outlier ratio = 0.01
- per-site scale groups
- `n_action_steps=10`
- `num_steps=10`
- seed = 1000 for evaluation

Vision SDPA QK/PV is intentionally untouched in this branch.
