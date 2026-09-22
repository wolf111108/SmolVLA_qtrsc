# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。
> 本 task 仅做 workload characterization；不新增规范外设计文档，实验设计统一保留在本文件。

- **实验名称**：2026-09-15_phaseI_vision-quantization / task: vision-sparsity-compute
- **状态**：draft
- **负责人**：
- **创建日期**：2026-09-22
- **相关前序实验**：父实验 VLIN（Vision 72-Linear FP8）、Phase H sparsity、2026-09-15 sparsity-ratio-quickscan

---

## 1. 实验目的

1. 在已经接入 Vision Encoder 72 个 QuantizedLinear 后，重新统计 **Vision / VLM / Expert 三个 component 的 runtime 元素级稀疏度、bit 级稀疏度与 static weight 稀疏度**。
2. 给出加入 Vision 后的 **per-`sample_actions()` MAC/FLOP 组成**，区分 quantized major ops 与仍为 raw 的 Vision SDPA QK/PV、connector projection。
3. 计算当前 VLIN 配置覆盖的 **quantized major-op FLOP coverage**，但不把 sparsity 或 FLOPs 直接换算成真实硬件 speedup。

本 task 采用 VLIN 完整 72-Linear FP8 配置，因为它是目前覆盖 Vision 计算量最大的已验证配置。父实验的准确率归因已经得到：

```text
G6-A strict baseline  = 90%
V2 Vision MLP-only    = 84%   (L_MLP  = 6pp)
V3 AttnProj-only      = 85%   (L_Attn = 5pp)
VLIN all 72 Linear    = 80%   (L_all  = 10pp)
interaction           = -1pp ≈ 0
```

因此当前重点从“是否能量化”转向“加入 Vision 后，量化 workload 的稀疏性和计算覆盖到底是多少”。

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | `smolvla_eval` |
| 仓库 revision / commit | 当前工作分支 `phaseI/vision-linear-full-integration` |
| LeRobot 路径与 commit | `lerobot_current/` |
| GPU | H100 系；遵循既有 EGL 配置，不在 runner 中设置 `CUDA_VISIBLE_DEVICES` |
| 关键依赖版本 | 继承父实验环境 |

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero` |
| 量化配置 | VLIN：VLM/Expert 224 Linear + 64 MatMul FP8；Vision 72 Linear FP8；Vision QK/PV raw SDPA；connector raw |
| scale | **严格复用** `scales/2026-09-15_phaseI_vision-quantization/vlin_full_vision_linear_fp8`，本 task 禁止 recalibrate |
| workload 数据 | LIBERO Goal task0 × 1 episode；只用于统计，不做 accuracy claim |

### 3.1 统计覆盖范围

当前统计框架只对 wrapped quantized operators 采集 sparsity，因此：

| component / block | sparsity | compute |
|---|---|---|
| Vision 72 Linear | ✅ runtime + static weight | ✅ dynamic workload + FLOPs |
| Vision SDPA QK/PV | ❌ raw，不进入 quantized sparsity denominator | ✅ 用 V0 已验证公式补入 |
| Connector projection | ❌ raw | ✅ 用结构公式补入 |
| VLM 112 Linear + 32 MatMul | ✅ runtime；Linear static weight | ✅ dynamic workload |
| Expert 112 Linear + 32 MatMul | ✅ runtime；Linear static weight | ✅ dynamic workload |

**禁止**把“Vision 72 Linear 的 sparsity”写成“完整 Vision Encoder sparsity”。完整 Vision 中仍有 raw SDPA QK/PV、connector，以及未计入本 major-op FLOP 口径的 patch embed / LayerNorm / softmax / GELU 等。

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | task0 × 1 ep，seed=1000 |
| 采样参数 | `n_action_steps=10`，`num_steps=10` |
| calibration | **`--skip-calibration` 必须启用**；避免 stat manager 被 calibration 覆盖的已知问题，并保证与 VLIN 数值完全一致 |
| sparsity primary | native runtime element sparsity / native runtime bit sparsity / static weight element+bit sparsity |
| unit sparsity | 关闭，避免额外统计开销；本 task 不需要 2×2 unit 指标 |
| chunk size | 4,194,304 |
| compute | MACs / FLOPs per `sample_actions()`，不等于 latency |

### 4.1 稀疏度口径

Runtime 必须使用 `module_sparsity.csv` 中的：

```text
zero_elements_native / total_elements_native
sparse_bits_native / total_bits_native
```

而不是 `reported`。原因是 PoT-FP8 + outlier protection 的 normal path 会把 protected FP sidepath 位置写成 0；若直接用 reported counters，会把这些人工 0 当作真实可稀疏 workload，从而**高估**稀疏度。

Static weight 采用 `weight_sparsity_static.csv`。MatMul 的 B 是 runtime tensor，不是 static weight，因此 static weight 行数期望只有全部 QuantizedLinear：

[
224 + 72 = oxed{296}
]

FP8 bit sparsity 延续当前框架语义：**E4M3 significand zero-bit metric**，不是完整 8-bit code 的“0 bit 比例”。因此 element sparsity 与 bit sparsity应分开报告。

### 4.2 计算量口径

Primary compute 用两部分合并：

1. `workload.csv`：对 QuantizedLinear 只取 `activation` 行、对 QuantizedMatMul 只取 `A` 行，避免一个物理算子因为 A/O/B 等 tensor role 被重复计 MAC；乘 `calls` 后再除以推断出的 `sample_actions()` 次数。
2. raw Vision major ops：按 V0 已验证结构公式加入 SDPA QK/PV 和 connector projection。

Vision 固定参数：

```text
layers=12
tokens/camera=1024
hidden=768
intermediate=3072
heads=12
head_dim=64
cameras=2
connector: 64 × 12288 -> 960
```

已知 sanity reference：

```text
Vision 72 Linear ≈ 347.89 GFLOPs / sample_actions
Vision QK + PV   ≈ 77.31 GFLOPs / sample_actions
Connector proj   ≈ 3.02 GFLOPs / sample_actions
```

即 Vision+Connector audited major ops ≈ 428.22 GFLOPs，与父实验 V0 一致。

最终还会动态加入 VLM / Expert 的实际 workload FLOPs。由于当前 major-op accounting 不含 patch embed / LayerNorm / softmax / GELU / pixel-shuffle bookkeeping，最终总量应称为 **audited major-op FLOPs**，而非严格全模型所有算术 FLOPs。

## 5. 实验变量与分组

| 组 | config | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| VSC-0 | `vsc_vlin_fp8_task0_1ep.yaml` | 开启 sparsity collector；VLIN 量化不变 | VLIN scales、task0、seed1000 | Primary：1ep 统计 Vision/VLM/Expert sparsity + compute |

### 5.1 预期 Gate

运行后必须满足：

```text
quantization_manifest rows = 360
  Vision Linear            = 72
  VLM Linear / MatMul      = 112 / 32
  Expert Linear / MatMul   = 112 / 32

weight_sparsity_static rows = 296

runtime components include:
  vision
  vlm
  expert

Expert flow_step:
  0..9 全覆盖

Vision 72 Linear:
  ≈347.89 GFLOPs/sample_actions
```

Primary summary按 component输出：

```text
Vision prefill:
  runtime native element sparsity
  runtime native bit sparsity
  static weight element sparsity
  static weight bit sparsity

VLM prefill:
  同上

Expert denoise:
  同上

all_quantized:
  分子/分母先求和再相除的 pooled 结果
```

禁止直接平均各 layer 的 sparsity ratio。

## 6. 运行命令

```bash
git checkout phaseI/vision-linear-full-integration

bash experiments/2026-09-15_phaseI_vision-quantization/tasks/vision-sparsity-compute/scripts/run_vision_sparsity_compute.sh
```

runner 会按顺序执行：

```text
Gate 0: py_compile + sparsity unit tests
Gate 1: VLIN 72/72 calibration coverage preflight
Gate 2: task0 × 1ep --skip-calibration + sparsity collection
Gate 3: primary CSV presence / non-empty
Gate 4: component sparsity + compute summarization
```

## 7. 输出目录映射

| config | 输出目录 | 状态 |
|---|---|---|
| `vsc_vlin_fp8_task0_1ep.yaml` | `outputs/2026-09-15_phaseI_vision-quantization/tasks/vision-sparsity-compute/vsc_vlin_fp8_task0_1ep/` | pending |

核心原始产物：

```text
sparsity/module_sparsity.csv
sparsity/weight_sparsity_static.csv
sparsity/quantization_manifest.csv
sparsity/workload.csv
```

汇总产物：

```text
sparsity_compute_summary.csv
sparsity_by_operator.csv
compute_summary.csv
```

## 8. 风险与注意事项

- **不得 calibration + sparsity 同 run**：既有框架中 calibration 可能替换 module 的 stat manager，导致 runtime sparsity 静默为空；本 runner 硬编码 `--skip-calibration`。
- VLIN scale 不齐时直接 fail，不允许自动生成新 scale，否则会破坏与 80% VLIN run 的严格一致性。
- `CUDA_VISIBLE_DEVICES` 在 runner 中主动 unset，避免 robosuite EGL assertion 与 conda 内 `MUJOCO_EGL_DEVICE_ID` 冲突。
- runtime sparsity 只描述执行到的**量化 tensor code**；raw Vision SDPA/connector 不得被悄悄纳入 denominator。
- `compute_summary.csv` 是理论/工作量核算，不是 measured latency；不得从 FLOPs 或 bit sparsity 直接声明实际 speedup。
- 如 1ep 结果需要稳定性验证，可在**同一 task 内新增 config 变体**扩到 3ep/10ep，不另建顶层实验。
