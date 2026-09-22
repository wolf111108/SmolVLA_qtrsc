# 实验设置（Experiment Setup）

> 本文档在实验开跑前填写；固定章节保留。

- **实验名称**：2026-09-22_phaseI_vision-smmm-sparsity
- **状态**：ready
- **负责人**：
- **创建日期**：2026-09-22
- **相关前序实验**：2026-09-22_quickscan_smmm-bit-sparsity；Phase I VLIN / vision-sparsity-compute

---

## 1. 实验目的

在 S|MMM 统计口径已通过 VLM/Expert quickscan 后，将同一口径扩展到包含 Vision Encoder 的完整 VLIN 量化 workload。暂时只跑 LIBERO Goal task0 × 1 episode，统计 Vision / VLM / Expert 的 runtime element sparsity、runtime S|MMM bit sparsity、static weight element/bit sparsity及 pooled 结果。

S|MMM 定义：E4M3 raw = `S EEEE MMM`，统计 code = `S MMM`；exponent 与 hidden leading 1 均不计。

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | `smolvla_eval` |
| 分支 | `phaseI/vision-linear-full-integration` |
| policy | `lerobot/smolvla_libero` |
| FP bit metric | **S|MMM v1** |
| GPU / simulator | 继承 Phase I / VLIN 当前环境与 EGL 约定 |

## 3. 模型与数据

| Component | Quantized operators | Raw / excluded |
|---|---|---|
| Vision | 72 Linear = 12×(q/k/v/out + fc1/fc2) | Vision SDPA QK/PV raw |
| VLM | 112 Linear + 32 MatMul | — |
| Expert | 112 Linear + 32 MatMul | — |
| Connector | — | raw |

预期 QuantizedLinear=296、QuantizedMatMul=64、总 quantized modules=360。

Scale 来源：`scales/2026-09-15_phaseI_vision-quantization/vlin_full_vision_linear_fp8`。runner 复制到本实验独立 scale_dir，并要求 source/target 均严格为 **1080** 个 `.p` 且 sha256 bit-identical；运行时强制 `--skip-calibration`。

## 4. 评测协议

| 项目 | 值 |
|---|---|
| benchmark | LIBERO Goal task0 |
| episodes | **1** |
| seed | 1000 |
| n_action_steps / num_steps | 10 / 10 |
| batch_size | 1 |
| calibration | 禁止，`--skip-calibration` |
| sparsity counter | `*_native` |
| unit sparsity | disabled |
| chunk_size | 4,194,304 |

使用 native 口径以排除 outlier protection 在 protected 位置产生的 artificial zero。

## 5. 实验变量与分组

| 组 | config | 变量取值 | 固定项 |
|---|---|---|---|
| VS1 | `vs1_full_vlin_smmm_1ep.yaml` | 完整 VLIN + S|MMM sparsity | VLIN scales、task0、seed1000、1ep |

结构 Gate：Vision runtime rows=144 / weight rows=72；VLM runtime rows=320 / weight rows=112；Expert runtime rows=3200 / weight rows=112 / flow_step=0..9；static weight total=296；manifest=360。Pooled ratio 必须先加 numerator/denominator 再相除。

## 6. 运行命令

```bash
bash experiments/2026-09-22_phaseI_vision-smmm-sparsity/scripts/run_vision_smmm_1ep.sh
```

顺序：Gate0 pytest → Gate1 1080 scale copy+sha256 → Gate2 VLIN coverage preflight → Gate3 task0×1ep → Gate4 primary CSV → Gate5 S|MMM summary。

## 7. 输出目录映射

| config | output | 状态 |
|---|---|---|
| `vs1_full_vlin_smmm_1ep.yaml` | `outputs/2026-09-22_phaseI_vision-smmm-sparsity/vs1_full_vlin_smmm_1ep/` | pending |

核心输出：`sparsity/module_sparsity.csv`、`sparsity/weight_sparsity_static.csv`、`sparsity/quantization_manifest.csv`、`vision_smmm_sparsity_summary.csv`、`scale_hashes.txt`、`run.log`。

`vision_smmm_sparsity_summary.csv` 显式写入 `fp_bit_metric=S|MMM` 和 `fp_bit_metric_version=1`，避免与历史 1MMM 混淆。

## 8. 风险与注意事项

- 当前 branch 的 FP sparsity core 已采用 S|MMM；历史 1MMM CSV 不得与本实验直接混用。
- 只有 1 episode，适合快速 workload characterization，不用于估计跨 task 方差。
- Vision QK/PV 与 connector 仍为 raw，因此 Vision bit sparsity只代表 Vision 72 QuantizedLinear，不代表完整 Vision Encoder 所有算子。
- S|MMM zero-bit ratio 是编码层面的 opportunity，不等价于真实硬件 MAC/energy reduction。