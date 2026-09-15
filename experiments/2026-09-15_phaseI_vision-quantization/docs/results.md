# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-15_phaseI_vision-quantization
- **状态**：running（draft / running / done / aborted）
- **最后更新**：2026-09-15

---

## 1. 摘要

Gate L0（legacy regression）已通过：用原封不动的 G6 canonical 配置 build，确认 `QuantizedLinear=224`、`QuantizedMatMul=64`、`vision.*=0`、`connector.*=0`，Phase I 新增代码未破坏既有 VLM/Expert wrapping。V0（Vision workload audit）已完成：真实模型为 12 层 Vision Transformer（hidden=768、intermediate=3072、heads=12），attn 实现为 `sdpa`；一次 `sample_actions()` 处理 **2 张相机图**，每张图 1024 patch tokens → connector 后 64 visual tokens（12288→960）。Vision+Connector 单次 sample_actions 理论 dense FLOPs ≈ **428.2 GFLOPs**，其中 Vision MLP 54.2%、attention projection 27.1%、QK/PV 18.1%、connector 0.71%，与手册 §0 粗估（430.6 G，72%）一致。

## 2. 总结果表

| 组 | config | 基准 | Success Rate | Δ vs baseline | episodes | 输出目录 | 备注 |
|---|---|---|---|---|---|---|---|
| Gate L0 | g6a_all_fp8_control（复用） | — | — | — | — | — | 224 Linear / 64 MatMul / 0 vision / 0 connector |
| V0 | v0_workload_audit | — | 100.0%（task0） | — | 1 | `outputs/.../v0_workload_audit` | 无量化，仅 workload/FLOPs 审计 |

## 3. 分组结果与分析

### 3.1 Gate L0 — legacy regression

| 项 | 期望 | 实测 | 判定 |
|---|---:|---:|---|
| QuantizedLinear | 224 | 224 | ✅ |
| QuantizedMatMul | 64 | 64 | ✅ |
| vision.* module | 0 | 0 | ✅ |
| connector.* module | 0 | 0 | ✅ |

### 3.2 V0 — Vision 结构 / 运行时 / FLOPs

静态结构（`vision_structure.json`）：

| 项 | 值 |
|---|---|
| Vision 层数 | 12 |
| hidden / intermediate | 768 / 3072 |
| heads | 12 |
| image_size / patch_size | 512 / 16（→ 1024 patch tokens） |
| attention impl | **sdpa**（⚠ 非 eager，V4 需先做 eager-equivalence gate） |
| connector | scale_factor=4，proj 12288→960（→ 64 tokens） |

运行时（`vision_runtime_shapes.json`）：

| 项 | 值 |
|---|---|
| cameras / sample_actions | **2.0**（真实 inference 实测：19 次 sample_actions × 2 = 38 vision calls） |
| vision 输出 shape | [1, 1024, 768] |
| connector 输出 shape | [1, 64, 960] |

FLOPs（`vision_flops.csv`，1 MAC = 2 FLOPs，per sample_actions）：

| operator | FLOPs | % of vision |
|---|---:|---:|
| Vision MLP (fc1/fc2) | 2.319e11 | 54.16% |
| Attention projection (q/k/v/out) | 1.160e11 | 27.08% |
| QK | 3.865e10 | 9.03% |
| PV | 3.865e10 | 9.03% |
| Connector proj | 3.020e09 | 0.71% |
| **合计** | **4.282e11** | 100% |

分析：Vision MLP 是 Vision 内部最大算力块（54%），其次 attention projection（27%）、QK/PV（18%）。与手册 §0 粗估一致，验证了「只补 Vision MLP + attention projection 即可覆盖约 80% Vision compute」的判断。

### 3.3 FLOPs 占比（饼图）

**Phase I 实测（V0）— Vision encoder + connector 内部拆分**（428.2 GFLOPs / `sample_actions()`）

![Phase I Vision FLOPs](figures/phaseI_flops_pie_vision.png)

| 子模块 | GFLOPs | Vision 内占比 |
|---|---:|---:|
| Vision MLP `fc1/fc2` | 231.93 | 54.16% |
| Attention Q/K/V/out projection | 115.96 | 27.08% |
| Attention `QK^T` | 38.65 | 9.03% |
| Attention `P×V` | 38.65 | 9.03% |
| Connector projection | 3.02 | 0.71% |
| **合计** | **428.22** | **100%** |

**整体推理 — 单次 `sample_actions()`**（手册 §0 粗估，595.7 GFLOPs ≈ 0.60 TFLOPs）

![Phase I inference FLOPs](figures/phaseI_flops_pie_inference.png)

| 组件 | GFLOPs | 占比 |
|---|---:|---:|
| Vision Encoder + pixel shuffle / connector | 430.60 | 72.28% |
| VLM 16-layer prefix prefill | 57.60 | 9.67% |
| Action Expert 10-step denoise | 107.50 | 18.05% |
| **总计** | **595.70** | **100%** |

两图并排（便于看出左图的蓝色扇区就是右图的放大）：

![Phase I FLOPs 2-panel](figures/phaseI_flops_pie_2panel.png)

**配色说明**：整体图给每个顶层分支一个色相（Vision 蓝 / VLM 橙 / Expert 绿）；Vision 内部图是对蓝色分支的放大，因此用**同一蓝色的 5 档明度**，按占比由大到小由深到浅。两份图共用 `scripts/figure_palette.py`，与 Phase F/G 图同源。

**交叉验证**：手册 §0 粗估 Vision+Connector 为 430.6 G，V0 实测为 428.2 G，两者相差 **0.6%**，互相印证。

生成命令（数值均从源文件解析，不硬编码；整体图的数字直接解析手册 §0 表格）：

```bash
python experiments/2026-09-15_phaseI_vision-quantization/scripts/plot_phaseI_flops_pies.py
```

## 4. 结论

- Phase I 新增代码向后兼容（Gate L0 PASS，224/64/0/0）。
- Vision+Connector 占单次 sample_actions 的 dense FLOPs 约 428 G（占全推理 ~72%），Vision MLP 是最大块。
- attention 实现为 sdpa，V4（QK/PV 量化）前必须先做 eager-backend equivalence gate。

## 5. 问题与后续

- V0 PASS，可进入 V1（Connector quantization）。
- 下一步：V1-R raw equivalence → V1-FP8 → calibration → task0×1 → Goal×100。
- V4 前置：sdpa → eager 等价性确认（手册 §12.2）。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-15 | 创建文档 | |
| 2026-09-15 | 回填 Gate L0 + V0（结构/运行时/FLOPs）结果 | |
| 2026-09-15 | 新增 §3.3 FLOPs 占比饼图（V0 实测 + 手册 §0 整体推理） | |
