# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-15_phaseI_vision-quantization / task: vision-sparsity-compute
- **状态**：done
- **最后更新**：2026-09-22

---

## 1. 摘要

VSC-0 已完成（2026-09-22，Gate 全 PASS）。在完整 VLIN 配置（360 quant modules，严格复用 VLIN=80% 的全部 scales，`calibration_policy: reuse` + `--skip-calibration`）下，Goal task0×1ep 的 workload characterization 得到：①**加入 Vision 72 Linear 后，quantized major-op FLOP 覆盖达 88.3%**（audited major ops 合计 688.50 GFLOPs/sample_actions，其中 608.17 G 已量化）；②Vision 的 runtime element sparsity（4.08%）约为 VLM（1.92%）的 2.1 倍，三 component 的 E4M3 significand bit sparsity 高度一致（~42%）；③Vision 72 Linear 实测 347.89 GFLOPs/sample_actions，与 V0 架构拆分完全一致（sanity PASS）。本实验仅做 workload 表征，不做新的 accuracy claim。

## 2. 总结果表

| 组 | config | runtime element sparsity | runtime bit sparsity | weight element sparsity | weight bit sparsity | audited GFLOPs | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| VSC-0 | vsc_vlin_fp8_task0_1ep | 3.44%（pooled） | 42.01%（pooled） | ~0%（4.9e-6） | 41.63%（pooled） | 688.50（FP8 覆盖 88.3%） | ✅ done |

## 3. 分组结果与分析

### 3.1 Vision / VLM / Expert sparsity

| component | runtime rows | runtime elem sparsity | runtime bit sparsity | weight rows | weight elem sparsity | weight bit sparsity |
|---|---:|---:|---:|---:|---:|---:|
| **Vision prefill** | 144 | **4.08%** | **41.97%** | 72 | 5.6e-6 | 41.95% |
| VLM prefill | 320 | 1.92% | 41.69% | 112 | 5.5e-6 | 40.86% |
| Expert denoise（10 step） | 3200 | 3.02% | 42.15% | 112 | 3.3e-6 | 42.58% |
| all_quantized pooled | 3664 | 3.44% | 42.01% | 296 | 4.9e-6 | 41.63% |

**要点**

- **Vision runtime element sparsity（4.08%）≈ VLM（1.92%）的 2.1 倍、Expert（3.02%）的 1.35 倍**：Vision FP8 激活中天然零元素最多，对 zero-skipping 类加速最友好。
- 三 component 的 E4M3 significand bit sparsity 高度一致（40.9–42.6%），与 Phase H 结论同源（INT8 runtime ~72.9%、INT16 ~62.8%不可直接对比：口径不同，此处为 E4M3 significand zero-bit metric）。
- static weight element sparsity ≈ 0（4.9e-6）：量化后权重无显著零元素（native 口径，未计入 outlier protection 的 artificial zero）。

### 3.2 Compute coverage

| component | block | quantized | GMAC/sample_actions | GFLOPs/sample_actions | 占 audited major ops |
|---|---|---|---:|---:|---:|
| vision | Vision 72 Linear | ✅ | 173.95 | **347.89** | 50.53% |
| vision | Vision SDPA QK | ❌ raw | 19.33 | 38.65 | 5.61% |
| vision | Vision SDPA PV | ❌ raw | 19.33 | 38.65 | 5.61% |
| connector | Connector projection | ❌ raw | 1.51 | 3.02 | 0.44% |
| vlm | VLM quantized（Linear+QK/PV） | ✅ | 55.02 | 110.04 | 15.98% |
| expert | Expert quantized（10 denoise） | ✅ | 75.12 | 150.24 | 21.82% |
| **TOTAL** | **audited major ops** | **88.3% 覆盖** | 344.25 | **688.50** | 100% |

**要点**

- **量化覆盖从 ~37%（VLM+Expert only）提升到 88.3%**：加入 Vision 72 Linear 后，audited major ops 中仅剩 Vision QK/PV（77.31 G，11.2%）+ connector（3.02 G，0.4%）为 raw。
- Vision 72 Linear 实测 347.89 GFLOPs，与 V0 架构公式（347.89 G）**完全一致**（sanity PASS）；Vision 全部主要算子 428.22 G 亦与 V0 拆分一致。
- 未量化部分（QK/PV 11.2%）正是 §5.1 后续 V4 的目标，且需先过 sdpa→eager 等价性 gate。
- 口径提醒：audited major ops 明确排除 patch embed / LayerNorm / softmax / GELU / pixel shuffle，不能冒充全部算术 FLOPs。

## 4. 结论

1. **VLIN 配置下 88.3% 的 audited major-op FLOPs 已入 FP8**（608.17/688.50 GFLOPs per sample_actions），代价为已测的 −10pp（VLIN=80% vs G6-A 90%）。
2. **Vision 是最大算力块（50.5%）且激活稀疏度最高**（elem 4.08%、bit 42%）：若后续引入 zero-skipping / 稀疏 kernel，Vision 是收益最大的部分。
3. 量化后权重几乎无零元素（elem sparsity ≈ 0）：**权重侧稀疏加速不可行**，稀疏收益只能在激活侧寻找。
4. 全部 Gate PASS（manifest 360、Vision 72、VLM/Expert 112+32、weight rows 296、Expert flow_step 0–9、Vision compute ≈ 347.89 G）。

## 5. 问题与后续

- V4（Vision QK/PV 量化，覆盖剩余 11.2%）前置：sdpa → eager 等价性 gate。
- 可选：基于本实验的 per-operator sparsity（`sparsity_by_operator.csv`）定位 Vision 内高稀疏度子集，评估选择性量化/跳零的收益。
- 本实验不构成新的 accuracy claim；VLIN 的精度代价以已有 Goal×100 为准。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-22 | 创建规范 task；实验设计统一放入 experiment_setup.md | |
| 2026-09-22 | 回填 VSC-0 结果：Gate 全 PASS；quantized FLOP 覆盖 88.3%（688.50 G audited）；Vision elem sparsity 4.08% ≈ 2.1× VLM；bit sparsity ~42% 三 component 一致；状态改 done | |
