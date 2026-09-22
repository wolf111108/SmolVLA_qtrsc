# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-15_phaseI_vision-quantization / task: vision-sparsity-compute
- **状态**：done（VSC-0 sparsity ✅ / compute ❌ invalid；**VSC-1 workload-fix ✅ compute 修正完成**）
- **最后更新**：2026-09-22

---

## 1. 摘要

VSC-0 于 2026-09-22 一次跑通，**稀疏度链路有效**：360 个 quant modules 路由正确，Vision/VLM/Expert runtime native sparsity 均有完整数据，static weight 296 rows，Expert flow_step 0–9 全覆盖。有效的 headline 为：Vision runtime element sparsity **4.08%**、VLM **1.92%**、Expert **3.02%**；三 component 的 E4M3 significand bit sparsity约 **42%**；static weight element sparsity接近 0。

但复核 compute 链路时发现一个 **P0 workload exporter bug**：旧版 `export_workload_csv()` 用 role-agnostic `module_last_dims[module_id]` 推导 MatMul 的 K/N，而 A/B/O 会依次覆盖同一个 entry，最终通常留下 O 的 shape。结果是 VLM/Expert QK/PV 的 MatMul MACs 被错误推导。因此 VSC-0 文档最初回填的 **688.50 GFLOPs / 88.3% quantized coverage 均判为 INVALID，不得引用**。

不受该 bug 影响的 compute 数据仍可保留：Vision 72 Linear = **347.89 GFLOPs/sample_actions**（Linear path）、Vision raw SDPA QK/PV = **77.31 G**（V0 已验证架构公式）、connector = **3.02 G**（结构公式）。仓库已修复 MatMul physical-MAC accounting，并新增 VSC-1 同条件 1ep rerun。

**VSC-1（workload-fix rerun）已于 2026-09-22 完成，Gate 全 PASS**：修正后的 audited major-op total = **594.41 GFLOPs/sample_actions**，quantized FP8 coverage = **86.5%**。VLM = 57.60 G、Expert = 108.59 G，与手册 §0 粗估（57.6 / 107.5 G）几乎完全一致，证实 VSC-0 的 MatMul MAC bug 已修复。稀疏度数字与 VSC-0 逐位一致（修复不影响 native sparsity counters）。

## 2. 总结果表

| 组 | config | runtime element sparsity | runtime bit sparsity | weight element sparsity | weight bit sparsity | audited GFLOPs | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| VSC-0 | vsc_vlin_fp8_task0_1ep | **3.44%（pooled，有效）** | **42.01%（pooled，有效）** | **~0%（有效）** | **41.63%（pooled，有效）** | **INVALID：旧 MatMul MAC accounting** | ⚠ sparsity valid / compute invalid |
| VSC-1 | vsc1_vlin_fp8_task0_1ep_workloadfix | 3.44%（与 VSC-0 一致） | 42.01%（一致） | ~0%（一致） | 41.63%（一致） | **594.41（coverage 86.5%）** | ✅ done |

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

### 3.2 Compute coverage — VSC-1 修正后结果（✅ 2026-09-22）

| component | block | quantized | GMAC/sample_actions | GFLOPs/sample_actions | 占 audited major ops |
|---|---|---|---:|---:|---:|
| vision | Vision 72 Linear | ✅ | 173.95 | **347.89** | 58.53% |
| vision | Vision SDPA QK | ❌ raw | 19.33 | 38.65 | 6.50% |
| vision | Vision SDPA PV | ❌ raw | 19.33 | 38.65 | 6.50% |
| connector | Connector projection | ❌ raw | 1.51 | 3.02 | 0.51% |
| vlm | VLM quantized（Linear+QK/PV） | ✅ | 28.80 | **57.60** | 9.69% |
| expert | Expert quantized（10 denoise） | ✅ | 54.29 | **108.59** | 18.27% |
| **TOTAL** | **audited major ops** | **86.5% 覆盖** | 297.21 | **594.41** | 100% |

**要点（VSC-1 vs VSC-0 invalid 值 vs 手册粗估）**

| 项 | VSC-0（INVALID） | **VSC-1（修正）** | 手册 §0 粗估 | 偏差 |
|---|---:|---:|---:|---|
| VLM | 110.04 G | **57.60 G** | 57.6 G | **0.0%** |
| Expert | 150.24 G | **108.59 G** | 107.5 G | **+1.0%** |
| TOTAL | 688.50 G | **594.41 G** | 595.7 G | **−0.2%** |
| coverage | 88.3% | **86.5%** | — | — |

- 修正后的 VLM/Expert/total 与手册 §0 粗估几乎完全一致（≤1%），三条独立路径（V0 架构公式 / 手册粗估 / VSC-1 runtime 实测）互验闭合。
- Vision 72 Linear = 347.89 G 与 VSC-0 相同（Linear path 不受 bug 影响），且与 V0 公式一致。
- 稀疏度数字与 VSC-0 逐位一致（4.08/1.92/3.02% elem、~42% bit），证实修复只影响 MatMul MAC shape accounting。

VSC-0 初始汇总曾得到：

```text
VLM quantized major ops    110.04 GFLOPs
Expert quantized major ops 150.24 GFLOPs
audited total              688.50 GFLOPs
quantized coverage          88.3%
```

**以上四项均不得继续引用。**

根因位于旧版 `src/vla_tcs2/quant/stat_manager.py::export_workload_csv()`：`collect_quant_tensor()` 对 MatMul 的 A/B/O 三个 role 都写入同一个 `module_last_dims[module_id]`，而 O 最后到达，因此 exporter 最终把 **O tensor shape 当成 MatMul operand K/N**。虽然 task summarizer 已经做到“一物理算子只取 A role”，但 A row 的 `MACs` 本身已经由错误的 K/N 生成，仍然会污染 VLM/Expert QK/PV compute。

本次审计已把 core accounting 改成：

[
oxed{	ext{MACs}_{
m matmul}=	ext{numel}(O)	imes K,quad K=A.shape[-1]}
]

并按 `(module_id, phase, flow_step, attention_kind)` 累计物理 MatMul call。新 exporter 给 MatMul 行写入：

```text
MAC_semantics = matmul_physical_exact_v2
```

summarizer 会拒绝旧 workload CSV，避免旧数据再次被误用。

**VSC-0 中仍然有效的 compute 子项：**

| block | GFLOPs/sample_actions | 状态 | 原因 |
|---|---:|---|---|
| Vision 72 Linear | **347.89** | ✅ valid | Linear MAC path不依赖 `module_last_dims`；且与 V0 公式完全一致 |
| ## 4. 结论

1. **稀疏度结果有效**：Vision/VLM/Expert runtime native element sparsity分别为 **4.08% / 1.92% / 3.02%**，E4M3 significand bit sparsity均约 **42%**。Vision 的 element sparsity相对最高，但绝对值仍只有约 4%，因此只说明“相对更稀疏”，**不能直接推导出显著 zero-skipping speedup**。
2. **static weight element sparsity接近 0**：这排除了传统“整元素为零”的 weight zero-skipping收益；但 weight bit sparsity仍约 41–43%，因此**不能写成‘权重侧稀疏加速不可行’**——bit-serial / bit-skip 类硬件仍可能利用 bit-level sparsity。
3. **VSC-0 compute headline 无效**：688.50 GFLOPs 与 88.3% coverage 因 MatMul shape accounting bug 作废。当前仅 Vision 72 Linear = 347.89 G、Vision QK/PV = 77.31 G、connector = 3.02 G 可安全引用。
4. routing / sparsity coverage Gate 本身仍 PASS：manifest 360、Vision Linear 72、VLM/Expert 112+32、weight rows 296、Expert flow_step 0–9 都是有效结构证据。

## 5. 问题与后续

- ~~第一优先级：跑 VSC-1 workload-fix rerun~~ ✅ 已完成（2026-09-22）：VLM = 57.60 G、Expert = 108.59 G、total = 594.41 G、coverage = 86.5%，与手册 §0 粗估互验闭合（≤1%）。
- V4（Vision QK/PV，覆盖剩余 13.0% audited FLOPs）现在具备决策依据；仍需 sdpa→eager equivalence gate。
- 可继续用 VSC-0/VSC-1 的 sparsity CSV 做 per-operator 稀疏度分析，因为本次 bug 只影响 workload MAC shape accounting，不影响 native sparsity counters。
- 硬件方向启示（结合准确 FLOPs 与稀疏度）：Vision 是最大算力块（58.5%）且 elem sparsity 最高（4.08%，但绝对值仍低）——element zero-skipping 空间有限；三 component ~42% 的 bit-level sparsity 更值得 bit-serial/bit-skip 类硬件探索。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-22 | 创建规范 task；实验设计统一放入 experiment_setup.md | |
| 2026-09-22 | 回填 VSC-0 首轮结果：sparsity + workload summary | |
| 2026-09-22 | **P0 审计修正：VSC-0 sparsity 有效，但 compute headline 作废**。发现旧 workload exporter 的 MatMul A/B/O 共用 `module_last_dims`，O role 覆盖导致错误 K/N；688.50 G / 88.3% 标记 INVALID。core 已改为 `O.numel() × A.shape[-1]` 物理 MAC accounting，并创建 VSC-1 rerun | |
| 2026-09-22 | **回填 VSC-1（workload-fix rerun）结果**：Gate 全 PASS（含 T12 回归 12 passed）；**audited total = 594.41 G，quantized coverage = 86.5%**；VLM 57.60 / Expert 108.59 G 与手册粗估几乎一致；稀疏度与 VSC-0 逐位一致；状态改 done | |
