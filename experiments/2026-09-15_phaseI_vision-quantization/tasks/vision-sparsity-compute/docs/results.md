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
| **Vision prefill** | 144 | **4.08%** | **41.97%** | 72 | 5.6e-6 ratio（≈0.00056%） | 41.95% |
| VLM prefill | 320 | 1.92% | 41.69% | 112 | 5.5e-6 ratio（≈0.00055%） | 40.86% |
| Expert denoise（10 step） | 3200 | 3.02% | 42.15% | 112 | 3.3e-6 ratio（≈0.00033%） | 42.58% |
| all_quantized pooled | 3664 | 3.44% | 42.01% | 296 | 4.9e-6 ratio（≈0.00049%） | 41.63% |

**要点**

- **Vision runtime element sparsity（4.08%）≈ VLM（1.92%）的 2.1 倍、Expert（3.02%）的 1.35 倍**：Vision 在三者中相对最适合 element zero-skipping；但绝对稀疏度仍只有约 4%，不能据此推导显著实际 speedup。
- 三 component 的 E4M3 significand bit sparsity 高度一致（40.9–42.6%），与 Phase H 结论同源（INT8 runtime ~72.9%、INT16 ~62.8%不可直接对比：口径不同，此处为 E4M3 significand zero-bit metric）。
- static weight element sparsity ≈ 0（4.9e-6）：量化后权重无显著零元素（native 口径，未计入 outlier protection 的 artificial zero）。

### 3.2 Compute coverage — VSC-1 修正后结果（✅ 2026-09-22）

| component | block | quantized | GMAC/sample_actions | GFLOPs/sample_actions | 占 audited major ops |
|---|---|---|---:|---:|---:|
| vision | Vision 72 Linear | ✅ | 173.95 | **347.89** | **58.53%** |
| vision | Vision SDPA QK | ❌ raw | 19.33 | 38.65 | 6.50% |
| vision | Vision SDPA PV | ❌ raw | 19.33 | 38.65 | 6.50% |
| connector | Connector projection | ❌ raw | 1.51 | 3.02 | 0.51% |
| vlm | VLM quantized（Linear+QK/PV） | ✅ | 28.80 | **57.60** | 9.69% |
| expert | Expert quantized（10 denoise） | ✅ | 54.29 | **108.59** | 18.27% |
| **TOTAL** | **audited major ops** | **86.5% 覆盖** | 297.21 | **594.41** | 100% |

量化 major-op：

[
347.89 + 57.60 + 108.59 = 514.08 {m GFLOPs}
]

因此：

[
rac{514.08}{594.41}=oxed{86.5%}
]

raw audited major-op 约为 **13.5%**，其中 Vision QK/PV = **77.31 G ≈ 13.0%**，connector = **3.02 G ≈ 0.5%**。

**VSC-1 与 VSC-0 / 手册 sanity check**

| 项 | VSC-0（INVALID） | **VSC-1（修正）** | 手册 §0 粗估 | 说明 |
|---|---:|---:|---:|---|
| VLM | 110.04 G | **57.60 G** | 57.6 G | 一致 |
| Expert | 150.24 G | **108.59 G** | 107.5 G | +1.0% |
| TOTAL | 688.50 G | **594.41 G** | 595.7 G | −0.2% |
| coverage | 88.3% | **86.5%** | — | VSC-1 为有效值 |

VSC-1 与旧手册粗估在数值上高度一致，可作为强 sanity check；但两者统计 scope 并非完全相同——VSC-1 是 **audited major ops**，明确排除 patch embed / LayerNorm / softmax / GELU / pixel-shuffle bookkeeping，因此不应描述为严格相同口径的“独立三路证明”。

VSC-0 的 688.50 G / 88.3% 已永久标记为 invalid。根因是旧版 `export_workload_csv()` 对 MatMul A/B/O 共用 role-agnostic `module_last_dims`，O role 最终覆盖前两者。当前实现改为：

[
oxed{mathrm{MACs}_{mathrm{MatMul}}=operatorname{numel}(O)	imes A.shape[-1]}
]

并按 `(module_id, phase, flow_step, attention_kind)` 累计 physical MatMul call；新 workload 行必须带 `MAC_semantics=matmul_physical_exact_v2`，summarizer 会拒绝 legacy 数据。

## 4. 结论

1. **稀疏度画像已闭环**：Vision/VLM/Expert runtime native element sparsity分别为 **4.08% / 1.92% / 3.02%**，E4M3 significand bit sparsity均约 **42%**。Vision element sparsity相对最高，但绝对值仍低，因此 element zero-skipping 的空间有限。
2. **static weight element sparsity几乎为零**（pooled ratio 4.9e-6，约 0.00049%），传统 weight element-zero skipping 基本没有收益；但 weight bit sparsity仍约 **41–43%**，bit-serial / bit-skip 路径仍值得研究。
3. **compute accounting 已经由 VSC-1 修正闭环**：audited major-op total = **594.41 GFLOPs/sample_actions**，其中 **514.08 G（86.5%）** 已进入当前 FP8 quantized scope。Vision 72 Linear 单独占 **347.89 G / 58.53%**，是最主要的已量化算力块。
4. 当前 raw audited major ops 为约 **13.5%**：Vision QK/PV 约 **13.0%**，connector 约 **0.5%**。若进入 V4，理论上主要针对的是 QK/PV 这 13.0%，而不是全部剩余 13.5%。
5. routing / sparsity coverage Gate 与 compute Gate 均闭合：manifest 360、Vision Linear 72、VLM/Expert 112+32、weight rows 296、Expert flow_step 0–9、T12 regression 通过，VSC-1 summarizer GATE PASS。

## 5. 问题与后续

- **VSC-1 workload-fix rerun 已完成**：VLM = 57.60 G、Expert = 108.59 G、audited total = 594.41 G、coverage = 86.5%；compute instrumentation 问题已闭环。
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
