# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-15_sparsity-ratio-quickscan
- **状态**：draft（draft / running / done / aborted）
- **最后更新**：2026-09-15

---

## 1. 摘要

待跑。本实验用 4 组量化配置（INT8 / INT16 / FP8(A/O)-W4+PoT(A/O) / FP8 PoT）在 `libero_goal task0 × 1ep` 下快速采集元素级与 bit 级 sparsity，按 `VLM prefill` 与 `Expert denoise` 两个 stage 分别汇总。**仅做 workload characterization，不做 SR claim。**

---

## 2. 总结果表

<!-- Primary 表（README §12）。数据源：quick_sparsity_summary.csv -->

| Config | Stage | Runtime element sparsity | Runtime bit sparsity | Weight element sparsity | Weight bit sparsity |
|---|---|---:|---:|---:|---:|
| INT8 | VLM prefill | | | | |
| INT8 | Expert denoise | | | | |
| INT16 | VLM prefill | | | | |
| INT16 | Expert denoise | | | | |
| FP8W4 PoT | VLM prefill | | | | |
| FP8W4 PoT | Expert denoise | | | | |
| FP8 PoT | VLM prefill | | | | |
| FP8 PoT | Expert denoise | | | | |

> 说明：INT bit sparsity 是 sign-aware sparse-bit；FP8 是 E4M3 4-bit significand zero-bit。二者不是同一编码定义，**不可直接相减**。FP8W4 PoT 的严格语义为 A/O PoT、W4 weight scale continuous。

---

## 3. 分组结果与分析

<!-- Secondary 表：按 role 细分。数据源：quick_sparsity_by_role.csv -->

### 3.1 按 tensor role 细分

| Config | Stage | Role | Element sparsity | Bit sparsity |
|---|---|---|---:|---:|
| | | Linear activation | | |
| | | Linear output | | |
| | | QK A / B / O | | |
| | | PV A / B / O | | |
| | | Weight q/k/v/o/gate/up/down | | |

角色语义：`PV A = softmax 后的 attention probability P`，`PV B = V`。

### 3.2 INT8 vs INT16

待跑。关注点：位宽增大后 quantization-induced element zeros 是否减少，sign-aware sparse-bit ratio 如何变化。

### 3.3 FP8 PoT vs FP8(A/O)-W4 PoT

待跑。预期：runtime A/O/QK/PV 主要仍是 FP8，因此 runtime sparsity 接近；差异应主要落在 static Linear weight（FP8 W vs INT4 W）。

### 3.4 Prefill vs Denoise

待跑。**必须分开报告**：每次 generation 中 VLM prefill 走 1 次，Expert denoise 走 10 个 flow step；相同 sparsity ratio 不代表相同总硬件节省。

---

## 4. 结论

待跑。

---

## 5. 问题与后续

- 本实验显式不做：正式 SR 对比、100ep rollout、unit/block sparsity、FP raw-code audit、BOP/hardware speedup、whole-model FLOP-weighted speedup。
- 如需 sparsity 收敛值（而非单 episode quickscan），应扩 episode 数；可参考 Phase H 的 H1(1ep) → H2(3ep) 收敛性检验。

---

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-15 | 创建文档（骨架 + Primary/Secondary 表结构） | |
