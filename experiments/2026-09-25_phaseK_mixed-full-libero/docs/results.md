# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-25_phaseK_mixed-full-libero
- **状态**：draft（draft / running / done / aborted）
- **最后更新**：2026-09-25

---

## 1. 摘要

待运行。固定前序混合精度配置，评测四 suite；本文件尚无实测结果。

<!-- 3-5 句话：做了什么、核心数字、最重要的结论 -->

## 2. 总结果表

<!-- 每个 config 一行；数值带单位；与 baseline 的差值单独列 -->

| Suite | B0 SR | M0 SR | Δ pp | B0/M0 episodes | 输出目录 |
|---|---:|---:|---:|---|---|
| Spatial | 待填 | 待填 | 待填 | 100/100（计划） | [原始输出](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/libero_spatial/) |
| Object | 待填 | 待填 | 待填 | 100/100（计划） | [原始输出](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/libero_object/) |
| Goal | 待填 | 待填 | 待填 | 100/100（计划） | [原始输出](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/libero_goal/) |
| Long | 待填 | 待填 | 待填 | 100/100（计划） | [原始输出](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/libero_10/) |
| 总体 | 待填 | 待填 | 待填 | 400/400（计划） | [汇总](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/) |

## 3. 分组结果与分析

<!-- 按实验变量分组展开：每组建一个小节，含数据表、图、简要分析 -->

### 3.1 稀疏度与计算量（待填）

| 取值 | Success Rate | 备注 |
|---|---|---|

分析：待填。分别回填四 suite 与 pooled 的 Vision、VLM、Vision+VLM、Expert runtime 元素/S|MMM 位稀疏度，及 INT8/INT4 静态权重统计。

计算量从 compute_summary.csv 回填各 component 的总 GFLOPs、GFLOPs/generation、GFLOPs/episode；connector/other 单列。禁止累加 Vision+VLM 与 Vision/VLM 三行造成重复。逐任务分析使用 task_success.csv 的成败翻转，不能把 SR 相同写成逐 episode 一致。

原始汇总：[success_summary.csv](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/success_summary.csv)、[sparsity_summary.csv](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/sparsity_summary.csv)、[compute_summary.csv](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/compute_summary.csv)。

## 4. 结论

<!-- 对 §1「实验目的」中每个问题的直接回答；可执行的结论（保留/否决某配置） -->

-

## 5. 问题与后续

<!-- 实验中发现的异常、失败 run、待验证问题、下一步实验（链接到新实验子目录） -->

-

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-25 | 创建文档 | |
