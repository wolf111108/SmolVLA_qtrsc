# 结果记录（Results）

- **实验名称**：2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal
- **状态**：draft
- **最后更新**：2026-09-29

## 1. 摘要

待运行。新增 Vision W4 / VLM W8 / Expert W4，AFP8 PoT 的 Goal SR 实验；尚无本轮实测结果。

## 2. 总结果表

| 组 | Goal successes / episodes | SR | 来源 |
|---|---|---|---|
| raw eager | 87/100 | 87% | 前序实验，历史参照 |
| Vision W8 / VLM W8 / Expert W4 | 83/100 | 83% | 前序实验，历史参照 |
| Vision W4 / VLM W8 / Expert W4 | 待填 /100（计划） | 待填 | 本轮 quant.yaml |

## 3. 分组结果与分析

待填：task_success.csv 的 task 0–9 成功数；384 位点实际执行与精度路由 Gate；1152 scales 审计。两组历史参照详见[前序结果](../../2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/docs/results.md)。

本轮[原始输出](../../../../outputs/2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal/)，含 summary.json、task_success.csv。未收集稀疏度或计算量。

## 4. 结论

待运行，不预设 Vision W4 的精度结论。

## 5. 问题与后续

独立校准、单 seed；比较历史结果时注明来源和环境一致性，不将 SR 相同解释为逐 episode 相同。

## 6. 修订记录

| 日期 | 修改内容 |
|---|---|
| 2026-09-29 | 创建实验与结果模板 |
