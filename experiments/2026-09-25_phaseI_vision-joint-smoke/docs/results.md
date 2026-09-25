# 结果记录（Results）

- **实验名称**：2026-09-25_phaseI_vision-joint-smoke
- **状态**：draft
- **最后更新**：2026-09-25

## 1. 摘要

实验已创建，未运行；没有成功率或稀疏度结果。

## 2. 总结果表

| 组 | 协议 | Episode success | Runtime element / bit | Weight element / bit | 状态 |
|---|---|---|---|---|---|
| VJ1 | Goal task0×1 | 待回填 | 待回填 | 待回填 | pending |

## 3. 分组结果与分析

待回填 preflight、288 scale审计、96 manifest、216 runtime与72 weight覆盖，以及 Linear/MatMul 分组统计。

## 4. 结论

暂无。单 episode 结果仅作工程验证。

## 5. 问题与后续

本地执行 setup 中的 runner，提交原始CSV及JSON以便独立复核。通过后再设计全模型分组件 W8/W4 混合精度评测。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-25 | 创建联合验证实验 | |
