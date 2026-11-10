# Phase N — activation BG p 扫描结果

## 摘要

状态：**implemented / pending GPU run**。已完成真实量化编码采集、七个 p 的同步统计、单 Goal task runner 与 CPU 验证。尚无模型任务实测的稀疏率。

## 总结果表

| p | VLM native BG skip | Expert native BG skip | VLM full-pack zero | Expert full-pack zero |
|---|---|---|---|---|
| 2 | 待测 | 待测 | 待测 | 待测 |
| 4 | 待测 | 待测 | 待测 | 待测 |
| 8 | 待测 | 待测 | 待测 | 待测 |
| 16 | 待测 | 待测 | 待测 | 待测 |
| 32 | 待测 | 待测 | 待测 | 待测 |
| 64 | 待测 | 待测 | 待测 | 待测 |
| 128 | 待测 | 待测 | 待测 | 待测 |

运行后从 `outputs/2026-11-10_phaseN_bg-p-sweep/bit_groups/summary.md` 填入。对论文 p 分析优先使用 `bsc_left`；同时保留 `all_inputs` 供前次 activation 口径对照。

## 分组结果与分析

CPU 合成数据仅验证统计逻辑，不填入上述实测表。完整 p-token pack 与尾块分别记录；缺少完整 pack 时 `full_native_pack_zero_rate=N/A`，不可填 0 或当成完整 p 测量。

## 结论

实现与 CPU 检查完成；p 增大时真实可跳过 BG 数量如何变化，需等待单 task 推理结果。

## 问题与后续

当前执行环境无 CUDA、LeRobot、LIBERO、模型/数据/尺度缓存及实验服务器连接。已补齐项目环境可直接运行的命令。实测后保留原始 CSV，并核查 p=64/128 的尾块占比。

## 修订记录

- 2026-11-10：新增扫描实验；所有实测值保留为待测。
