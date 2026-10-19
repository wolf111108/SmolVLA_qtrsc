# Phase M 结果

状态：software-validated / full-inference-pending

## 摘要

软件数值、事务调度和小模型全阶段流程已验证。完整 SmolVLA 结果等待用户本地运行；本页不填估计性能数字。

## 总结果表

| 验证/实验 | 状态 | 产物 |
|---|---|---|
| 新增数学、捕获与事务检查 | 14 项通过 | `tests/test_paper_bipolar.py` |
| 既有硬件检查 | 36 项通过 | `tests/test_hardware*.py` |
| phaseL 导出兼容性 | 1 项通过 | phaseL `scripts/check_export.py` |
| 小模型 1 prefill + 10 denoise、10 组重放与绘图 | 通过 | 本地 `outputs/2026-10-19_phaseM_paper-bipolar-vlm-expert/software-smoke/`，不进 git |
| 完整 SmolVLA inference | 待运行 | `outputs/2026-10-19_phaseM_paper-bipolar-vlm-expert/int8_w4/` |

## 分组结果与分析

D0/D1/D2/D3 及带宽、理想单元和辅助吞吐敏感性分析均复用同一捕获。
本地完整运行后，先阅读 `replays/D3/report.md`，再对照 `comparison.json` 与时间线；记录命令、日期和输出目录后补充本节。

## 结论

软件级验证已完成；尚不能给出 SmolVLA 的延迟、EMA 或加速比结论。

## 问题与后续

在现有 GPU 环境运行一键脚本，核对输出溢出审计、实际辅助算子覆盖和输出动作。
用理想 memory/BSC 对照检查延迟主因；IAG 为 score-only，不评估任务精度损失或剪枝收益。

## 修订记录

- 2026-10-19：新增实验与软件验证记录；完整模型结果留空。
