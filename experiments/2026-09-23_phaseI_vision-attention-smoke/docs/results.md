# 结果记录（Results）

- **实验名称**：2026-09-23_phaseI_vision-attention-smoke
- **状态**：draft
- **最后更新**：2026-09-23

## 1. 摘要

已创建 Vision QK/PV 接入与单 episode 验证实验。用户要求本地运行，checkpoint / LIBERO 验证未执行。

## 2. 总结果表

| 组 | 结果 | 输出 |
|---|---|---|
| 小型 attention 单元测试 | 6 passed（CPU torch 2.14.0+cpu / transformers 4.52.4，在用户要求停止验证前完成） | 非 benchmark |
| checkpoint raw gate | pending | ../outputs 路径见 setup |
| VM1 Goal task0×1 | pending；无 SR 数据 | 见 setup |

## 3. 分组结果与分析

小型单元测试包含 eager/SDPA、masked/unmasked、scale 重载及量化统计、异常时上下文恢复。未运行完整 LeRobot 模型。

## 4. 结论

代码与验证入口已提供；真实模型接入及闭环结果等待本地回填。

## 5. 问题与后续

运行 run_smoke.sh，回填 preflight.json、coverage_summary.json 与 episode 成功情况。若 raw gate 失败，先分析版本和 dtype 差异，不直接放宽容差。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-23 | 创建实验与待运行记录 | |
