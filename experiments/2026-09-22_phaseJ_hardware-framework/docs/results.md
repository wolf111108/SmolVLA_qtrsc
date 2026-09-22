# 结果记录（Results）

## 1. 摘要

软件单元验证完成；真实模型GPU rollout尚未运行。

## 2. 总结果表

| 项目 | 结果 |
|---|---|
| 标准库单元测试 | 15 passed |
| PyTorch集成测试 | 3 skipped：当前环境未安装torch，需在smolvla_eval运行 |
| HW0/HW1 rollout | pending |
| 芯片校准 | 未进行 |

## 3. 分组结果与分析

已通过：尾块手算、broadcast MatMul、schema/shape检查、逻辑位宽、容量拒绝、outlier unsupported、截断、异常报告、同配置CLI重放一致性。

## 4. 结论

纯Python模型与trace链路具备首版实现；不能对真实模型端到端延迟或加速比作结论。

## 5. 问题与后续

在smolvla_eval运行全部集成测试和HW0/HW1，回填模块/调用数、截断和unsupported原因；再加入outlier多路映射、稀疏特征和存储层次。

## 6. 修订记录

- 2026-09-22：新增实现与软件验证结果，GPU项目保留pending。
