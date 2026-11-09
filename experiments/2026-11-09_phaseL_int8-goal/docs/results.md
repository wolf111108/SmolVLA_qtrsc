# Phase L — INT8 Goal 结果

状态：**implemented / pending GPU run**。尚无本实验成功率或稀疏度实测值。

## 摘要

已添加三组配置、独立校准、smoke、Goal 100 episodes 评测和统计汇总。两组量化均包含 QK/PV INT8。baseline 采用相同 eager attention 路径。

## 总结果表

| Arm | Goal successes / episodes | SR | Δ baseline | VLM activation bit sparsity | VLM weight bit sparsity | Expert activation bit sparsity | Expert weight bit sparsity |
|---|---|---|---|---|---|---|---|
| baseline | 待测 / 100 | 待测 | — | N/A | N/A | N/A | N/A |
| a8w8 | 待测 / 100 | 待测 | 待测 | 待测 | 待测（W8） | 待测 | 待测（W8） |
| a8w4_expert | 待测 / 100 | 待测 | 待测 | 待测 | 待测（W8） | 待测 | 待测（W4） |

## 分组结果与分析

运行完成后，从 `outputs/2026-11-09_phaseL_int8-goal/summary.md` 填入上表；逐任务及激活子类别结果分别见 `task_success.csv`、`sparsity_by_component.csv`。不混用历史 Goal/Object 结果作为本轮 baseline。

## 结论

当前仅确认实验配置和汇总逻辑通过 CPU 检查。量化成功率损失、Expert W4 的影响及真实稀疏度均待 GPU 实测。

## 问题与后续

- 开发环境无 PyTorch / CUDA GPU，未执行模型加载、校准或 LIBERO rollout。
- 在项目评测环境运行 `scripts/run_all.sh`，保留原始输出，补齐本表与运行日志。
- 指标定义、outlier 旁路及 Vision 保留浮点范围见 [experiment_setup.md](experiment_setup.md)。

## 修订记录

- 2026-11-09：新增实验与待测结果表；CPU 回归测试使用合成数据，无实测值。
