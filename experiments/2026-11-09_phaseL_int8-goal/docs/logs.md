# Phase L — 运行日志

## 当前状态

implemented / pending GPU run

## 运行日志

| 日期 | 操作 | 状态 | 输出 |
|---|---|---|---|
| 2026-11-09 | 新建 baseline、a8w8、a8w4_expert；复用既有量化/校准/评测接口 | 完成 | configs / scripts |
| 2026-11-09 | `python .../scripts/run_arm.py preflight` | PASS | 三组同协议配置检查 |
| 2026-11-09 | `python .../scripts/test_experiment.py` | PASS，10 tests | 合成 CSV 聚合、精度/覆盖/旁路记录拒绝、N/A baseline、完整汇总 |
| 2026-11-09 | `bash -n .../scripts/run_all.sh` | PASS | shell 语法检查 |
| — | GPU 校准 / smoke / Goal 评测 | 未运行 | `outputs/2026-11-09_phaseL_int8-goal/` 待生成 |

## 后台任务

无。

## 异常与处理

开发环境缺少 PyTorch / GPU。未启动 GPU 作业，未将合成测试结果填写为实验结果。

## 修订记录

- 2026-11-09：建立实验与检查记录。
