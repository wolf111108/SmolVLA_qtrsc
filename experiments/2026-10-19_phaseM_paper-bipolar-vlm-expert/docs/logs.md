# Phase M 运行日志

## 当前状态

软件实现和验证完成，完整 SmolVLA 推理等待用户本地运行。

## 运行日志

| 日期 | 命令/任务 | 状态 | 输出目录 |
|---|---|---|---|
| 2026-10-19 | `PYTHONPATH=src python -m unittest discover -s tests -p 'test_paper_bipolar.py' -q` | 14 项通过 | 临时目录 |
| 2026-10-19 | `PYTHONPATH=src python -m unittest discover -s tests -p 'test_hardware*.py' -q` | 36 项通过 | 临时目录 |
| 2026-10-19 | phaseL `scripts/check_export.py` | 1 项通过 | 临时目录 |
| 2026-10-19 | 小模型实际量化捕获；phaseM `replay.py --suite`、`plot.py` | 通过，10 组重放 | `outputs/2026-10-19_phaseM_paper-bipolar-vlm-expert/software-smoke/` |

小模型通过测试模块 `toy_capture(root, complete=True)` 生成，严格区别于完整 SmolVLA 工作负载。

## 后台任务

无。完整 GPU inference 未启动。

## 异常与处理

- 修复大 K 步长下先装载全部 p64 行导致过早逐出的风险：按 p8 子块消费并锁定 staging 工作集，超容量明确失败。
- 查表捕获选中地址，避免将 embedding 记成整个词表读入。
- 将坐标元数据就绪接入 BSC 依赖；补齐阶段 flush 的资源占用归属与 Perfetto 数字轨道 ID。
- 保持既有 O8 未饱和行为并审计；部分和按坐标替换 spill；metadata 也受容量约束。

## 修订记录

- 2026-10-19：依据已确认方案新增 paper 包、opt-in quant observer、实验配置、脚本和文档；既有 component_sim 保留兼容。
