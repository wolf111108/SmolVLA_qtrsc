# Phase N 运行记录

## 当前状态

implemented / pending GPU run

## 运行日志

- 2026-11-10：核对分支 `6a138f3` 与最新 `a8w4_expert` 配置，确认 Vision 浮点、QK/PV INT8、1% outlier。
- 2026-11-10：`run.py --preflight` 通过；配置只改变 task/episode、输出位置和统计开关。
- 2026-11-10：`tests/test_bit_group_sparsity.py` 的 10 个测试全部通过，包括独立 scalar oracle、batch/head 隔离、K 转置、-128、尾块、FP 旁路、跨线程采集和真实 Linear/QK/PV 前向逐元素等价性。使用 PyTorch 2.7.1 CPU；无跳过。
- 2026-11-10：脚本语法、Python 编译及 `git diff --check` 通过。
- 2026-11-10：前次 Phase L 的 10 项实验测试通过；既有 sparsity accounting 的 11 项可独立运行测试通过。`test_t9_runtime_hook_flow_step_sequencing` 依赖 LeRobot，当前环境未通过其 import 阶段，未验证。
- 2026-11-10：执行 `run.py --check-env`，在 `import lerobot` 处失败；同时确认没有 CUDA GPU。没有产生实测输出或完成标记。

## 后台任务

无。未启动完整数据集或 GPU 实验。

## 异常与处理

初始环境无 PyTorch；为测试安装了隔离的 CPU 版。无 CUDA 与 LeRobot/LIBERO，无法完成真实任务 rollout。未用合成数据代替实测。

## 修订记录

- 2026-11-10：新增配置、统计实现、验证结果与待运行说明。
