# 实验日志（Logs）

- **实验名称**：2026-09-23_phaseI_vision-attention-smoke
- **状态**：draft
- **最后更新**：2026-09-23

## 1. 当前状态

代码与配置已创建；checkpoint preflight、校准、rollout 等待用户本地运行。

## 2. 运行日志

| 日期 | 阶段 | 命令 | 状态 | 备注 |
|---|---|---|---|---|
| 2026-09-23 | 小型 CPU 单元测试 | python -m pytest tests/test_vision_attention.py -q | 6 passed | 用户要求停止运行验证前完成 |
| 2026-09-23 | checkpoint / rollout | scripts/run_smoke.sh | pending | 留给用户本地执行 |

## 3. 后台任务

无。

## 4. 异常与处理

无 checkpoint 或 rollout 运行记录；不得填写推测的成功率。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-23 | 创建日志；记录本地待运行状态 | |
