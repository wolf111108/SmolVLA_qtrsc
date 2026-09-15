# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**（区别于 `results.md` 的结果记录）。
> 每次启动/完成一次运行、发现或修复异常时追加一条；章节为固定结构，不可删除；无内容写「无」。
> 运行日志按时间**正序**追加，最新状态反映在「当前状态」与头部字段。

- **实验名称**：2026-09-15_phaseI_vision-quantization
- **状态**：running（draft / running / done / aborted）
- **最后更新**：2026-09-15

---

## 1. 当前状态

Gate L0（legacy regression）与 V0（Vision workload audit）已完成并通过，可进入 V1（Connector quantization）。

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| Gate L0 legacy regression | ✅ done | 2026-09-15 | 224/64/0/0，向后兼容 |
| V0 workload audit | ✅ done | 2026-09-15 | 12 层 / 768 / 2 camera / 428 GFLOPs |
| V1 Connector quant | ⏳ pending | — | 待启动 |

## 2. 运行日志

| 日期 | 阶段 | 命令 / 配置 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-09-15 | Gate L0 | `gate_l0_legacy_regression.py --config g6a_all_fp8_control.yaml` | ✅ | — | 224 Linear/64 MatMul/0 vision/0 connector |
| 2026-09-15 | V0 | `audit_vision_structure.py --config v0_workload_audit.yaml` | ✅ | `outputs/.../v0_workload_audit` | 1ep，SR=100%，cameras/sample=2.0 |

## 3. 后台任务

| PID | 阶段 | 启动时间 | 状态 | 备注 |
|---|---|---|---|---|
| — | — | — | — | 无（Phase H H3 与 G6 在并行） |

## 4. 异常与处理

- V0 FLOPs 首版把 `vision_forward_calls=38`（整个 rollout 总数）误当 camera 数，导致 FLOPs 被 ×38。修正为 hook `_get_action_chunk` 计数 sample_actions 次数，得 `cameras_per_sample_actions=2.0`（19 sample_actions × 2 camera = 38 calls）。FLOPs 按「per sample_actions」语义重算。
- 关键发现：vision attention 实现为 `sdpa`（非 eager），V4 前须先做 eager-equivalence gate（手册 §12.2）。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-15 | 创建文档 | |
| 2026-09-15 | 回填 Gate L0 + V0 进度 | |
