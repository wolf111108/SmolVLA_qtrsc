# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**（区别于 `results.md` 的结果记录）。
> 每次启动/完成一次运行、发现或修复异常时追加一条；章节为固定结构，不可删除；无内容写「无」。
> 运行日志按时间**正序**追加，最新状态反映在「当前状态」与头部字段。

- **实验名称**：2026-09-25_phaseK_mixed-full-libero
- **状态**：running（draft / running / done / aborted）
- **最后更新**：2026-09-28

---

## 1. 当前状态

prepare/calibrate/smoke 已完成；Spatial baseline 完成（86/100），Spatial quant 首次运行中断后已归档并续跑中；Object / Goal / Long 待跑。

<!-- 一句话概括当前进度；下面用阶段状态表追踪各阶段（H0/H1/H2/...） -->

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| prepare | done | 2026-09-28 | commit/packages/resolved YAML 齐全 |
| calibrate | done | 2026-09-28 | 耗时 6697s；scale_audit.json 已落盘 |
| smoke | done | 2026-09-28 | Goal task0 × 1ep 通过 |
| Spatial baseline | done | 2026-09-28 03:57 | SR = 86/100 |
| Spatial quant | running | — | 首跑中断（见异常），已归档续跑 |
| Object B0/M0 | pending | 无 | — |
| Goal B0/M0 | pending | 无 | — |
| Long B0/M0 | pending | 无 | — |

## 2. 运行日志

<!-- 每次运行一条；记录命令/配置、状态、输出目录、耗时、SR 等关键信息 -->

| 日期 | 阶段 | 命令 / 配置 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-09-25 | 创建 | 见 experiment_setup.md | draft | outputs/2026-09-25_phaseK_mixed-full-libero/ | 等待本地运行 |
| 2026-09-28 | prepare/calibrate/smoke | run_all.sh | done | prepared.json、calibrate/、smoke/ | calibrate 6697s；smoke Goal task0×1ep 通过 |
| 2026-09-28 | Spatial baseline | run_arm.py baseline --suite libero_spatial | done | libero_spatial/baseline/ | SR = 86/100（10 tasks × 10 ep） |
| 2026-09-28 | Spatial quant（首跑） | run_arm.py quant --suite libero_spatial | interrupted | failed/libero_spatial_quant_0928/ | 约 04:09 中断于 task0 ep3 rollout 21%，无报错记录 |
| 2026-09-28 | Spatial quant（续跑） | run_arm.py quant --suite libero_spatial（setsid nohup，GPU 0） | running | libero_spatial/quant/ | 日志 run_resume_0928.log；复用已完成 prepare/calibrate |

## 3. 后台任务

<!-- 长时运行任务：PID、启动时间、预计时长、当前进度；任务结束后移入运行日志并清理本条 -->

| PID | 阶段 | 启动时间 | 状态 | 备注 |
|---|---|---|---|---|
| 1086044 | Spatial quant 续跑 | 2026-09-28 14:46 | running | setsid nohup 守护；日志 run_resume_0928.log；预计约 3h（GPU 0 与他任务共用） |

## 4. 异常与处理

### 2026-09-28：Spatial quant 首跑中断

- **现象**：进程于 04:09 左右消失，中断于 task0 ep3 rollout 21%（58/280 步）；`libero_spatial/quant/` 下只有 config.yaml、scale_audit.json 与空 videos/，无 completed.json / result.json。
- **定位**：run.log 无 Traceback、无 CUDA OOM、无 killed 记录，判断为终端会话断开或进程被外部终止，非代码报错。
- **处理**：按实验文档规则将失败目录整体归档至 `failed/libero_spatial_quant_0928/`，以 `setsid nohup` 方式在 GPU 0 续跑（避免再次因会话断开中断）；prepare/calibrate 已完成不重跑。
- **备注**：GPU 0 有其他长任务占用约 33GB/97GB 显存，正式评测吞吐可能受影响。

<!-- 运行中发现的报错、定位过程、修复措施（链接到 commit/PR） -->

-

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-25 | 创建实验；语法、形状 MAC 公式、四 suite 汇总与缺失输入拒绝检查通过；未运行模型、未安装 PyTorch | |
| 2026-09-28 | 回填 prepare/calibrate/smoke/Spatial baseline 结果；记录 Spatial quant 首跑中断与归档续跑 | lfwang |
