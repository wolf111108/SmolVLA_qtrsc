# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**（区别于 `results.md` 的结果记录）。
> 每次启动/完成一次运行、发现或修复异常时追加一条；章节为固定结构，不可删除；无内容写「无」。
> 运行日志按时间**正序**追加，最新状态反映在「当前状态」与头部字段。

- **实验名称**：2026-09-25_phaseK_mixed-full-libero
- **状态**：done（draft / running / done / aborted）
- **最后更新**：2026-09-29

---

## 1. 当前状态

**全部运行已完成**（2026-09-29 19:46）。三个正式 suite 的 quant 均 PASS，已回填 `results.md`。baseline 统一采用前序 FP baseline（`mj332_A_na10_ep10`），Goal 跳过，不执行 summarize（决策 D）。

<!-- 一句话概括当前进度；下面用阶段状态表追踪各阶段（H0/H1/H2/...） -->

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| prepare | done | 2026-09-28 | commit/packages/resolved YAML 齐全 |
| calibrate | done | 2026-09-28 | 耗时 6697s；scale_audit.json 已落盘 |
| smoke | done | 2026-09-28 | Goal task0 × 1ep 通过 |
| Spatial baseline | done | 2026-09-28 03:57 | SR = 86/100（仅作旁证） |
| Spatial quant | done | 2026-09-28 18:23 | SR = 81/100（首跑中断后归档续跑，见 §4） |
| Object baseline | done | 2026-09-29 13:19 | SR = 89/100（仅作旁证） |
| Object quant | done | 2026-09-29 15:21 | SR = 94/100；eval 7266s |
| Goal B0/M0 | **skipped** | — | 已由单 suite 实验覆盖，M0=83 跨轮引用（见 §4） |
| Long baseline | **skipped** | — | 采用 FP baseline 69.0（见 §1.1） |
| Long quant | done | 2026-09-29 19:46 | SR = 74/100；eval 15843s（4.40h） |
| summarize | **skipped** | — | 决策 D：不执行 `summarize.py`，保留各 suite 原始产出人工读取 |

### 1.1 Baseline 参照数据（统一采用前序 FP baseline）

**唯一 baseline 来源**：`outputs/table2_repro_audit/06_simulator/mj332_A_na10_ep10_*`（2026-08-27）。
逐项核对协议完全一致：`chunk_size=50`、`n_action_steps=10`、`num_steps=10`、`seed=1000`、`pretrained_path=lerobot/smolvla_libero`、100 episodes。四个 suite 齐全（100 ep 各）：

| suite | FP baseline | 本轮 in-house raw | 差 | 采用 |
|---|---:|---:|---:|---|
| Spatial | **85.0** | 86 | +1 | FP baseline |
| Object | **95.0** | 89 | −6 | FP baseline |
| Goal | **89.0** | —（未跑） | — | FP baseline |
| Long | **69.0** | —（未跑） | — | FP baseline |

per-task 数据已从四个日志（`mj332_A_na10_ep10_logs/libero_*.log`）的 `Aggregated Metrics for per_task` 块解析。

> ⚠️ **可对比性风险（未消除）**：审计运行 `pretrained_revision=None`（**未钉 checkpoint 版本**）且时间较早（2026-08-27）。实测其与本轮 in-house raw 存在偏差（Spatial +1、**Object −6**）。因此本报告所有 Δ 均含未知口径漂移；严格结论需同轮 baseline。

## 2. 运行日志

<!-- 每次运行一条；记录命令/配置、状态、输出目录、耗时、SR 等关键信息 -->

| 日期 | 阶段 | 命令 / 配置 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-09-25 | 创建 | 见 experiment_setup.md | draft | outputs/2026-09-25_phaseK_mixed-full-libero/ | 等待本地运行 |
| 2026-09-28 | prepare/calibrate/smoke | run_all.sh | done | prepared.json、calibrate/、smoke/ | calibrate 6697s；smoke Goal task0×1ep 通过 |
| 2026-09-28 | Spatial baseline | run_arm.py baseline --suite libero_spatial | done | libero_spatial/baseline/ | SR = 86/100（10 tasks × 10 ep） |
| 2026-09-28 | Spatial quant（首跑） | run_arm.py quant --suite libero_spatial | interrupted | failed/libero_spatial_quant_0928/ | 约 04:09 中断于 task0 ep3 rollout 21%，无报错记录 |
| 2026-09-28 | Spatial quant（续跑） | run_arm.py quant --suite libero_spatial（setsid nohup，GPU 0） | done | libero_spatial/quant/ | SR = 81/100；eval 13008s；日志 run_resume_0928.log |
| 2026-09-29 | Object baseline | run_arm.py baseline --suite libero_object | done | libero_object/baseline/ | 13:07:56→13:19:27；SR = **89/100**；eval 649.4s |
| 2026-09-29 | Object quant | run_arm.py quant --suite libero_object | done | libero_object/quant/ | 13:19:29 启动，15:21 完成；SR = **94/100**；eval 7266s；PPID=1（链路 wrapper 已终止） |
| 2026-09-29 | Long quant（守护链路） | quant --suite libero_10（等 Object quant 退出后自动） | done | libero_10/quant/ | 15:21:27→19:46:10；SR = **74/100**；eval 15843s（4.40h）；日志 run_long_quant_0929.log |
| 2026-09-29 | 回填与基准变更 | — | done | docs/results.md | 统一改用前序 FP baseline（`mj332_A_na10_ep10`）；补全 Long；重算四 suite 逐 task、翻转与合计 |

## 3. 后台任务

<!-- 长时运行任务：PID、启动时间、预计时长、当前进度；任务结束后移入运行日志并清理本条 -->

| PID | 阶段 | 启动时间 | 状态 | 备注 |
|---|---|---|---|---|
| 1086044 | Spatial quant 续跑 | 2026-09-28 14:46 | done | 18:23 完成，SR = 81/100 |
| 1749318 | Object baseline→quant 链路 | 2026-09-29 13:07 | done | 13:19:29 被 kill 终止；其启动的 quant 子进程孤儿化后继续运行（见 PID 1756339） |
| 1754657 | Long 守护链路（旧） | 2026-09-29 13:16 | killed | 依赖链 1749318（已终止），已随基线停跑一并终止；不再使用 |
| 1756339 | Object quant | 2026-09-29 13:19 | done | 15:21 完成，SR = 94/100 |
| 1757471 | Long quant 守护链路 | 2026-09-29 13:22 | done | 19:46 完成（SR = 74/100）；守护进程随之退出 |

**全部进程已退出，无残留。**

## 4. 异常与处理

### 2026-09-29：baseline 统一改用前序实验的 FP baseline（协议偏离，已批准）

- **要求（分两步）**：（1）不再跑 baseline，直接跑 quant；（2）baseline **统一使用之前实验的 FP baseline**。
- **选定来源**：`outputs/table2_repro_audit/06_simulator/mj332_A_na10_ep10_*`（2026-08-27，FP/raw）。它与本实验协议**逐项一致**（`chunk_size=50`、`n_action_steps=10`、`num_steps=10`、`seed=1000`、`pretrained_path=lerobot/smolvla_libero`、100 ep），且**四个 suite 齐全**：Spatial 85 / Object 95 / Goal 89 / Long 69。
- **排除的候选**：Phase F 曾用的 `outputs/verify_libero/*` FP 参照（ep50）已不在磁盘上；其余 `na1_ep50` / `na30_ep50` / `na50_ep50` 的 `n_action_steps` 与协议不符。
- **处理**：
  - **Object baseline 实际已跑完**（13:19:27，早于终止命令生效），保留为**旁证**，但**不作为主参照**。
  - Spatial / Object 的本轮 in-house raw（86 / 89）同样仅作 §3.1 旁证。
  - **Long baseline 不跑**，直接采用 FP baseline **69**。
  - 旧的 Long 守护进程（1754657，会跑 baseline→quant）已作废终止。
- **数据可用性**：审计运行目录**只有 videos 与日志、无 `result.json`**；逐 task 数据从其日志的 `Aggregated Metrics for per_task` 块解析获得，已保存中转副本。
- **风险（未消除）**：审计运行 `pretrained_revision=None`（未钉 checkpoint 版本）、时间较早。实测与本轮 in-house raw 差异：Spatial 85 vs 86（+1）、**Object 95 vs 89（−6）**。**故本报告所有 Δ 均含未知口径漂移；严格结论需同轮 baseline。**（注：同轮三 suite 合计 Δ = 0.0 pp 在两种 baseline 下均成立。）

### 2026-09-29：跳过 Goal suite（协议偏离，已批准）

- **背景**：Goal 已由前序单 suite 实验 `2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal` 完整覆盖（commit `4cdccc4`，SR 87→83，100 ep，全 Gate PASS）。核对后确认两者**配置实质等价**：量化 overrides、checkpoint `31d453f7…`、`n_action_steps=10`、`num_steps=10`、`chunk_size=50`、seed 1000、10 tasks×10 ep 逐项一致；`src/` 框架代码在 `4cdccc4..b8115e2` 之间**零改动**。
- **决策**：本轮不重跑 Goal，改为 Spatial + Object + Long **三 suite**（各 400 ep→共 600 ep/组）。
- **后果**：`summarize.py` 硬编码 `SUITES`（四 suite）且断言 `episodes==400`，跳过 Goal 后会因 `libero_goal/baseline/result.json` 不存在而失败；`pooled` 与「四 suite 400 ep」表述也不再成立。
- **处理（决策 D，已采纳）**：**不执行 `summarize.py`**，也不生成根级 `summary.json` / `success_summary.csv` / `task_success.csv` / `sparsity_summary.csv` / `compute_summary.csv` / `outlier_summary.csv`；因此不修改 `summarize.py`。各 suite 原始产出（`result.json`、`eval_info.json`、`execution.json`、`compute.csv`、`compute_summary.json`、`coverage_summary.json`、`scale_audit.json`、`sparsity/*.csv`）保留在 `outputs/` 下供人工读取。
- **约束**（源码指纹，已随运行结束解除）：`run_arm.py` 每阶段会断言 `source_fingerprint()==prepared['sources']`（覆盖 `src/**/*.py`、`EXP/scripts/*.py|*.sh`、`EXP/configs/*.yaml`）。所有正式 stage 已于 2026-09-29 19:46 结束，该约束**不再生效**；本次仅修改 `docs/` 下的 md/json（不在指纹覆盖范围内），未动脚本与配置。

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
| 2026-09-29 | 回填 Spatial quant 完成（81/100）；启动 Object baseline→quant 链路并登记 | lfwang |
| 2026-09-29 | 记录跳过 Goal 的决策与理由、Long 守护链路 PID；登记 summarize.py 四 suite 硬依赖待决问题 | lfwang |
| 2026-09-29 | 汇总方式定为决策 D：不执行 summarize.py、不生成根级汇总文件，保留原始产出人工读取 | lfwang |
| 2026-09-29 | 按要求停跑剩余 baseline；后统一改用前序 FP baseline（`mj332_A_na10_ep10`）为主参照 | lfwang |
| 2026-09-29 | Object quant（94/100）与 Long quant（74/100）完成；实验转 done；新增 §1.1 FP baseline 参照表 | lfwang |
| 2026-09-29 | 按要求停跑剩余 baseline：Object baseline 已完成（89/100）予以保留；Long 改用审计同协议数据（69.0），只跑 quant；新增 §1.1 参照表与可对比性风险说明 | lfwang |
