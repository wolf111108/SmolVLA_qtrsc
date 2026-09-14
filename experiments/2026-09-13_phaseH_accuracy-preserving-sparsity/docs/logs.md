# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**（区别于 `results.md` 的结果记录）。
> 每次启动/完成一次运行、发现或修复异常时追加一条；章节为固定结构，不可删除；无内容写「无」。
> 运行日志按时间**正序**追加，最新状态反映在「当前状态」与头部字段。

- **实验名称**：2026-09-13_phaseH_accuracy-preserving-sparsity
- **状态**：running（draft / running / done / aborted）
- **最后更新**：2026-09-15

---

## 1. 当前状态

**H2（30ep convergence）已完成**（2026-09-15 00:35，20/20 tasks 全部 PIPELINE COMPLETED，无报错）。结果：**S0（FP8-all）= 90.0%**、**S1（Expert-W4）= 86.7%**（各 10 tasks × 3ep = 30 episodes），gap = 3.3pt。当前处于 **H2 收敛判断** 阶段，待判定后启动 H3 100ep formal。

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| H0 correctness smoke | ✅ done | 2026-09-14 | S0/S1 SR=100%，Gate 全 PASS |
| H1 10ep pilot | ✅ done | 2026-09-14 | S0=90.0%、S1=80.0%（各 10 tasks × 1ep） |
| H1-Audit | ✅ done | 2026-09-14 | output/O 0.5%→39.9% 统计 bug 定位+修复 |
| H2 30ep convergence | ✅ done | 2026-09-15 | S0=90.0%、S1=86.7%（各 10 tasks × 3ep），gap=3.3pt |
| H3 100ep formal | ⏳ pending | — | 待 H2 收敛判断后启动 |

## 2. 运行日志

| 日期 | 阶段 | 命令 / 配置 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-09-13 | — | 创建实验骨架 | ✅ | — | configs/scripts/docs 就绪 |
| 2026-09-14 | H0 smoke | S0/S1 task0 × 1ep | ✅ | `h0_smoke/s0|s1/task00` | SR=100%，Gate 全 PASS |
| 2026-09-14 | H1 pilot | S0/S1 各 10 tasks × 1ep | ✅ | `h1_10ep/s0|s1/taskXX` | S0=90.0%、S1=80.0% |
| 2026-09-14 | H1-Audit | `s0_fp8_all_h1_audit_task0.yaml`（S0 task0 × 1ep，`--skip-calibration`） | ✅ | `h1_audit/s0/task00` | output/O 39.90%/39.91%，与独立 E4M3 audit 一致 |
| 2026-09-14 | H2 | S0/S1 各 10 tasks × 3ep | ❌→✅ | `h2_30ep/s0|s1/taskXX` | 首次启动因 conda 环境未激活报 ModuleNotFoundError，修复脚本后重启（见 §4） |
| 2026-09-15 | H2 | S0/S1 各 10 tasks × 3ep | ✅ | `h2_30ep/s0|s1/taskXX` | 20/20 完成；S0=90.0%、S1=86.7%；S0 task06=33.3%、task07=66.7%，S1 task03=66.7%、task06=33.3%、task07=66.7% |
| 2026-09-15 | H2 汇总 | `build_results_tables.py --stage h2_30ep --compare h1_10ep` | ✅ | — | results.md 回填 §14 全表；§13 收敛 gate 同口径全部 PASS；output/O INVALID 解除 |

## 3. 后台任务

| PID | 阶段 | 启动时间 | 状态 | 备注 |
|---|---|---|---|---|
| 3542208 | H2 30ep | 2026-09-14 | ✅ finished | `run_h2_30ep.sh`，S0+S1 各 10 tasks × 3ep，20/20 完成，日志 `outputs/.../h2_run.log` |

## 4. 异常与处理

- **H1 output/O 稀疏度误报 0.3-0.6%**：`quant_methods.py` 3 处 `out_normal_quant.to(torch.float32).mul_(M_q)` 为 in-place 操作，把 quantized code（`[-448,448]`）原地改写成 dequant 巨大值，后续收集 output_code 时值 `>448` round 到 E4M3 变 NaN，稀释 sparse_bit_rate。修复为 `mul`（非 in-place），forward 数值不变（SR 保持 100%）。见 commit `e4034aa`，回归测试 T11。
- **H2 首次启动失败（ModuleNotFoundError: No module named 'vla_tcs2'）**：`run_h2_30ep.sh` 用裸 `python`，在未激活 `smolvla_eval` 的 shell 里后台启动，导致找不到 `vla_tcs2` 包。修复为 `conda run -n smolvla_eval python`（与 G6 脚本一致），并同步修复 `run_h3_100ep.sh`。
- **`summarize_sparsity.py` 从仓库根无法运行（FileNotFoundError）**：`REPO_ROOT = HERE/../..` 只上溯到 `experiments/`，应为 `../../..`。已修复。
- **H0/H1 的「全口径」aggregate 不可与 H2 直接比较**：H0/H1 的 output/O 行为 pre-fix 污染值（≈0），使全口径 aggregate 降到 25-27%、unit sparsity 抬到 42-44%。已新增 `build_results_tables.py --exclude-roles output,O` 提供同口径对比（H1 vs H2 差 0.01-0.02pp）。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-14 | 创建文档，记录 H0/H1/H1-Audit 完成状态与 H2 启动 | |
| 2026-09-15 | 回填 H2 完成状态与结果（S0=90.0%、S1=86.7%） | |
| 2026-09-15 | 回填 H2 sparsity 汇总到 results.md（§14 全表 + §13 收敛 gate）；新增 `build_results_tables.py`，修复 `summarize_sparsity.py` 路径 bug | |
