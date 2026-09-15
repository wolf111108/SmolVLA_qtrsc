# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**（区别于 `results.md` 的结果记录）。
> 每次启动/完成一次运行、发现或修复异常时追加一条；章节为固定结构，不可删除；无内容写「无」。
> 运行日志按时间**正序**追加，最新状态反映在「当前状态」与头部字段。

- **实验名称**：2026-09-13_phaseH_accuracy-preserving-sparsity
- **状态**：running（draft / running / done / aborted）
- **最后更新**：2026-09-15

---

## 1. 当前状态

**H2（30ep convergence）已完成**（2026-09-15，20/20 tasks 全部 PIPELINE COMPLETED，无报错）。结果：**S0（FP8-all）= 90.0%（27/30）**、**S1（Expert-W4）= 86.7%（26/30）**，只差 1 个 episode；Wilson 95% CI 大幅重叠（见 results.md §2.1）。H2 sparsity 汇总与 er.md 二次审计修正已回填 `results.md`。**当前 H3（100ep formal，accuracy confirmation）已启动。**

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| H0 correctness smoke | ✅ done | 2026-09-14 | S0/S1 SR=100%，Gate 全 PASS |
| H1 10ep pilot | ✅ done | 2026-09-14 | S0=90.0%、S1=80.0%（各 10 tasks × 1ep） |
| H1-Audit | ✅ done | 2026-09-14 | output/O 0.5%→39.9% 统计 bug 定位+修复 |
| H2 30ep convergence | ✅ done | 2026-09-15 | S0=90.0%、S1=86.7%（各 10 tasks × 3ep）；sparsity 汇总 + 审计修正完成 |
| H2 审计修正 | ✅ done | 2026-09-15 | common-scope 比较、PV A 语义、unit 口径、BOP 降级、相关性实算 |
| H3 100ep formal | 🔄 running | — | stats-on SR 正式确认，重点 task03/06/07 |

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
| 2026-09-15 | H2 审计修正 | er.md 二次审计 + `build_results_tables.py` v2 | ✅ | — | 加入 common-scope runtime/weight（去重）、Pearson/Spearman、Wilson CI；更正 PV A 语义、unit 口径、BOP 节 |
| 2026-09-15 | H3 100ep formal | `run_h3_100ep.sh`（S0/S1 各 10 tasks × 10ep） | 🔄 running | `h3_100ep/s0|s1/taskXX` | stats-on SR 正式确认；重点 task03/06/07 |

## 3. 后台任务

| PID | 阶段 | 启动时间 | 状态 | 备注 |
|---|---|---|---|---|
| 3542208 | H2 30ep | 2026-09-14 | ✅ finished | `run_h2_30ep.sh`，S0+S1 各 10 tasks × 3ep，20/20 完成，日志 `outputs/.../h2_run.log` |
| (见 §1) | H3 100ep | 2026-09-15 | 🔄 running | `run_h3_100ep.sh`，S0+S1 各 10 tasks × 10ep，日志 `outputs/.../h3_run.log` |

## 4. 异常与处理

- **H1 output/O 稀疏度误报 0.3-0.6%**：`quant_methods.py` 3 处 `out_normal_quant.to(torch.float32).mul_(M_q)` 为 in-place 操作，把 quantized code（`[-448,448]`）原地改写成 dequant 巨大值，后续收集 output_code 时值 `>448` round 到 E4M3 变 NaN，稀释 sparse_bit_rate。修复为 `mul`（非 in-place），forward 数值不变（SR 保持 100%）。见 commit `e4034aa`，回归测试 T11。
- **H2 首次启动失败（ModuleNotFoundError: No module named 'vla_tcs2'）**：`run_h2_30ep.sh` 用裸 `python`，在未激活 `smolvla_eval` 的 shell 里后台启动，导致找不到 `vla_tcs2` 包。修复为 `conda run -n smolvla_eval python`（与 G6 脚本一致），并同步修复 `run_h3_100ep.sh`。
- **`summarize_sparsity.py` 从仓库根无法运行（FileNotFoundError）**：`REPO_ROOT = HERE/../..` 只上溯到 `experiments/`，应为 `../../..`。已修复。
- **H0/H1 的「全口径」aggregate 不可与 H2 直接比较**：H0/H1 的 output/O 行为 pre-fix 污染值（≈0），使全口径 aggregate 降到 25-27%、unit sparsity 抬到 42-44%。已新增 `build_results_tables.py --exclude-roles output,O` 提供同口径对比（H1 vs H2 差 0.01-0.02pp）。
- **S0/S1 的 own-scope aggregate 混入了不同 workload（er.md 审计 P1）**：S0 `linear.include: ['*']` 含 VLM Linear，S1 只有 `expert.*`。修复：
  - runtime：新增 **common scope**（从 S0 剔除 VLM Linear）⇒ S0 42.40% vs S1 42.42%，**Δ = +0.02pp**（原文档写的 0.34pp 混入了 S0 独有的 VLM Linear，已作废）；
  - weight：新增 **Expert common scope**（并按 `module_id` 去重，原 raw 行数 = 10 × modules）⇒ S0 Expert FP8 42.58% vs S1 Expert W4 73.72%，**Δ = +31.15pp**（原「41.52% → 73.72%，+32.2pp」是 *(VLM FP8+Expert FP8)* vs *Expert W4*，scope 不一致，已作废）。
- **PV 的 A 矩阵语义写错（er.md 审计 P1）**：`model_wrapper.py` 中 `att_output = pv_matmul(probs, value_states...)`，因此 **PV 的 A = softmax 后的 attention probability `P`，不是 value activation**。原文档「RoPE/位置编码后 value 激活的固有结构」「不是量化产物」两点均错误，已更正；并明确统计口径是 post-quant code，小值 probability 可能被 FP8 scale 映射为 0。
- **unit sparsity 未做 native 修正（er.md 审计 P1）**：`collect_unit_sparsity_structured()` 直接对 masked normal tensor 计数，exporter 不与 `outlier_partition` join ⇒ 该指标是 **masked normal quant-path unit sparsity**，不是 native unit sparsity。已统一改名为 `quant-path unit sparsity (2×2, masked)` 并在文档中说明。
- **BOP proxy 不可加（er.md 审计 P1）**：`export_workload_csv` 把同一个 `S_bit` 同时用于 A/B 两侧、按 role 逐行生成（一个物理算子多行）、且用的是 reported 而非 native sparsity。已将该节降级为「legacy/debug BOP proxy — NOT physically additive」，不得作为硬件结论。
- **结论措辞过强（er.md 审计 P2）**：已修正三处——(a)「sparsity 与 SR 无相关」→ 改为报告实算的 pooled Pearson r=−0.057 / Spearman ρ=+0.050，并说明 n=20 且 SR 只有 4 个离散值，**不能证明独立**；(b)「task06/07 是 task 固有难度」→ 改为「与共享 task difficulty 或同 seed 采样效应一致，暂不能归因于量化配置」；(c) 「H1/H0 INVALID 标注解除」→ 澄清 pre-fix 原始 CSV **仍然 INVALID**，H2 只证明修复有效、替代测量可用。
- **SR 的统计显著性**：30ep 下 Wilson 95% CI 为 S0 [74.4%, 96.5%]、S1 [70.3%, 94.7%]，大幅重叠，gap 仅 1 个 episode ⇒ 已加入文档，正式 accuracy claim 交 H3 100ep。
- **H1 与 H2 共享 seed=1000**：H2 的 3 episodes 很可能包含 H1 那一集，因此措辞改为「从 1ep 扩到 3ep 后 sparsity estimate 几乎不变」，而非「独立实验重复验证」。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-14 | 创建文档，记录 H0/H1/H1-Audit 完成状态与 H2 启动 | |
| 2026-09-15 | 回填 H2 完成状态与结果（S0=90.0%、S1=86.7%） | |
| 2026-09-15 | 回填 H2 sparsity 汇总到 results.md（§14 全表 + §13 收敛 gate）；新增 `build_results_tables.py`，修复 `summarize_sparsity.py` 路径 bug | |
