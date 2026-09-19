# Phase H 结果记录（Results）

> 实验名称：2026-09-13_phaseH_accuracy-preserving-sparsity  
> 状态：done  
> 最后更新：2026-09-19

---

## 1. 摘要

**H3（100ep formal）已完成 —— 本实验全部闭环（2026-09-19）。** 正式准确率：**S0 FP8-all SR=89.0%（89/100）**、**S1 Expert-W4 SR=85.0%（85/100）**，gap = **4.0pp**。

> **SR 的置信区间**：100ep 下 Wilson 95% CI 为 S0 `[81.4%, 93.7%]`、S1 `[76.7%, 90.7%]`，**仍然重叠**（重叠区间 81.4–90.7%，重叠宽度约 9.3pp）。因此正式表述是：**S1（Expert-W4）相对 S0（全 FP8）的精度损失在 4pp 量级，但 100ep 仍不足以把该 gap 判为统计显著**。

**H2（30ep convergence）摘要（保留）**：S0 = 90.0%（27/30）、S1 = 86.7%（26/30），Wilson 95% CI `[74.4%, 96.5%]` / `[70.3%, 94.7%]` 大幅重叠 ⇒ 30ep 只能说「与 Phase G 的 88%/84% 相容」，不足以确定 gap 大小。

**H3 同时把 sparsity 收敛性推到 100ep**：H2(30ep) → H3(100ep) 的 §13 gate **全部 PASS**（最大 |ΔS_bit| = 0.05pp，远低于 0.5pp），weight static 逐位相同（42.58% / 73.72%）⇒ **sparsity estimate 在 10× 采样扩展下依旧不变**（§3.2）。

Sparsity 侧（native significand sparse-bit rate，4-bit 1MMM 口径，numerator/denominator 加总）：

| 指标 | S0 FP8-all | S1 Expert-W4 | 可比性 |
|---|---:|---:|---|
| native bit sparsity — **own scope** | 42.08% | 42.42% | ⚠️ scope 不同（S0 含 VLM Linear） |
| native bit sparsity — **common scope** | 42.40% | 42.42% | ✅ Δ = **+0.02pp** |
| quant-path unit sparsity (2×2, masked) | 7.36% | 7.99% | ⚠️ 未做 native 修正 |
| FP sidepath ratio | 1.23% | 1.27% | ✅ |
| weight sparse_bit_rate — **own scope** | 41.52% (all Linear) | 73.72% (expert only) | ❌ scope 不同，**不可作差** |
| weight sparse_bit_rate — **Expert common scope** | **42.58%** | **73.72%** | ✅ Δ = **+31.15pp** |

**关键结论**：

1. **§13 收敛 gate 成立**。所有非 output/O 的 aggregate 大类在 H1(10ep)→H2(30ep) 间 |ΔS_bit| ≤ 0.14pp、|Δunit| < 0.5pp，远低于 0.5pp gate ⇒ **从每 task 1ep 扩到 3ep 后 sparsity estimate 几乎不变**。
   > 注意 H1 与 H2 共用 `seed=1000`（作为 rollout `start_seed`），H2 的 3 episodes 很可能**包含 H1 的那一条**。因此这是「扩展采样下估计稳定」，**不是两组独立实验的重复验证**。
2. **H1-Audit 结论被 H2 复现**。H1-Audit（S0 goal task0 × 1ep）测得 output=39.90%、O=39.91%；H2 30ep aggregate 给出 output=39.89-39.90%、O=39.87-39.95% ⇒ 修复有效且经 multi-task 确认。**注意：H0/H1 的 pre-fix output/O 原始 CSV 仍然是 INVALID，只是其结论已被 post-fix 数据替代**（见 §3.6）。
3. **W4 的增益只存在于 weight 侧**。Expert common scope 下 weight sparse_bit_rate 由 **42.58% → 73.72%（+31.15pp）**；而 runtime（activation/output/MatMul）在 common scope 下 S0/S1 差 **0.02pp**。即 **W4 未改变运行时 FP8 code 的稀疏结构**。
4. **PV 的 A 矩阵是唯一结构性高稀疏的算子**：对应 **softmax 后的 attention probability `P`**（不是 value state），expert 60.99% / vlm 54.26%，element zero rate 亦高（31.53% / 18.42%）。其余 role 全部收敛到 39.9-41.0% 窄带。
5. **flow step 对绝大多数 GEMM operand 无影响**（<0.02pp），但 **PV A 呈倒 U 形**（S0 61.02→61.38→59.85%），说明 denoise 过程中 **attention distribution 的集中程度在变化**。这是 H2 最值得保留的新发现（§3.4）。
6. output/O 的 unit zero rate 是 ~3.5%，与 activation 一致；pre-fix 时期看到的 ~90% 是 bug 造成的伪值。

> **口径声明**：
> - native bit sparsity 是「4-bit significand（1MMM）」口径，**尚未做 exponent-alignment**（EffLoc ineffective-bit），不等于硬件对齐后的真实 bit sparsity —— 需 Phase I 实现。
> - **quant-path unit sparsity 未做 native (FP-protected) 修正**，衡量的是 masked normal quant datapath 内的 block-zero 机会。
> - `BOP_active_proxy` / `ideal_sparse_upper_bound` 是 **legacy/debug 代理量，非可加物理量，不能作为硬件结论**（§3.7）。



---

## 2. 总结果表

### 2.1 准确率（episode 加权，Goal suite，n_action_steps=10）

| Config | H0 smoke (1ep) | H1 (10 tasks × 1ep) | H2 (10 tasks × 3ep) | H3 (10 tasks × 10ep) | Wilson 95% CI (H3) |
|---|---:|---:|---:|---:|---|
| S0 FP8-all | 100.0% | 90.0% | 90.0% (27/30) | **89.0% (89/100)** | **[81.4%, 93.7%]** |
| S1 Expert-W4 | 100.0% | 80.0% | 86.7% (26/30) | **85.0% (85/100)** | **[76.7%, 90.7%]** |
| gap (S0−S1) | 0.0 | 10.0 pp | 3.3 pp (1 episode) | **4.0 pp (4 episodes)** | **区间仍重叠 ~9.3pp** |

**H3 逐 task 成功数（每 task 10 episodes）**

| Config | t00 | t01 | t02 | t03 | t04 | t05 | t06 | t07 | t08 | t09 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 FP8-all | 10 | 10 | 10 | 8 | 10 | 9 | 5 | 9 | 10 | 8 | **89** |
| S1 Expert-W4 | 10 | 10 | 8 | 5 | 10 | 10 | 6 | 9 | 10 | 7 | **85** |
| Δ (S0−S1) | 0 | 0 | +2 | **+3** | 0 | −1 | −1 | 0 | 0 | +1 | **+4** |

> **task03 被 H3 确认为 S1 独有弱项**（H2 中已初现：S0 100% vs S1 66.7%）：100ep 下 S0 = 8/10、S1 = 5/10，**Δ = −3 episodes，是全部 10 个 task 中最大的单个 gap**，也是唯一在 H2 与 H3 两次测量中都指向 S1 的 task。
> **task06 / task07 被 H3 排除为 config 效应**：H2 中两者在 S0/S1 同步掉分（33.3%/66.7%）；H3 下 task06 变成 S1 反而更高（6 vs 5）、task07 持平（9 vs 9）⇒ **与共享 task difficulty / 同 seed 采样效应一致，不能归因于 W4**。
> **task05 / task06 是 S0 略低**（−1），进一步削弱「S1 处处更差」的叙事。

> ⚠️ 不要把 H2 的 3.3pp 或 H3 的 4.0pp 当作已确立的精度损失：100ep 下两个 Wilson CI 仍重叠约 9.3pp。H3 的价值是把 gap 从「1 个 episode」提升到「4 个 episode」，并定位到 task03。

### 2.2 Sparsity（numerator/denominator 加总，native = 扣除 FP protected 后）

**Runtime（module_sparsity / unit_sparsity）**

| Config | Scope | Episodes | Native bit sparsity | Quant-path unit sparse (2×2, masked) | FP sidepath |
|---|---|---:|---:|---:|---:|
| S0 | **own**（含 VLM Linear） | 30 | 42.08% | 7.36% | 1.23% |
| S1 | **own**（VLM Linear 未量化） | 30 | 42.42% | 7.99% | 1.27% |
| S0 | **common**（drop VLM Linear） | 30 | **42.40%** | — | — |
| S1 | **common**（同集合） | 30 | **42.42%** | — | — |
| **Δ (common, H2)** | | | **+0.02 pp** | — | — |
| S0 | **own**（含 VLM Linear） | **100** | 42.08% | 7.35% | 1.23% |
| S1 | **own**（VLM Linear 未量化） | **100** | 42.41% | 7.98% | 1.27% |
| S0 | **common**（drop VLM Linear） | **100** | **42.40%** | — | — |
| S1 | **common**（同集合） | **100** | **42.41%** | — | — |
| **Δ (common, H3)** | | | **+0.018 pp** | — | — |

> **H2(30ep) 与 H3(100ep) 的 runtime aggregate 几乎逐位相同**（own: 42.08/42.08 与 42.42/42.41；common Δ: +0.02pp 与 +0.018pp；unit: 7.36/7.35 与 7.99/7.98；sidepath 完全相同）⇒ **采样从 30ep 扩到 100ep 不再改变任何 aggregate 数字**。

> 只有 **common scope** 的 Δ 才能解释为「配置变化对 runtime code sparsity 的影响」。原文档的「Δ=0.34pp」混入了 S0 独有的 VLM Linear，不能作为该结论。

**H0/H1 的口径与 pre-fix 说明**

| Config | Stage | 口径 | Native bit sparsity | Quant-path unit sparse | FP sidepath |
|---|---|---:|---:|---:|---:|
| S0 | H0 smoke | 全口径（**pre-fix**） | 25.59% | 43.68% | 1.23% |
| S1 | H0 smoke | 全口径（**pre-fix**） | 26.61% | 42.56% | 1.27% |
| S0 | H1 | 排除 pre-fix output/O | 43.60% | 9.72% | 1.43% |
| S1 | H1 | 排除 pre-fix output/O | 44.01% | 10.46% | 1.47% |
| S0 | H2 | 排除 output/O（与 H1 同口径） | 43.59% | 9.72% | 1.43% |
| S1 | H2 | 排除 output/O（与 H1 同口径） | 44.03% | 10.50% | 1.47% |

> H0/H1 的 output/O 行产生于 instrumentation 修复前（§3.6），其 CSV 值 ≈0，会把「全口径」aggregate 拖到 25-27%、unit sparsity 抬到 42-44%。**H0/H1 的 pre-fix output/O 原始测量仍然标 INVALID**，此处仅以「排除 output/O」的干净口径与 H2 比较。

**Weight static（episode 无关，已按 module_id 去重）**

| Config | Scope | Unique modules | Raw CSV rows | zero_rate | sparse_bit_rate |
|---|---|---:|---:|---:|---:|
| S0 | all Linear (VLM + Expert) | 224 | 2240 | 0.00% | 41.52% |
| S1 | expert only | 112 | 1120 | 8.68% | 73.72% |

> **H3(100ep) 的 weight static 与 H2 逐位相同**（S0 0.00% / 41.52%；S1 8.68% / 73.72%；Expert common scope 42.58% → 73.72%，Δ +31.15pp）—— weight 与 episode 无关，100ep 只是重复导出同一份 CSV（已被 `_dedupe_modules()` 按 `module_id` 去重）。

| **Expert common scope** | S0 FP8 | S1 W4 | Δ |
|---|---:|---:|---:|
| Modules | 112 | 112 | — |
| zero_rate | 0.00% | 8.68% | **+8.68 pp** |
| **sparse_bit_rate** | **42.58%** | **73.72%** | **+31.15 pp** |

> **论文中的 W4 增益必须引用 Expert common scope 行。** 原「41.52% → 73.72% (+32.2pp)」比较的是 *(VLM FP8 + Expert FP8)* vs *Expert W4*，scope 不同。
> 每个 task 各导出一份完全相同的 static weight CSV（raw 行数 = 10 × modules）；若未来要汇总 total_bits / 参数量 / weight BOP，**必须先按 module_id 去重**，否则放大 10×。

> 生成命令：
>
> ```bash
> conda run -n smolvla_eval python experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/build_results_tables.py \
>   --stage h2_30ep --compare h1_10ep                       # 完整表 + 收敛 gate
> conda run -n smolvla_eval python experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/build_results_tables.py \
>   --stage h2_30ep --compare h1_10ep --exclude-roles output,O   # H1/H2 同口径
> ```




---

## 3. 分组结果与分析

### 3.1 Correctness smoke

| Gate 项 | S0 FP8-all | S1 Expert-W4 | 判定 |
|---|---|---|---|
| static weight rows | 224 | 112 | ✅ PASS |
| prefill present | 320 行 | 96 行 | ✅ PASS |
| denoise steps | 0..9 齐全（每 step 320 行） | 0..9 齐全 | ✅ PASS |
| Linear roles | activation + output | activation + output | ✅ PASS |
| MatMul roles | A / B / O | A / B / O | ✅ PASS |
| attention kind | self / cross | self / cross | ✅ PASS |
| outlier accounting | sidepath 4753 行 | sidepath 4417 行 | ✅ PASS |
| native counter 非负 | 无负值 | 无负值 | ✅ PASS |
| SR | 100.0% | 100.0% | ✅ PASS |

weight 静态稀疏度（与 episode 无关，作参考）：

| Config | weight zero_rate | weight sparse_bit_rate |
|---|---:|---:|
| S0 FP8-all | 0.00% | 41.52% |
| S1 Expert-W4 | 8.68% | 73.72% |

判定：**PASS**（可进入 H1）。

> runtime 稀疏度（native significand sparse-bit rate，4-bit 1MMM 口径）：
>
> | component | phase | role | reported | native |
> |---|---|---|---:|---:|
> | vlm | prefill | activation | 41.10% | 40.53% |
> | vlm | prefill | A / B | 50.61% / 41.17% | 50.20% / 39.89% |
> | expert | denoise | activation | 41.00% | 40.43% |
> | expert | denoise | A / B | 54.55% / 41.18% | 54.11% / 39.78% |
> | expert | denoise | output / O | **INVALID (pre-fix)** | **INVALID (pre-fix)** |
>
> 说明：activation/A/B 的 native significand sparse_bit_rate ≈ 40-54%，不是 0%。**output/O 在原 H0/H1 中产生于 instrumentation bug 修复前（`mul_` in-place 污染了 output code），因此原始 CSV 的 output/O 字段已失效，标 INVALID**；修复后的可靠测量来自 H1-Audit（S0 Goal task0 × 1ep，output 39.90% / O 39.91%）与 **H2 post-fix 30ep aggregate（output 39.89-39.90% / O 39.87-39.95%）**，二者一致（见 §3.6）。这一指标是「4-bit significand（1MMM）」口径，尚未做 exponent-alignment（EffLoc 的 ineffective-bit 口径），因此不等于硬件对齐后真正的 bit sparsity——该口径需 Phase I 单独实现。W4 权重侧（weight_sparsity_static）S1≈73.7% 是另一条独立的稀疏来源。

### 3.2 收敛 gate（§13：|ΔS_bit| < 0.5pp）—— H1→H2→H3

#### 3.2.1 H1(10ep) → H2(30ep)

同口径（排除 pre-fix 的 output/O）下，H1(10ep) → H2(30ep) 的 aggregate 对比：

| Metric | H1 S0 | H2 S0 | Δ | H1 S1 | H2 S1 | Δ | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| native bit sparsity（全量化算子） | 43.60% | 43.59% | **−0.01 pp** | 44.01% | 44.03% | **+0.02 pp** | ✅ PASS |
| quant-path unit rate（2×2） | 9.72% | 9.72% | **0.00 pp** | 10.46% | 10.50% | **+0.04 pp** | ✅ PASS |
| FP sidepath | 1.43% | 1.43% | 0.00 pp | 1.47% | 1.47% | 0.00 pp | ✅ PASS |
| SR | 90.0% | 90.0% | 0.0 pp | 80.0% | 86.7% | +6.7 pp | — |

> **关于「独立复现」的措辞**：H1 与 H2 共用 `seed=1000`，evaluation 把它作为 rollout `start_seed` 传入。H2 只是把每 task 从 1 episode 扩到 3 episodes，**H2 很可能包含 H1 的那一条 episode**，而不是独立的一批 30 episodes。因此正确表述是：
>
> > 从每 task 1ep 扩展到 3ep 后，sparsity estimate 几乎不变。
>
> 而不是「两组独立实验重复证明了收敛」。这不影响结论，只影响措辞。

逐 component（H1 → H2 干净口径），全部 aggregate 大类：

| Config | Component | Phase | Role | H1 | H2 | Δ (pp) | Gate |
|---|---|---|---|---:|---:|---:|---|
| S0 | VLM | prefill | activation | 40.70% | 40.70% | +0.00 | ✅ PASS |
| S0 | VLM.PV | prefill | A | 54.26% | 54.26% | +0.00 | ✅ PASS |
| S0 | VLM.PV | prefill | B | 39.87% | 39.88% | +0.01 | ✅ PASS |
| S0 | VLM.QK | prefill | A | 40.46% | 40.47% | +0.00 | ✅ PASS |
| S0 | VLM.QK | prefill | B | 40.43% | 40.44% | +0.00 | ✅ PASS |
| S0 | EXPERT | denoise | activation | 40.58% | 40.58% | −0.00 | ✅ PASS |
| S0 | EXPERT.PV | denoise | A | 61.02% | 60.99% | −0.04 | ✅ PASS |
| S0 | EXPERT.PV | denoise | B | 39.95% | 39.96% | +0.00 | ✅ PASS |
| S0 | EXPERT.QK | denoise | A | 40.45% | 40.45% | −0.00 | ✅ PASS |
| S0 | EXPERT.QK | denoise | B | 40.21% | 40.20% | −0.00 | ✅ PASS |
| S1 | VLM.PV | prefill | A | 54.25% | 54.26% | +0.01 | ✅ PASS |
| S1 | VLM.PV | prefill | B | 40.57% | 40.58% | +0.00 | ✅ PASS |
| S1 | VLM.QK | prefill | A | 40.63% | 40.64% | +0.01 | ✅ PASS |
| S1 | VLM.QK | prefill | B | 40.59% | 40.59% | +0.00 | ✅ PASS |
| S1 | EXPERT | denoise | activation | 40.65% | 40.65% | −0.00 | ✅ PASS |
| S1 | EXPERT.PV | denoise | A | 60.30% | 60.43% | +0.14 | ✅ PASS |
| S1 | EXPERT.PV | denoise | B | 40.24% | 40.24% | −0.00 | ✅ PASS |
| S1 | EXPERT.QK | denoise | A | 40.46% | 40.45% | −0.00 | ✅ PASS |
| S1 | EXPERT.QK | denoise | B | 40.29% | 40.28% | −0.01 | ✅ PASS |

**结论：gate 全部 PASS**（最大 |Δ| = 0.14pp，S1 EXPERT.PV A），unit rate 同样全部 < 0.5pp。⇒ **从 1ep/task 扩到 3ep/task 后 sparsity 估计无变化**；H3 的 100ep 主要用于 stats-on SR 复验，而非 sparsity 收敛。

#### 3.2.2 H2(30ep) → H3(100ep)（采样再扩 3.3×）

生成命令：`build_results_tables.py --stage h3_100ep --compare h2_30ep`（全量输出见 `/tmp/h3_clean.md` 同构内容，可复现）。

| Metric | H2 S0 | H3 S0 | Δ | H2 S1 | H3 S1 | Δ | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| native bit sparsity（own scope） | 42.08% | **42.08%** | **0.00 pp** | 42.42% | **42.41%** | **−0.01 pp** | ✅ PASS |
| native bit sparsity（common scope） | 42.40% | **42.40%** | **0.00 pp** | 42.42% | **42.41%** | **−0.01 pp** | ✅ PASS |
| quant-path unit rate（2×2） | 7.36% | **7.35%** | **−0.01 pp** | 7.99% | **7.98%** | **−0.01 pp** | ✅ PASS |
| FP sidepath | 1.23% | **1.23%** | 0.00 pp | 1.27% | **1.27%** | 0.00 pp | ✅ PASS |
| Expert weight sparse_bit_rate | 42.58% | **42.58%** | 0.00 pp | 73.72% | **73.72%** | 0.00 pp | ✅ PASS |
| SR | 90.0% | 89.0% | −1.0 pp | 86.7% | 85.0% | −1.7 pp | — |

逐 component × role 的 19 项 aggregate 大类（非 output/O）：**全部 |Δ| ≤ 0.05pp**（最大为 S0 EXPERT.PV A 的 −0.05pp）。

**结论：gate 全部 PASS，且 Δ 比 H1→H2 更小**（0.05pp vs 0.14pp）⇒

> **sparsity estimate 在 1ep → 3ep → 10ep（每 task）三级采样扩展下都保持不动。** H1/H2 的收敛结论被 100ep 完整确认，且这次包含 10× 更多 episode，已不存在「H2 包含 H1 那条 episode」这类质疑。

**流程（H3 的实际执行配置）**：H3 全程 `--skip-calibration`（复用 Phase G 的 scale）+ 保留 module sparsity counters + flow-step tagging，关闭了 `fp_code_audit`（§5 的决定），因此 H3 既是 accuracy confirmation，也是 sparsity 的最终 sanity check。

> output/O 单独看。下表的 H1 列是 **pre-fix INVALID 原始测量**，仅作对照，**不参与 gate**：

| Role | H1（pre-fix，INVALID） | H1-Audit（S0 task0，post-fix） | H2 S0（30ep） | H2 S1（30ep） |
|---|---:|---:|---:|---:|
| output (linear) | 0.40% | **39.90%** | **39.89%** | **39.90%** |
| O (QK/PV) | 0.02-2.21% | **39.91%** | **39.87-39.95%** | **39.87-39.94%** |

**H1-Audit 的单 task post-fix 测量（39.90%/39.91%）被 H2 的 30ep aggregate（39.87-39.95%）精确复现**（差 <0.05pp）⇒ §3.6 的修复在 multi-task 层面得到确认。**注意：这确认的是「修复有效、替代测量可用」，H0/H1 的 pre-fix output/O 原始 CSV 仍然 INVALID。**

### 3.3 Component / operator

**Component 级（H2，post-fix 全口径，30ep aggregate）**

| Config | Component | Phase | Role | Zero native | Bit sparse native | Quant-path unit sparse (2×2) | FP sidepath |
|---|---|---|---|---:|---:|---:|---:|
| S0 | VLM | prefill | activation | 0.04% | 40.70% | 3.84% | 0.95% |
| S0 | VLM | prefill | output | 0.00% | 39.89% | 3.40% | 0.96% |
| S0 | VLM.QK | prefill | A / B / O | 0.00% | 40.47% / 40.44% / 39.95% | 3.86% / 8.06% / 5.59% | 1.56% / 2.47% / 0.56% |
| S0 | VLM.PV | prefill | A / B / O | 18.42% | **54.26%** / 39.88% / 39.89% | 32.37% / 4.05% / 3.51% | 0.56% / 1.81% / 1.56% |
| S0 | EXPERT | denoise | activation | 0.04% | 40.58% | 3.79% | 0.96% |
| S0 | EXPERT | denoise | output | 0.00% | 39.90% | 3.40% | 0.96% |
| S0 | EXPERT.QK | denoise | A / B / O | 0.00% | 40.45% / 40.20% / 39.91% | 3.86% / 7.72% / 5.46% | 1.56% / 2.66% / 0.74% |
| S0 | EXPERT.PV | denoise | A / B / O | **31.53%** | **60.99%** / 39.96% / 39.87% | 34.94% / 4.25% / 3.50% | 0.74% / 1.97% / 1.56% |
| S1 | VLM.PV | prefill | A / B / O | 18.42% | 54.26% / 40.58% / 39.90% | 32.35% / 4.41% / 3.51% | 0.56% / 1.57% / 1.56% |
| S1 | VLM.QK | prefill | A / B / O | 0.00% | 40.64% / 40.59% / 39.94% | 3.92% / 8.11% / 5.61% | 1.56% / 2.47% / 0.56% |
| S1 | EXPERT | denoise | activation | 0.04% | 40.65% | 3.82% | 0.96% |
| S1 | EXPERT | denoise | output | 0.00% | 39.90% | 3.40% | 0.96% |
| S1 | EXPERT.QK | denoise | A / B / O | 0.00% | 40.45% / 40.28% / 39.90% | 3.86% / 7.74% / 5.46% | 1.56% / 2.65% / 0.74% |
| S1 | EXPERT.PV | denoise | A / B / O | 30.75% | 60.43% / 40.24% / 39.87% | 34.19% / 4.41% / 3.50% | 0.74% / 1.86% / 1.56% |

> S1 无 VLM Linear 行（`linear.include: expert.*`，VLM 保持 raw FP，与 G1-D 一致）；VLM 的 QK/PV MatMul 仍被量化（`quantize_matmul: true`）。

**PV 的 A 是什么（重要更正）**

`model_wrapper.py` 的 attention 实现为：

```python
probs = nn.functional.softmax(masked_att_weights, dim=-1)   # 已加 causal mask
att_output = pv_matmul(probs, value_states.permute(0, 2, 1, 3))
```

因此：

$$PV:\quad A = P = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + \text{mask}\right),\qquad B = V$$

即 **PV 的 A 矩阵对应 softmax 之后的 attention probability `P`**，**不是 value activation**。

正确解释：

> PV A 的高稀疏与 attention probability 的**长尾/集中分布**、**attention mask** 以及 **FP8 quantization** 共同相关。

**不要写「这是 RoPE/位置编码后 value 激活的固有结构」，也不要写「不是量化产物」** —— 这里统计的是 **post-quant code**，小的非零 probability 很可能被 FP8 scale 映射为 0。要区分「softmax 原生的小值结构」与「量化引入的零」，需要额外收集 **pre-quant `probs`**。

**要点**

- **PV 的 A（= attention probability）是唯一的高稀疏算子**：≈61%（EXPERT.PV）/ 54%（VLM.PV），element zero rate 也显著（31.5% / 18.4%），unit zero rate 34-35%。
- 其余所有 role **收敛到 39.9-40.7%** 的窄带，说明该窄带由 E4M3 的 significand 位模式（1MMM 口径下 FP8 值本身的可压缩性）主导，而非算子/层位置。
- **S0 与 S1 在同 role 上差异 ≤ 0.4pp**：W4 只改 weight 的存储宽度，不改运行时 activation/output 的 FP8 code。
- output/O 的 quant-path unit rate ≈ 3.4-5.6%，**不是** pre-fix 时期的 ~90% —— 后者是 (code × scale) 巨大值破坏 2×2 unit 对齐造成的伪高稀疏。

**Operator 级（linear，跨层聚合，bit sparse native / unit sparse / FP sidepath）**

| Component | Operator | Role | S0 | S1 |
|---|---|---|---|---|
| VLM | q/k/v/gate/up_proj | activation | 40.67-40.69% | — |
| VLM | q/k/v/gate/up_proj | output | 39.87-39.94% | — |
| VLM | o_proj | activation / output | 39.90% / 39.90% | — |
| VLM | down_proj | activation / output | 41.04% / 39.89% | — |
| EXPERT | q/gate/up_proj | activation | 40.66% | 40.66% |
| EXPERT | k_proj | activation | 40.52% | 40.60% |
| EXPERT | v_proj | activation | 40.20% | 40.64% |
| EXPERT | o_proj | activation / output | 39.88% / 39.89% | 39.88% / 39.89% |
| EXPERT | down_proj | activation / output | 41.02% / 39.89% | 41.03% / 39.89% |
| EXPERT | q/k/v/gate/up/down | output | 39.89-39.92% | 39.89-39.92% |

> `down_proj` 的 activation 略高（≈41.0%），但仅比其余算子高 ~0.35pp，无结构性差异。完整表（含 unit / FP sidepath 列）由 `build_results_tables.py` 输出。

### 3.4 Flow step

Expert denoise，native bit sparse rate（%），按 flow_step 0..9：

| step | S0 QK A | S0 QK B | S0 QK O | S0 PV A | S0 PV B | S0 PV O |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40.45 | 40.20 | 39.91 | 61.02 | 39.95 | 39.87 |
| 1 | 40.45 | 40.20 | 39.91 | 61.26 | 39.95 | 39.87 |
| 2 | 40.45 | 40.20 | 39.91 | 61.35 | 39.95 | 39.87 |
| 3 | 40.45 | 40.20 | 39.91 | 61.38 | 39.95 | 39.87 |
| 4 | 40.45 | 40.21 | 39.91 | 61.36 | 39.95 | 39.87 |
| 5 | 40.46 | 40.21 | 39.90 | 61.28 | 39.95 | 39.87 |
| 6 | 40.46 | 40.21 | 39.90 | 61.09 | 39.96 | 39.87 |
| 7 | 40.46 | 40.21 | 39.90 | 60.83 | 39.96 | 39.87 |
| 8 | 40.46 | 40.21 | 39.90 | 60.42 | 39.96 | 39.87 |
| 9 | 40.45 | 40.20 | 39.90 | 59.85 | 39.96 | 39.88 |

| step | S1 QK A | S1 QK B | S1 QK O | S1 PV A | S1 PV B | S1 PV O |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40.46 | 40.28 | 39.91 | 60.29 | 40.24 | 39.87 |
| 1 | 40.45 | 40.28 | 39.91 | 60.56 | 40.23 | 39.86 |
| 2 | 40.45 | 40.28 | 39.91 | 60.70 | 40.23 | 39.87 |
| 3 | 40.45 | 40.28 | 39.91 | 60.79 | 40.24 | 39.87 |
| 4 | 40.45 | 40.28 | 39.90 | 60.81 | 40.24 | 39.87 |
| 5 | 40.46 | 40.28 | 39.89 | 60.77 | 40.24 | 39.87 |
| 6 | 40.46 | 40.28 | 39.90 | 60.62 | 40.24 | 39.87 |
| 7 | 40.46 | 40.28 | 39.90 | 60.40 | 40.24 | 39.87 |
| 8 | 40.45 | 40.28 | 39.89 | 59.99 | 40.24 | 39.87 |
| 9 | 40.45 | 40.28 | 39.89 | 59.41 | 40.24 | 39.87 |

**要点**

- QK A/B/O 与 PV B/O 在全部 10 个 flow step 上**几乎完全平坦**（变化 < 0.02pp），说明 diffusion 去噪的时间维不改变这些 operand 的 significand 位模式统计。
- **唯一有趋势的是 PV A（= attention probability `P`）**：呈倒 U 形，step0 → step3-4 上升（S0: 61.02%→61.38%），随后单调下降至 step9（59.85%）；S1 同形状但幅度略小（60.29%→60.81%→59.41%）。峰谷差 ≈1.4-1.5pp。
- 既然 PV A 是 `P`，这个趋势的正确解读是：

  > **Action Expert 在 flow matching 不同 denoise step 上，attention distribution 的集中程度发生了变化。**

  这比「activation bits 随 step 改变」准确得多。
- 该趋势不影响 headline 数字（PV A 在 denoise 上的加权平均 ≈60.99% / 60.43%），但**可能形成一个独立的小结论**：

  > flow step 对大多数 GEMM operand sparsity 几乎无影响，但会改变 attention-probability sparsity。

**Phase I / appendix 建议补测**（用于确认上面的解释）：

```text
H(P) = -Σ_j P_j log P_j             attention entropy
fraction(P < FP8 quantization threshold)
post-quant zero ratio
top-1 / top-k attention mass
```

看它们是否与 PV A 的 61→59% 趋势同步。若同步，则该结论可直接由 entropy/mass 支撑而不仅是 bit 统计。

### 3.5 Task-wise correlation

H2，per-task native bit sparsity 与 SR：

| Config | Task | SR | Native bit sparsity | FP sidepath |
|---|---|---:|---:|---:|
| S0 | task00 | 100.0% | 42.04% | 1.23% |
| S0 | task01 | 100.0% | 42.11% | 1.23% |
| S0 | task02 | 100.0% | 42.11% | 1.23% |
| S0 | task03 | 100.0% | 42.05% | 1.23% |
| S0 | task04 | 100.0% | 42.13% | 1.23% |
| S0 | task05 | 100.0% | 42.00% | 1.23% |
| S0 | **task06** | **33.3%** | 42.10% | 1.23% |
| S0 | **task07** | **66.7%** | 42.06% | 1.23% |
| S0 | task08 | 100.0% | 42.15% | 1.23% |
| S0 | task09 | 100.0% | 42.12% | 1.23% |
| S1 | task00 | 100.0% | 42.38% | 1.27% |
| S1 | task01 | 100.0% | 42.46% | 1.26% |
| S1 | task02 | 100.0% | 42.45% | 1.26% |
| S1 | **task03** | **66.7%** | 42.35% | 1.26% |
| S1 | task04 | 100.0% | 42.48% | 1.26% |
| S1 | task05 | 100.0% | 42.33% | 1.26% |
| S1 | **task06** | **33.3%** | 42.44% | 1.27% |
| S1 | **task07** | **66.7%** | 42.42% | 1.27% |
| S1 | task08 | 100.0% | 42.47% | 1.26% |
| S1 | task09 | 100.0% | 42.46% | 1.27% |

**相关系数（实际计算，非定性描述）**

| Scope | n | Pearson r | Spearman ρ（tie-averaged） |
|---|---:|---:|---:|
| S0 | 10 | −0.0085 | +0.1643 |
| S1 | 10 | +0.1169 | +0.3820 |
| **pooled** | 20 | **−0.0574** | **+0.0496** |

**要点**

- per-task bit sparsity 全部落在 **42.00-42.48%** 窄带（跨 task 极差 0.48pp），而 SR 从 33.3% 到 100% 变化。
- **但结论必须弱化**：该统计只有 20 个数据点，且每 task 仅 3 episodes ⇒ SR 只能取 `{0, 33.3, 66.7, 100}%` 这 4 个值（**大量并列**），相关系数分辨率极低。因此：

  > H2 pilot 中**未观察到明显**的 per-task sparsity–SR association（pooled Pearson r ≈ −0.06，Spearman ρ ≈ +0.05）。由于每 task 仅 3 episodes，**不能据此证明两者统计独立**。

- **task06/task07 在两个 config 上同步掉分**（33.3%/66.7%）：

  > 与**共享的 task difficulty 或相同 seed 下的 episode sampling effect** 一致，**暂不能归因于量化配置**。

- **只有 task03 是 S1 独有掉分**（66.7% vs S0 100%）⇒ 这是唯一可能是 W4 精度损失的信号，需 H3 100ep 复验。

**H3（100ep）复验**——每 task 10 episodes，SR 取值分辨率提到 10%（{0,10,...,100}%）：

| Config | Task | SR | 成功/10 | Native bit sparsity | FP sidepath |
|---|---|---:|---:|---:|---:|
| S0 | task00 | 100.0% | 10 | 42.04% | 1.24% |
| S0 | task01 | 100.0% | 10 | 42.12% | 1.23% |
| S0 | task02 | 100.0% | 10 | 42.11% | 1.23% |
| S0 | **task03** | **80.0%** | **8** | 42.06% | 1.23% |
| S0 | task04 | 100.0% | 10 | 42.12% | 1.23% |
| S0 | task05 | 90.0% | 9 | 41.99% | 1.23% |
| S0 | **task06** | **50.0%** | **5** | 42.08% | 1.23% |
| S0 | task07 | 90.0% | 9 | 42.09% | 1.23% |
| S0 | task08 | 100.0% | 10 | 42.14% | 1.23% |
| S0 | task09 | 80.0% | 8 | 42.10% | 1.23% |
| S1 | task00 | 100.0% | 10 | 42.38% | 1.27% |
| S1 | task01 | 100.0% | 10 | 42.46% | 1.26% |
| S1 | **task02** | **80.0%** | **8** | 42.45% | 1.26% |
| S1 | **task03** | **50.0%** | **5** | 42.36% | 1.26% |
| S1 | task04 | 100.0% | 10 | 42.47% | 1.26% |
| S1 | task05 | 100.0% | 10 | 42.32% | 1.26% |
| S1 | **task06** | **60.0%** | **6** | 42.44% | 1.27% |
| S1 | task07 | 90.0% | 9 | 42.43% | 1.26% |
| S1 | task08 | 100.0% | 10 | 42.47% | 1.26% |
| S1 | task09 | **70.0%** | **7** | 42.44% | 1.27% |

**相关系数（H3，100ep，实际计算）**

| Scope | n | Pearson r | Spearman ρ（tie-averaged） |
|---|---:|---:|---:|
| S0 | 10 | +0.1698 | +0.5078 |
| S1 | 10 | +0.1546 | +0.3039 |
| **pooled** | 20 | **−0.0725** | **+0.1250** |

**H3 要点**

- **per-task bit sparsity 依旧落在窄带**（S0 41.99-42.14%，跨 task 极差 0.15pp；S1 42.32-42.47%，极差 0.15pp），而 SR 从 50% 到 100% 变化 —— 与 H2 结论一致。
- **pooled Pearson r ≈ −0.07、Spearman ρ ≈ +0.13（n = 20）**。相关系数仍指向「无线性/单调关系」，但这里必须注意：**n 只有 20，且 SR 只能取 11 个离散值（大量并列）**，分辨率依旧很低。因此只能说：

  > **H3（100ep）仍然没有观察到 per-task sparsity–SR 的明显 association，但仍不能据此证明两者统计独立。**

  （注：`build_results_tables.py` 输出里的「10 tasks × 3 episodes = 30 episodes」注释是 H2 阶段写死的文案；H3 实际上是 10 tasks × 10ep = 100 episodes。此处已按 H3 实际采样量重述。）
- **task03 被确认是 S1 独有弱项**：H2 中已初现（S0 100% vs S1 66.7%），H3 下 S0 = 8/10、S1 = 5/10（Δ = −3 episodes），是 10 个 task 中最大的单个 gap，也是唯一在两次测量中都与 S1 同向的 task。
- **task06 / task07 被排除为 config 效应**：H2 下两者在 S0/S1 同步掉分（33.3% / 66.7%），H3 下 task06 反而 S1 更高（6 vs 5）、task07 持平（9 vs 9）⇒ **与共享 task difficulty 或同 seed 的 episode sampling effect 一致，不能归因于 W4**。
- **task05 / task06 是 S0 略低**（−1 episode），说明「S1 处处更差」的叙事不成立。

### 3.6 H1-Audit：output/O 稀疏度统计 bug 的独立验证与修复（已闭环）

早期 H1 报告 output/O 的 significand sparse_bit_rate ≈ 0.3-0.6%，与 activation/A/B 的 40-56% 显著不一致，触发独立审计（er.md H1-Audit）。

**方法**：在 `collect_quant_tensor()` 旁路新增独立 E4M3 raw-code 审计 `_collect_fp_code_audit()`，不调用现有 `_extract_sm_from_raw()`、不改 forward 数值、不用 RNG，直接对真正送进 collector 的 post-quant FP8 code 做 bit-pattern histogram（zero/subnormal/nan/saturation/significand 零位率）。

**结论**：er.md §12「情况 2」成立——主 bit accounting 存在 bug，但根因不在 `stat_manager.py`，而在 `quant_methods.py` forward 路径：

- 3 处 `out_normal_quant.to(torch.float32).mul_(M_q)` 是 **in-place** 操作，把 quantized code（`[-448, 448]`）原地改写成 dequant 巨大值（`code × M_q`）。
- 后续 `output_code=out_normal_quant` 收集到的是巨大值；round 到 E4M3 时 PyTorch 对 `>448` 的值返回 **NaN 而非饱和**，98% 元素变 NaN，稀释 `sparse_bit_rate` 到 ~0.5%。
- 该 bug 不影响 forward 数值（`div(..., 2^16)` 为 out-of-place，dequant 正确），故 SR 一直是 100%。

**修复**：3 处 `mul_` → `mul`（非 in-place），不改变 forward 数值。

| role | 修复前 native | 修复后 native | 独立 audit sig-nz |
|---|---:|---:|---:|
| output | 0.47% | **39.90%** | 39.90% |
| O | 0.36% | **39.91%** | 39.91% |

**验证**：SR 保持 100%；新增回归测试 T11（`test_t11_output_code_not_mutated_by_dequant`）；全部 19 个测试通过。修复后 output/O 与 activation/A/B 的 sparsity 量级一致（≈40%），符合 FP8 激活主导稀疏度的预期。

> **测量口径**：修复后的 39.90%/39.91% 是 **H1-Audit S0 Goal task0 × 1ep** 的测量值（不是原 H1 10ep 或 H0 的 aggregate——那些是 pre-fix 污染的）。

**H2 复现（已闭环）**：H2（10 tasks × 3ep，post-fix collector）给出的 aggregate 为 output = **39.89%（S0）/ 39.90%（S1）**、O = **39.87-39.95%（S0）/ 39.87-39.94%（S1）**，与 H1-Audit 的单 task 值（39.90%/39.91%）在 **0.05pp 内**一致。

> **措辞澄清（重要）**：H1/H0 的 pre-fix output/O **原始测量仍然是 INVALID，没有被"解除"**。H2 证明的是：
>
> > 该异常已通过 H1-Audit 与 H2 post-fix aggregate 完成闭环；正式结果由 post-fix 数据替代。
>
> 旧 CSV 里的 0.3-0.6% 永远无效，引用时应使用 post-fix 数值。

### 3.7 Weight static 与 workload 代理

**Weight static sparsity（episode 无关，已按 `module_id` 去重）**

| Config | Scope | Unique modules | Raw CSV rows | zero_rate | sparse_bit_rate |
|---|---|---:|---:|---:|---:|
| S0 FP8-all | **all Linear (VLM + Expert)** | 224 | 2240 | 0.00% | 41.52% |
| S1 Expert-W4 | **expert only** | 112 | 1120 | 8.68% | 73.72% |

以上两行 **scope 不同，不可作差**（S0 含 VLM FP8 Linear，S1 的 VLM Linear 根本没有进入 QuantizedLinear / static CSV，这也是 224 vs 112 的来源）。

**Expert common scope（唯一可用于 W4 增益论断的比较）**

| Metric | S0 Expert FP8 | S1 Expert W4 | Δ |
|---|---:|---:|---:|
| Modules | 112 | 112 | — |
| zero_rate | 0.00% | 8.68% | **+8.68 pp** |
| **sparse_bit_rate** | **42.58%** | **73.72%** | **+31.15 pp** |

**要点**

- **W4 是唯一显著提升可压缩性的手段**：Expert 同 scope 下 weight sparse_bit_rate **+31.15pp**（42.58% → 73.72%），element zero_rate 从 0% 升到 8.68%。
- 但这是 **weight 侧**收益；runtime activation/output 仍受 FP8 主导（§3.3），common scope 下 S0/S1 runtime 只差 **0.02pp**。
- **去重必要**：每个 task 各导出一份完全相同的 static weight CSV（raw 行数 = 10 × modules）。比例不受影响（$\frac{10N_{sparse}}{10N_{total}} = \frac{N_{sparse}}{N_{total}}$），但 `total_bits` / 参数量 / weight BOP / memory volume 若直接相加会**放大 10×**。

**Legacy/debug BOP proxy — 不能用于硬件结论**

| Config | Rows | BOP dense proxy | BOP active proxy | active/dense | Calls |
|---|---:|---:|---:|---:|---:|
| S0 FP8-all | 35200 | 2.3928e14 | 7.9557e13 | 0.3325 | 1,474,880 |
| S1 Expert-W4 | 32960 | 1.4839e14 | 4.7602e13 | 0.3208 | 1,430,464 |

> ⚠️ **不要把 66.8% / 67.9% 当作论文结果或 hardware reduction headline。** `export_workload_csv` 有三处已知缺陷：
>
> 1. **Linear 应分别缩放 A 与 W**（`MAC·b_A(1−S_A)·b_W(1−S_W)`），但 exporter 把**同一个 `S` 同时用于两侧**；
> 2. **MatMul 应按 `S_A`、`S_B` 分别缩放，且一个物理算子只应有一行**。当前按 role 逐行生成：A 行用 `S_A` 缩两边，B 行又生成一次完整 operator MAC ⇒ **不可加的物理量**；
> 3. 这里取的是 `entry["sparse_bit_rate"]`（**reported**），**不是** headline 使用的 **native corrected** sparsity。
>
> `ideal_sparse_upper_bound = 1/(1−S_bit)` 同样只是 bit-level 上界，不是实测加速比。
>
> **Phase I 应重写为 one physical operator, one row**：`(operator, phase, step) → (S_A, S_W/B, S_O, M, K, N, R_FP)`。
>
> whole-model MAC coverage 因 raw-op runtime denominator 未 instrument，按 experiment_setup §14.1 记 **N/A**（§16 禁止在无 denominator 时声称 whole-model sparsity）。

**Quant-path unit sparsity 的语义（重要）**

bit sparsity 做了 native 修正：

$$\text{native} = \text{reported} - \text{protected (FP outlier)}$$

但 **unit sparsity 没有做这个修正**。代码直接把进入 normal quant path 的 tensor 送给 `collect_unit_sparsity_structured()`，计算 2×2 unit zero；outlier forward 中 protected 位置已在 normal tensor 里被 mask 为 0；CSV exporter 也不与 `outlier_partition` 做任何 join：

$$\text{unit\_zero\_rate} = \frac{N_{zero\text{-}unit}}{N_{total\text{-}unit}}$$

因此 H2 的 S0 unit = 7.36%、S1 = 7.99% 的准确含义是：

$$\boxed{\text{masked normal quant-path 2×2 unit sparsity}}$$

**不是** native unit sparsity after excluding protected elements。本表中该列一律命名为 `quant-path unit sparsity (2×2, masked)`，不要与 bit sparsity 的 native 口径混读。从硬件角度这个指标仍然有价值——它描述实际 normal quant datapath 中的 block-zero opportunity。

---

## 4. 结论

1. **准确率（正式，H3 100ep）**：S0 FP8-all SR=**89.0%（89/100）**、S1 Expert-W4 SR=**85.0%（85/100）**，gap = **4.0pp（4 个 episode）**。Wilson 95% CI `[81.4%, 93.7%]` / `[76.7%, 90.7%]`，**仍重叠 ~9.3pp** ⇒ 结论为「**Expert-W4 的精度损失在 4pp 量级，但 100ep 未将 gap 推到统计显著**」。相对 H2（90.0%/86.7%，3.3pp）两个配置各降 ~1 个 episode，指向同一 baseline。$1/(1-S)$ 仍不能作为实际加速比（§3.7）。
2. **收敛性**：§13 gate 在**同口径**下三级全部 PASS——H1(10ep)→H2(30ep) 最大 |ΔS_bit| = 0.14pp；**H2(30ep)→H3(100ep) 最大 |ΔS_bit| = 0.05pp**；|Δunit| 均 < 0.5pp。runtime aggregate 在 30ep 与 100ep 间几乎逐位相同（own 42.08%/42.08% 与 42.42%/42.41%；common Δ +0.02/+0.018pp），weight static 完全相同（42.58%/73.72%）⇒ **sparsity estimate 在 1ep → 3ep → 10ep 扩展下均不变**，已不再有「H2 包含 H1 episode」这类质疑。
3. **稀疏来源（scope 修正后）**：
   - **weight 侧**：Expert common scope 下 W4 把 weight sparse_bit_rate 从 **42.58% 提升到 73.72%（+31.15pp）**，zero_rate 0% → 8.68%。**这是 W4 的唯一增益，也是论文中应引用的数字。**
   - **runtime 侧**：common scope 下 S0/S1 只差 **+0.02pp**（42.40% vs 42.42%）⇒ **W4 不改变运行时 FP8 code 的稀疏结构**（activation/output/MatMul 由 FP8 主导）。
   - 原文档的「+32.2pp」与「Δ=0.34pp」均因 scope 不一致而失效。
4. **唯一结构性的高稀疏算子**：**PV 的 A = softmax 后的 attention probability `P`**（EXPERT 60.99%、VLM 54.26%），element zero rate 亦高（31.53%/18.42%）。其余 role 全部收敛到 39.9-41.0% 窄带。
5. **Flow step 对 operand sparsity 基本无影响**（QK A/B/O、PV B/O 在 step 0-9 平坦 <0.02pp），但 **PV A（= `P`）呈倒 U 形**（S0 61.02→61.38→59.85%）⇒ **denoise 过程中 attention distribution 的集中程度在变化**。Phase I 应补测 attention entropy / top-k mass / pre-quant zero ratio 来支撑该解释。
6. **Sparsity 与 task 的 association（弱化后）**：per-task bit sparsity 落在 S0 41.99-42.14% / S1 42.32-42.47% 窄带（跨 task 极差仅 0.15pp），而 SR 从 50% 到 100% 变化。H3(100ep) pooled Pearson r ≈ −0.07、Spearman ρ ≈ +0.13（n = 20）⇒ **仍只能说「未观察到明显 association」，不能证明统计独立**（n 小且 SR 离散）。task03 是 **H2 与 H3 两次测量中唯一都指向 S1 的 task**（H2: 100% vs 66.7%；H3: 8/10 vs 5/10，是最大单项 gap）⇒ **它是 W4 精度损失的主要候选信号**；而 task06/task07 被 H3 排除（task06 反而 S1 更高、task07 持平），归为共享 task difficulty。
7. **统计链路正确性已闭环**：output/O 的 ≈0.5% 为 instrumentation bug（§3.6），修复后 H1-Audit 单 task 值被 H2 30ep aggregate 在 0.05pp 内复现。**注意 pre-fix 的 H0/H1 原始 CSV 仍然 INVALID，只是结论已被 post-fix 数据替代。**

## 5. 问题与后续

- **H3（100ep Goal formal）= ✅ 已完成（2026-09-19）—— accuracy confirmation + sparsity 最终 sanity check 双重完成**。S0 = 89.0%（89/100，`[10,10,10,8,10,9,5,9,10,8]`）、S1 = 85.0%（85/100，`[10,10,8,5,10,10,6,9,10,7]`），gap = 4.0pp。按 §5 的决定执行：保留 module sparsity counters / flow-step tagging / eval SR，**关闭 `fp_code_audit`**（已在 §3.6 闭环）。结果见 §2.1 / §3.2.2 / §3.5。
  - **task03 需后续关注**：它是 H2与 H3 两次测量中唯一都指向 S1 的 task（H3 下 −3 episodes）。若要做 second-order 归因，它是首选单 task 案例。
- **G6（Expert-FP8 背景 + VLM selective precision）= ✅ 已完成**（`../2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/`）：四组 SR = 90/69/41/21，三项 L 值判据全过（L_attn 21 vs 18、L_mlp 49 vs 51、L_all 69 = 69）⇒ **「VLM MLP 的 W4 敏感性 ≫ Attention」不依赖 Expert 是否为 raw FP**。因此 **Phase H 不再新增 S2 = VLM-FP8 + Expert-W4**：该配置会同时承受「VLM 侧 W4」与「Expert 侧 W4」两条损失，而 G6 已证明后者的贡献与前者不在同一量级（同一 VLM-W4 在 Expert-FP8 背景下仍然得到 69/41/21，与 G5 的 72/39/21 几乎一致），新增 S2 不会提供新信息。
- **修复 workload exporter**（Phase I）：改为 one physical operator, one row，且用 native corrected sparsity、A/B 分别缩放。
- **补充 attention-probability 分析**：pre-quant `probs` 的 entropy / top-k mass / FP8 threshold 以下比例，用于支撑 §3.4 的 PV A 趋势解释。
- **口径扩展**：当前为「4-bit significand（1MMM）」，未做 exponent-alignment（EffLoc ineffective-bit）。若论文要声称硬件对齐后的 bit sparsity，需 Phase I 实现。
- **G6**（Expert-FP8 背景 + VLM selective precision）已完成，结论：**G5 的「MLP ≫ Attention」对 Expert-FP8 背景鲁棒**（90/69/41/21；L_attn 21 / L_mlp 49 / L_all 69，三项 |Δ| ≤ 6pp 判据全过）。因此 **不新增 S2 = VLM-FP8 + Expert-W4**（理由见上）。
- **工程修复**（本次审计）：`build_results_tables.py` 新增 common-scope runtime / Expert-only weight（含去重）/ Pearson+Spearman；重命名 unit 列与 BOP 节；output/O 行在收敛表中标 N/A 而非 FAIL；修复 `summarize_sparsity.py` 的 `REPO_ROOT` 路径。

---

## 6. 修订记录

- 2026-09-13：创建模板。
- 2026-09-14：回填 H0 smoke 结果（Gate 全 PASS，S0/S1 SR=100%，weight sparse_bit_rate 41.5%/73.7%）。
- 2026-09-14：回填 H1（10ep）结果（S0 SR=90.0%、S1 SR=80.0%，runtime sparsity 分布）。
- 2026-09-14：回填 H1-Audit 结论（output/O 稀疏度 0.5%→39.9% 为统计 bug 修正，根因 `quant_methods.py` 3 处 `mul_` in-place，新增 T11 回归测试）。
- 2026-09-14：按 er.md 审阅修正 results.md 语义——H1/H0 的 output/O 标 INVALID（pre-fix 污染），39.9% 归为 H1-Audit S0 task0 post-fix 测量。
- 2026-09-15：回填 **H2（30ep）完整结果**——SR S0=90.0%/S1=86.7%；§14 全表；§13 收敛 gate 同口径全部 PASS；新增 `build_results_tables.py` 并修复 `summarize_sparsity.py` 路径 bug。
- 2026-09-15：**按 er.md 二次审计修正分析口径**——
  - §2/§3.7 weight 比较改为 **Expert common scope**（+31.15pp，原 +32.2pp 因 scope 不一致作废），并按 `module_id` 去重；
  - §2/§3.1 新增 **common-scope runtime aggregate**（Δ=+0.02pp，原 +0.34pp 作废）；
  - §3.3/§3.4 更正 **PV A = softmax 后的 attention probability `P`**（原「value/RoPE 激活」「不是量化产物」为错误解释）；
  - §3.7 明确 **unit sparsity 未做 native 修正**，改称 quant-path unit sparsity（2×2, masked）；
  - §3.7 BOP 节降级为 legacy/debug proxy，列出三处不可加缺陷；
  - §3.5 改为**实际计算**的 Pearson/Spearman 并弱化「无相关/固有难度」措辞；
  - §1/§2.1/§3.6 加入 Wilson CI、「扩展采样」措辞修正、pre-fix INVALID 表述澄清。
- 2026-09-19：**状态改 done；回填 H3（100ep）全部结果**——
  - §1/§2.1 正式准确率 S0 = **89.0%（89/100）**、S1 = **85.0%（85/100）**，gap = **4.0pp**，Wilson 95% CI `[81.4, 93.7]` / `[76.7, 90.7]`（仍重叠）；新增 H3 逐 task 成功数表。
  - §2.2 新增 H3 runtime / weight 行，指出 **H2 与 H3 的 aggregate 几乎逐位相同**。
  - §3.2 重命名为「收敛 gate H1→H2→H3」，新增 **§3.2.2 H2(30ep) → H3(100ep)**：19 项非 output/O 大类 **全部 |Δ| ≤ 0.05pp，gate 全 PASS**（比 H1→H2 更小）。
  - §3.5 新增 H3 100ep per-task 表与相关系数（pooled Pearson r = −0.0725、Spearman ρ = +0.1250），**task03 确认为 S1 独有弱项，task06/task07 排除**；并指出 `build_results_tables.py` 的「30 episodes」注释为 H2 遗留文案。
  - §4 结论 1/2/6 按 H3 正式值重写；§5 将 H3 与 G6 标为已完成，并据 G6 结论**明确不新增 S2**。
