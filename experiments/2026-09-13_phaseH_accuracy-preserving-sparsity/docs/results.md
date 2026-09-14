# Phase H 结果记录（Results）

> 实验名称：2026-09-13_phaseH_accuracy-preserving-sparsity  
> 状态：running  
> 最后更新：2026-09-15

---

## 1. 摘要

**H2（30ep convergence）已完成**（2026-09-15，20/20 tasks 无报错）。准确率：**S0 FP8-all SR=90.0%**、**S1 Expert-W4 SR=86.7%**（各 10 tasks × 3ep = 30 episodes，episode 加权），gap=3.3pp。

Sparsity 侧（native significand sparse-bit rate，4-bit 1MMM 口径，numerator/denominator 加总）：

| 指标 | S0 FP8-all | S1 Expert-W4 |
|---|---:|---:|
| native bit sparsity（全量化算子） | **42.08%** | **42.42%** |
| unit sparsity（2×2 unit） | 7.36% | 7.99% |
| FP sidepath ratio | 1.23% | 1.27% |
| weight static sparse_bit_rate | 41.52% | **73.72%** |

**关键结论**：

1. **§13 收敛 gate 成立**。所有非 output/O 的 aggregate 大类在 H1(10ep)→H2(30ep) 间 |ΔS_bit| ≤ 0.14pp、|Δunit| < 0.5pp，远低于 0.5pp gate ⇒ **sparsity characterization 在 10ep 即已收敛，不依赖 100ep**。
2. **H1-Audit 结论被 H2 独立复现**。H1-Audit（S0 goal task0 × 1ep）测得 output=39.90%、O=39.91%；H2 30ep×10task aggregate 给出 output=39.89-39.90%、O=39.87-39.95%（S0/S1 一致）⇒ §3.6 的 instrumentation 修复得到 multi-task 层面的确认，**H1/H0 的 output/O INVALID 标注正式解除**。
3. runtime sparsity **由 FP8 激活主导**（activation/A/B ≈ 39.9-61.0%），W4 只贡献 weight 侧（S1 weight sparse_bit_rate 41.5%→73.7%，+32.2pp）。因此 S0/S1 的 runtime bit sparsity 几乎相同（42.08% vs 42.42%，Δ=0.34pp），而 SR 只差 3.3pp。
4. **稀疏度与 task 难度解耦**：per-task native bit sparsity 在 42.0-42.5% 窄带内，与 per-task SR（33.3%→100%）无相关 ⇒ sparsity 是体系结构/量化属性，不是任务属性。
5. output/O 的 unit zero rate **不是** ~90% 的高稀疏（那是 pre-fix bug 的伪值）：修复后 output/O 的 2×2 unit zero rate ≈ 3.5%，与 activation 一致。

> **口径声明**：本节的 native bit sparsity 是「4-bit significand（1MMM）」口径，**尚未做 exponent-alignment**（EffLoc 的 ineffective-bit 口径），因此不等于硬件对齐后真正的 bit sparsity——该口径需 Phase I 单独实现。`BOP_active_proxy`/`ideal_sparse_upper_bound` 是**代理量**，不是实测加速比（manual §55 / experiment_setup §7.7）。


---

## 2. 总结果表

### 2.1 准确率（episode 加权，Goal suite，n_action_steps=10）

| Config | H0 smoke (1ep) | H1 (10 tasks × 1ep) | H2 (10 tasks × 3ep) | H3 (100ep) |
|---|---:|---:|---:|---:|
| S0 FP8-all | 100.0% | 90.0% | **90.0%** | pending |
| S1 Expert-W4 | 100.0% | 80.0% | **86.7%** | pending |
| gap (S0−S1) | 0.0 | 10.0 pp | **3.3 pp** | — |

### 2.2 Sparsity（numerator/denominator 加总，native = 扣除 FP protected 后）

| Config | Stage | Episodes | 口径 | Native bit sparsity | Unit sparsity | FP sidepath |
|---|---|---:|---|---:|---:|---:|
| S0 FP8-all | H0 smoke | 1 | 全口径（**pre-fix**） | 25.59% | 43.68% | 1.23% |
| S1 Expert-W4 | H0 smoke | 1 | 全口径（**pre-fix**） | 26.61% | 42.56% | 1.27% |
| S0 FP8-all | H1 | 10 | 排除 pre-fix output/O | 43.60% | 9.72% | 1.43% |
| S1 Expert-W4 | H1 | 10 | 排除 pre-fix output/O | 44.01% | 10.46% | 1.47% |
| **S0 FP8-all** | **H2** | **30** | **全口径（post-fix）** | **42.08%** | **7.36%** | **1.23%** |
| **S1 Expert-W4** | **H2** | **30** | **全口径（post-fix）** | **42.42%** | **7.99%** | **1.27%** |
| S0 FP8-all | H2 | 30 | 排除 output/O（与 H1 同口径） | 43.59% | 9.72% | 1.43% |
| S1 Expert-W4 | H2 | 30 | 排除 output/O（与 H1 同口径） | 44.03% | 10.50% | 1.47% |
| S0/S1 | H3 | 100 | — | pending | pending | pending |

> **口径说明**：H0/H1 的 output/O 行产生于 instrumentation 修复前（§3.6），其 CSV 值 ≈0，会把「全口径」aggregate 拖到 25-27%、「unit sparsity」抬到 42-44%。因此 H0/H1 不能按全口径与 H2 比较；同口径（排除 output/O）下 H1 与 H2 差异仅 0.01-0.02pp（见 §3.2）。H2 为 post-fix 正式 aggregate，两种口径都给出。
>
> 生成命令：
>
> ```bash
> conda run -n smolvla_eval python experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/build_results_tables.py \
>   --stage h2_30ep --compare h1_10ep                       # 全口径 + 收敛 gate
> conda run -n smolvla_eval python experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/build_results_tables.py \
>   --stage h2_30ep --exclude-roles output,O                # 与 H1 同口径
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

### 3.2 H1 vs H2 收敛（§13 gate：|ΔS_bit| < 0.5pp）

同口径（排除 pre-fix 的 output/O）下，H1(10ep) → H2(30ep) 的 aggregate 对比：

| Metric | H1 S0 | H2 S0 | Δ | H1 S1 | H2 S1 | Δ | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| native bit sparsity（全量化算子） | 43.60% | 43.59% | **−0.01 pp** | 44.01% | 44.03% | **+0.02 pp** | ✅ PASS |
| unit sparsity（2×2） | 9.72% | 9.72% | **0.00 pp** | 10.46% | 10.50% | **+0.04 pp** | ✅ PASS |
| FP sidepath | 1.43% | 1.43% | 0.00 pp | 1.47% | 1.47% | 0.00 pp | ✅ PASS |
| SR | 90.0% | 90.0% | 0.0 pp | 80.0% | 86.7% | +6.7 pp | — |

逐 component（post-fix 全口径，H1 → H2），全部 aggregate 大类：

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

**结论：gate 全部 PASS**（最大 |Δ| = 0.14pp，S1 EXPERT.PV A），unit rate 同样全部 < 0.5pp。⇒ **sparsity characterization 在 10ep 即已收敛，H3 的 100ep 主要用于 stats-on SR 复验，而非 sparsity 收敛。**

> output/O 的表现单独看（H1 列为 pre-fix 污染值，不参与 gate）：

| Role | H1（pre-fix，INVALID） | H1-Audit（S0 task0，post-fix） | H2 S0（30ep） | H2 S1（30ep） |
|---|---:|---:|---:|---:|
| output (linear) | 0.40% | **39.90%** | **39.89%** | **39.90%** |
| O (QK/PV) | 0.02-2.21% | **39.91%** | **39.87-39.95%** | **39.87-39.94%** |

**H1-Audit 的单 task post-fix 测量（39.90%/39.91%）被 H2 的 30ep × 10task aggregate（39.87-39.95%）精确复现** ⇒ §3.6 的修复在 multi-task 层面得到独立确认，H1/H0 的 output/O INVALID 标注正式解除，且 output/O 与 activation/A/B 同量级（≈39.9%）的判断成立。

### 3.3 Component / operator

**Component 级（H2，post-fix 全口径，30ep aggregate）**

| Config | Component | Phase | Role | Zero native | Bit sparse native | Unit sparse | FP sidepath |
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

> S1 无 VLM linear 行（`linear.include: expert.*`，VLM 保持 raw FP，与 G1-D 一致）；VLM 的 QK/PV MatMul 仍被量化（`quantize_matmul: true`）。

**要点**

- **PV 的 A 矩阵是唯一的高稀疏算子**：≈61%（EXPERT.PV）/ 54%（VLM.PV），element zero rate 也显著（31.5% / 18.4%），unit zero rate 34-35%。这是 RoPE/位置编码后 value 激活的固有结构，不是量化产物。
- 其余所有 role **收敛到 39.9-40.7%** 的窄带，说明该窄带由 E4M3 的 significand 位模式（1MMM 口径下 FP8 值本身的可压缩性）主导，而非算子/层位置。
- **S0 与 S1 在同 role 上差异 ≤ 0.4pp**：W4 只改 weight 的存储宽度，不改运行时 activation/output 的 FP8 code。
- output/O 的 unit zero rate ≈ 3.4-5.6%，**不是** pre-fix 时期的 ~90% —— 后者是 (code × scale) 巨大值破坏 2×2 unit 对齐造成的伪高稀疏。

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

- QK A/B/O 与 PV B/O 在全部 10 个 flow step 上**几乎完全平坦**（变化 < 0.02pp），说明 diffusion 去噪的时间维不改变激活的 significand 位模式统计。
- **唯一有趋势的是 PV A**：呈倒 U 形，step0 → step3-4 上升（S0: 61.02%→61.38%），随后单调下降至 step9（59.85%）；S1 同形状但幅度略小（60.29%→60.81%→59.41%）。峰谷差 ≈1.5pp。
- 该趋势不影响 headline 数字（PV A 在 denoise 上的加权平均 ≈60.99% / 60.43%）。

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

**要点**

- per-task bit sparsity 全部落在 **42.00-42.48%** 窄带（跨 task 极差 0.48pp，跨 config 差 0.34pp），而 SR 从 33.3% 到 100% 变化 ⇒ **sparsity 与 task 难度/SR 无相关**（共 20 个 (config, task) 点，相关性实质为 0）。
- **task06/task07 在两个 config 上同步掉分**（33.3%/66.7%）⇒ 这是 **task 固有难度 + 3ep 采样方差**，不是量化配置差异。
- **task03 只在 S1 掉分**（66.7% vs S0 100%）⇒ 这是唯一可能是 W4 精度损失的信号，需 H3 100ep 复验。

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

**H2 独立复现（已闭环）**：H2（10 tasks × 3ep，post-fix collector）给出的 aggregate 为 output = **39.89%（S0）/ 39.90%（S1）**、O = **39.87-39.95%（S0）/ 39.87-39.94%（S1）**，与 H1-Audit 的单 task 值（39.90%/39.91%）在 **0.05pp 内**一致。⇒ 修复正确、非 task-specific、非采样偶然。**H1/H0 的 output/O INVALID 标注至此正式解除。**

### 3.7 Weight static 与 workload 代理

**Weight static sparsity（episode 无关，`stat_semantics=full_weight_quantized_no_dynamic_outlier_mask`）**

| Config | Rows | zero_rate | sparse_bit_rate | ideal_sparse_upper_bound |
|---|---:|---:|---:|---:|
| S0 FP8-all | 2240 (10 × 224) | 0.00% | 41.52% | 1.710 |
| S1 Expert-W4 | 1120 (10 × 112) | 8.68% | **73.72%** | 3.806 |

> **W4 是唯一显著提升可压缩性的手段**：S1 的 weight sparse_bit_rate 比 S0 高 **+32.2pp**（73.72% vs 41.52%），element zero_rate 从 0% 升到 8.68%。但这是 **weight 侧**收益；runtime activation/output 仍受 FP8 主导（§3.3），故 S0/S1 的 runtime aggregate 只差 0.34pp。

**Workload / BOP 代理（仅已量化算子集合内可加）**

| Config | Rows | BOP dense proxy | BOP active proxy | active/dense | Calls |
|---|---:|---:|---:|---:|---:|
| S0 FP8-all | 35200 | 2.3928e14 | 7.9557e13 | 0.3325 (66.8%) | 1,474,880 |
| S1 Expert-W4 | 32960 | 1.4839e14 | 4.7602e13 | 0.3208 (67.9%) | 1,430,464 |

> `BOP_active = MACs · b_A · b_B`，其中 `b = B_full · (1 − S_bit)`（`stat_manager.export_workload_csv`，manual §55）。因此 S0 的 0.3325 ≈ (1−0.42)² = 0.336 —— 该代理把 A/B 两侧都用同一 bit-sparsity 缩放，是**一阶代理**，**不是**实测加速比或 whole-model MAC coverage（raw-op runtime denominator 未 instrument，按 experiment_setup §14.1 记 **N/A**）。
>
> `ideal_sparse_upper_bound = 1/(1−S_bit)` 同样只是 bit-level 上界。若要声称「whole-model 有 X% sparsity」必须补齐 denominator（experiment_setup §16 禁止）。

---

## 4. 结论

1. **准确率**：H2（30ep）S0 FP8-all SR=**90.0%**、S1 Expert-W4 SR=**86.7%**，gap 从 H1 的 10.0pp 收窄到 **3.3pp**。S0 在 1ep→3ep 下稳定在 90.0%（与 Phase G 参考 G1-A 88% 同量级），说明 H1 的 90.0% 不是采样噪声；S1 从 80.0%→86.7%，H1 的部分 gap 是单 episode 方差。
2. **收敛性**：§13 gate 在**同口径**下全部 PASS（最大 |ΔS_bit| = 0.14pp，|Δunit| < 0.5pp）⇒ **sparsity characterization 在 10ep 即已收敛，不需要 100ep 才稳定**。H3 的价值转为 stats-on SR 复验（100ep Goal 表）。
3. **稀疏来源**：runtime native significand sparsity **由 FP8 激活主导**（activation/A/B ≈ 39.9-61.0%），**W4 只贡献 weight 侧**（weight sparse_bit_rate 41.5%→73.7%，+32.2pp）。因此 S0/S1 的 runtime aggregate 仅差 0.34pp（42.08% vs 42.42%），而 SR 差 3.3pp。
4. **唯一结构性的高稀疏算子**：PV 的 A 矩阵（EXPERT 61.0%、VLM 54.3%），element zero rate 亦高（31.5%/18.4%）。其余 role 全部收敛到 39.9-41.0% 窄带。
5. **Flow step 不显著**：QK A/B/O、PV B/O 在 flow_step 0-9 上平坦（<0.02pp）；PV A 呈倒 U 形（峰谷 ≈1.5pp），不影响 headline。
6. **Sparsity 与 task 解耦**：per-task bit sparsity 全部落在 42.00-42.48%，与 per-task SR（33.3%→100%）无相关。**task06/task07 在两个 config 上同步掉分** ⇒ task 固有难度/采样方差；**只有 task03 是 S1 独有掉分**，是唯一可能是 W4 损失的信号。
7. **统计链路正确性已闭环**：output/O 的 ≈0.5% 为 instrumentation bug（§3.6），修复后 H1-Audit 单 task 值（39.90%/39.91%）被 H2 30ep aggregate 在 0.05pp 内复现；output/O 的 unit zero rate 由 pre-fix 伪值 ~90% 回落到 ~3.5%，与 activation 一致。

## 5. 问题与后续

- **H3（100ep Goal formal）**：用于最终 Goal 表与 stats-on SR 复验；重点观察 S1 task03 是否收敛回 100%、S0/S1 gap 是否稳定在 3.3pp 附近。
- **口径扩展**：当前为「4-bit significand（1MMM）」口径，**未做 exponent-alignment**（EffLoc ineffective-bit）。若论文要声称硬件对齐后的 bit sparsity，需 Phase I 实现该口径。
- **统计算子的可加性**：`BOP_active_proxy` 对 A/B 两侧使用同一 S_bit 缩放，是过度简化的代理；若要做 deployment 级 energy/latency 论断，需要在 stat_manager 中分别记录 A/B 的 bit-sparsity。
- **G6（Expert-FP8 背景 + VLM selective precision）** 并行推进中，其结论将决定 Phase H 是否新增 S2 = VLM-FP8 + Expert-W4（用于检验「把 VLM 压到 FP8 会额外损失多少」）。
- **工程修复**：本次汇总发现并修复 `summarize_sparsity.py` 的 `REPO_ROOT` 路径少一层 `..`（原脚本从仓库根无法运行），并新增 `build_results_tables.py`（§14 全表生成，支持 `--exclude-roles` 同口径对比）。

---

## 6. 修订记录

- 2026-09-13：创建模板。
- 2026-09-14：回填 H0 smoke 结果（Gate 全 PASS，S0/S1 SR=100%，weight sparse_bit_rate 41.5%/73.7%）。
- 2026-09-14：回填 H1（10ep）结果（S0 SR=90.0%、S1 SR=80.0%，runtime sparsity 分布）。
- 2026-09-14：回填 H1-Audit 结论（output/O 稀疏度 0.5%→39.9% 为统计 bug 修正，根因 `quant_methods.py` 3 处 `mul_` in-place，新增 T11 回归测试）。
- 2026-09-14：按 er.md 审阅修正 results.md 语义——H1/H0 的 output/O 标 INVALID（pre-fix 污染），39.9% 归为 H1-Audit S0 task0 post-fix 测量，正式 aggregate 待 H2。
- 2026-09-15：回填 **H2（30ep）完整结果**——SR S0=90.0%/S1=86.7%；§14 全表（summary / component / operator / flow-step / weight-static / workload-BOP / task-wise）；§13 收敛 gate 同口径全部 PASS；output/O INVALID 正式解除（H1-Audit 值被 H2 aggregate 复现）；新增 `build_results_tables.py` 并修复 `summarize_sparsity.py` 路径 bug。
