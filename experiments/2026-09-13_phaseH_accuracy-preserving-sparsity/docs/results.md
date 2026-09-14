# Phase H 结果记录（Results）

> 实验名称：2026-09-13_phaseH_accuracy-preserving-sparsity  
> 状态：running  
> 最后更新：2026-09-14

---

## 1. 摘要

H0 correctness smoke 已通过全部 Gate；weight 静态稀疏度（与 episode 无关）：S0 sparse_bit_rate≈41.5%、S1≈73.7%，W4 权重确实产生显著更多 sparse bits。H1（10ep pilot）已完成：S0 FP8-all SR=90.0%、S1 Expert-W4 SR=80.0%（各 10 tasks × 1ep，task 级 0/1 粗粒度）。runtime significand sparse_bit_rate（native）：activation/A/B ≈ 40-56%，output/O ≈ 0.3-0.6%，S0 与 S1 之间几乎一致（稀疏度由 FP8 激活主导，W4 权重差异不在 activation 侧体现）。H2（30ep）待跑。

---

## 2. 总结果表

| Config | Stage | Episodes | SR | Native bit sparsity | Unit sparsity | FP sidepath | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| S0 FP8-all | H0 smoke | 1 | 100.0% | — | — | 1.30% | correctness smoke |
| S1 Expert-W4 | H0 smoke | 1 | 100.0% | — | — | 1.29% | correctness smoke |
| S0 FP8-all | H1 | 10 | 90.0% | — | — | 1.23% | 10 tasks × 1ep |
| S1 Expert-W4 | H1 | 10 | 80.0% | — | — | 1.27% | 10 tasks × 1ep |
| S0 FP8-all | H2 | 30 | | | | | |
| S1 Expert-W4 | H2 | 30 | | | | | |
| S0 FP8-all | H3 | 100 | | | | | |
| S1 Expert-W4 | H3 | 100 | | | | | |

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

> runtime 稀疏度（significand 4-bit 口径，reported/native）：
>
> | component | phase | role | reported | native |
> |---|---|---|---:|---:|
> | vlm | prefill | activation | 41.10% | 40.53% |
> | vlm | prefill | A / B | 50.61% / 41.17% | 50.20% / 39.89% |
> | expert | denoise | activation | 41.00% | 40.43% |
> | expert | denoise | A / B | 54.55% / 41.18% | 54.11% / 39.78% |
> | expert | denoise | output / O | 1.32% / 1.09% | 0.36% / 0.15% |
>
> 说明：activation/A/B 的 significand sparse_bit_rate ≈ 40-54%，不是 0%；output/O 显著偏低（≈1%）。这一指标是「4-bit significand（1MMM）」口径，尚未做 exponent-alignment（EffLoc 的 ineffective-bit 口径），因此不等于硬件对齐后真正的 bit sparsity——该口径需 Phase I 单独实现。W4 权重侧（weight_sparsity_static）S1≈73.7% 仍是主要稀疏来源。

### 3.2 H1 vs H2 收敛

H1 结果（native sparse_bit_rate，10ep）：

| Metric | H1 S0 | H1 S1 | H2 | Δ(pp) |
|---|---:|---:|---:|---:|
| VLM prefill activation | 40.70% | 40.65%(N/A) | | |
| Expert denoise A | 56.10% | 55.55% | | |
| Expert denoise activation | 40.58% | 40.65% | | |
| Expert denoise output | 0.50% | 0.51% | | |
| QK/PV A native | 50.6-56.1% | 50.7-55.6% | | |
| QK/PV O native | 0.3-0.6% | 0.3-0.6% | | |
| FP sidepath | 1.23% | 1.27% | | |

> 注：H1 是 1ep/task 的 pilot，SR 为 0/1 粗粒度（S0=9/10、S1=8/10）。sparsity 各指标 S0/S1 几乎一致（activation/A/B 由 FP8 主导，W4 只影响 weight 侧），符合预期。H2（3ep/task）跑完后对比收敛。

### 3.3 Component / operator

TBD。

### 3.4 Flow step

TBD。

### 3.5 Task-wise correlation

TBD。

---

## 4. 结论

- H0/H1 均已完成：S0（FP8-all）H1 SR=90.0%、S1（Expert-W4）H1 SR=80.0%，与 Phase G 参考（G1-A 88%、G1-D 84%）同量级。
- runtime sparsity 由 FP8 激活主导（activation/A/B ≈ 40-56%），output/O 极低（≈0.5%）；W4 稀疏只在 weight 侧（S1 weight sparse_bit_rate≈73.7%）。
- 最终 SR 与 sparsity 的稳定结论待 H2（30ep）/H3（100ep）。

## 5. 问题与后续

- H1 为 1ep/task，SR 噪声大；需 H2（3ep/task）判断收敛。
- H3（100ep）用于最终 Goal 表与 stats-on SR 复验。
- G6（Expert-FP8 背景 VLM selective precision）并行推进中，其结论将决定 Phase H 是否新增 S2=VLM-FP8+Expert-W4。

---

## 6. 修订记录

- 2026-09-13：创建模板。
- 2026-09-14：回填 H0 smoke 结果（Gate 全 PASS，S0/S1 SR=100%，weight sparse_bit_rate 41.5%/73.7%）。
- 2026-09-14：回填 H1（10ep）结果（S0 SR=90.0%、S1 SR=80.0%，runtime sparsity 分布）。
