# Phase H 结果记录（Results）

> 实验名称：2026-09-13_phaseH_accuracy-preserving-sparsity  
> 状态：running  
> 最后更新：2026-09-14

---

## 1. 摘要

H0 correctness smoke 已通过全部 Gate：S0（FP8-all）与 S1（Expert-W4）各 1 episode 均 SR=100%，flow_step 0..9 齐全、Linear activation/output 与 MatMul A/B/O role 齐全、self/cross attention 区分、outlier sidepath 非空、native counter 无负值、static weight 行数与 wrapped QuantizedLinear 一致（S0=224、S1=112）。weight 静态稀疏度（与 episode 无关）：S0 sparse_bit_rate≈41.5%、S1≈73.7%，W4 权重确实产生显著更多 sparse bits。H1（10ep）运行中。

---

## 2. 总结果表

| Config | Stage | Episodes | SR | Native bit sparsity | Unit sparsity | FP sidepath | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| S0 FP8-all | H0 smoke | 1 | 100.0% | — | — | 1.30% | correctness smoke |
| S1 Expert-W4 | H0 smoke | 1 | 100.0% | — | — | 1.29% | correctness smoke |
| S0 FP8-all | H1 | 10 | | | | | 运行中 |
| S1 Expert-W4 | H1 | 10 | | | | | 运行中 |
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

| Metric | H1 | H2 | Δ(pp) |
|---|---:|---:|---:|
| VLM prefill A bit sparsity | | | |
| Expert denoise A bit sparsity | | | |
| Expert denoise O bit sparsity | | | |
| QK bit sparsity | | | |
| PV bit sparsity | | | |
| FP sidepath | | | |

### 3.3 Component / operator

TBD。

### 3.4 Flow step

TBD。

### 3.5 Task-wise correlation

TBD。

---

## 4. 结论

TBD。

---

## 5. 问题与后续

TBD。

---

## 6. 修订记录

- 2026-09-13：创建模板。
- 2026-09-14：回填 H0 smoke 结果（Gate 全 PASS，S0/S1 SR=100%，weight sparse_bit_rate 41.5%/73.7%）。
