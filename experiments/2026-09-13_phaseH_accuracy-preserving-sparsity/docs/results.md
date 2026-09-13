# Phase H 结果记录（Results）

> 实验名称：2026-09-13_phaseH_accuracy-preserving-sparsity  
> 状态：running  
> 最后更新：TBD

---

## 1. 摘要

TBD。

---

## 2. 总结果表

| Config | Stage | Episodes | SR | Native bit sparsity | Unit sparsity | FP sidepath | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| S0 FP8-all | H1 | 10 | | | | | |
| S1 Expert-W4 | H1 | 10 | | | | | |
| S0 FP8-all | H2 | 30 | | | | | |
| S1 Expert-W4 | H2 | 30 | | | | | |
| S0 FP8-all | H3 | 100 | | | | | |
| S1 Expert-W4 | H3 | 100 | | | | | |

---

## 3. 分组结果与分析

### 3.1 Correctness smoke

- static weight rows:
- prefill present:
- denoise steps:
- Linear roles:
- MatMul roles:
- outlier accounting:
- stats ON/OFF numerical equality:

判定：TBD。

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
