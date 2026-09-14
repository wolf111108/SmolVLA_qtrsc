# Phase H 结果记录（Results）

> 实验名称：2026-09-13_phaseH_accuracy-preserving-sparsity  
> 状态：running  
> 最后更新：2026-09-14

---

## 1. 摘要

H0 correctness smoke 已通过全部 Gate；weight 静态稀疏度（与 episode 无关）：S0 sparse_bit_rate≈41.5%、S1≈73.7%，W4 权重确实产生显著更多 sparse bits。H1（10ep pilot）已完成：S0 FP8-all SR=90.0%、S1 Expert-W4 SR=80.0%（各 10 tasks × 1ep，task 级 0/1 粗粒度）。runtime native significand sparse-bit rate：activation/A/B ≈ 40-56%，output/O 经 H1-Audit 独立 E4M3 审计确认为 ≈39.9%（修复 instrumentation in-place aliasing 后，S0 task0 × 1ep；原 H1/H0 的 output/O CSV 已污染标 INVALID，见 §3.6）。H2（30ep）待跑。

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
> 说明：activation/A/B 的 native significand sparse_bit_rate ≈ 40-54%，不是 0%。**output/O 在原 H0/H1 中产生于 instrumentation bug 修复前（`mul_` in-place 污染了 output code），因此原始 CSV 的 output/O 字段已失效，标 INVALID**；修复后的唯一可靠测量来自 H1-Audit（S0 Goal task0 × 1ep）：output native ≈ 39.90%、O native ≈ 39.91%，与独立 E4M3 raw-code audit 完全一致（见 §3.6）。H2 将用修复后的 collector 得到正式 multi-task aggregate。这一指标是「4-bit significand（1MMM）」口径，尚未做 exponent-alignment（EffLoc 的 ineffective-bit 口径），因此不等于硬件对齐后真正的 bit sparsity——该口径需 Phase I 单独实现。W4 权重侧（weight_sparsity_static）S1≈73.7% 仍是主要稀疏来源。

### 3.2 H1 vs H2 收敛

H1 结果（native significand sparse-bit rate，10ep）：

| Metric | H1 S0 | H1 S1 | H1-Audit S0 task0 | H2 |
|---|---:|---:|---:|---:|
| VLM prefill activation | 40.70% | 40.65%(N/A) | — | |
| Expert denoise A | 56.10% | 55.55% | — | |
| Expert denoise activation | 40.58% | 40.65% | — | |
| Expert denoise output | **INVALID (pre-fix)** | **INVALID (pre-fix)** | **39.90%** | |
| QK/PV A native | 50.6-56.1% | 50.7-55.6% | — | |
| QK/PV O native | **INVALID (pre-fix)** | **INVALID (pre-fix)** | **39.91%** | |
| FP sidepath | 1.23% | 1.27% | — | |

> 注：H1 是 1ep/task 的 pilot，SR 为 0/1 粗粒度（S0=9/10、S1=8/10）。activation/A/B/FP sidepath 这些指标 S0/S1 几乎一致（由 FP8 主导，W4 只影响 weight 侧），且产生于 output code mutation 之前或来自独立 static collector，因此仍然有效。**output/O 的原始 H1 10ep CSV 产生于修复前，已标 INVALID，不作为 H1 aggregate；修复后唯一可靠测量来自 H1-Audit（S0 Goal task0 ×1ep）= 39.90%/39.91%。** H2（3ep/task）将用修复后的 collector 得到正式 multi-task aggregate，届时判断是否稳定在 ≈39.9%。

### 3.3 Component / operator

TBD。

### 3.4 Flow step

TBD。

### 3.5 Task-wise correlation

TBD。

### 3.6 H1-Audit：output/O 稀疏度统计 bug 的独立验证与修复

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

> **测量口径**：修复后的 39.90%/39.91% 是 **H1-Audit S0 Goal task0 × 1ep** 的测量值（不是原 H1 10ep 或 H0 的 aggregate——那些是 pre-fix 污染的）。正式 multi-task aggregate 待 H2（10 tasks × 3ep）用修复后的 collector 产出。

---

## 4. 结论

- H0/H1 均已完成：S0（FP8-all）H1 SR=90.0%、S1（Expert-W4）H1 SR=80.0%，与 Phase G 参考（G1-A 88%、G1-D 84%）同量级。
- runtime native significand sparsity 由 FP8 激活主导（activation/A/B ≈ 40-56%，output/O ≈ 39.9% post-fix）；W4 稀疏只在 weight 侧（S1 weight sparse_bit_rate≈73.7%）。
- output/O 早前的 ≈0.5% 已确认为 instrumentation bug（§3.6），修复后 S0 task0 audit ≈39.9%；正式 aggregate 待 H2。
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
- 2026-09-14：回填 H1-Audit 结论（output/O 稀疏度 0.5%→39.9% 为统计 bug 修正，根因 `quant_methods.py` 3 处 `mul_` in-place，新增 T11 回归测试）。
- 2026-09-14：按 er.md 审阅修正 results.md 语义——H1/H0 的 output/O 标 INVALID（pre-fix 污染），39.9% 归为 H1-Audit S0 task0 post-fix 测量，正式 aggregate 待 H2。
