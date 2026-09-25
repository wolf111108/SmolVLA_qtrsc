# 结果记录（Results）

- **实验名称**：2026-09-25_phaseI_vision-joint-smoke
- **状态**：done（VJ1 一次跑通，check_outputs PASS）
- **最后更新**：2026-09-25

## 1. 摘要

VJ1（2026-09-25，commit `ca7e352`）一次跑通全链路：pytest 6 passed → preflight **PASS_WITH_BACKEND_DRIFT**（硬门槛全过）→ 校准（96 模块 × 3 role = **288 个 PoT scale**，150.8s，含 72 Linear 的 w/a/o 与 24 MatMul 的 A/B/O）→ Goal task0×1ep **SR=100%**（476.8s）→ 稀疏统计（runtime 216 行 + weight 72 行）→ **check_outputs PASS**。Vision 全量化（Linear+QK/PV）下 pooled runtime S\|MMM bit sparsity = **54.58%**，与 09-22 全 VLIN 实验的 Vision 通路量级一致。

## 2. 总结果表

| 组 | 协议 | Episode success | Runtime element / bit | Weight element / bit | 状态 |
|---|---|---|---|---|---|
| VJ1 | Goal task0×1 | ✅ 100% | Linear 4.07% / **48.56%**；MatMul 7.20% / **57.60%**；pooled 6.16% / **54.58%** | Linear elem 0.00056% / **bit 54.41%** | **done，check_outputs PASS** |

## 3. 分组结果与分析

### 3.1 Gate 核验（check_outputs 全 PASS）

| 项 | 要求 | 实测 |
|---|---|---|
| scale 文件 | 288 | 288 ✅（72 Linear×w/a/o + 24 MatMul×A/B/O，全部 PoT 2 的幂值） |
| manifest | 96 模块 | 96 ✅（72 Linear + 24 MatMul） |
| runtime 行 | 216 | 216 ✅（Linear 72×2 + MatMul 24×3，phase=prefill、attention_kind=self） |
| weight 行 | 72 | 72 ✅（static weight sparsity 72 行，总计 472/84,934,656 elem） |
| preflight | — | PASS_WITH_BACKEND_DRIFT（backend drift 与 09-23 实验同模式，仅报告） |
| episode | — | success=true ✅ |

### 3.2 分组稀疏统计（S\|MMM v1，native 口径）

| 分组 | element sparsity | S\|MMM bit sparsity | 分子/分母（bits） |
|---|---:|---:|---|
| runtime Linear（activation+output） | 4.07% | **48.56%** | 11,767,650,686 / 24,231,149,568 |
| runtime MatMul（A/B/O） | 7.20% | **57.60%** | 27,871,129,129 / 48,388,678,512 |
| runtime pooled | 6.16% | **54.58%** | 39,638,779,815 / 72,619,828,080 |
| static weight（Linear w） | 0.00056% | **54.41%** | 184,854,991 / 339,738,624 |

### 3.3 与历史实验交叉验证

- **Linear 通路与 09-22 全 VLIN 实验高度一致**：static weight bit 54.41% 完全复现（184,854,991/339,738,624 逐位一致）；runtime bit 48.56% vs 09-22 的 48.64%，微小差异来自本次为联合量化（QK/PV 也量化，前向数值略变）与单 episode 采样。
- **MatMul 通路与 09-23 QK/PV-only 实验一致**：runtime bit 57.60% vs 57.68%、element 7.20% vs 7.24%，联合量化下几乎不变。
- **09-23 实验遗留的 raw 等价性结论在新环境下未回退**：preflight 硬门槛（adapter_vs_eager 等）全过。

### 3.4 运行环境备注

- 校准 150.8s、rollout 476.8s（较 QK/PV-only 的 78s 显著变慢，因 96 个量化模块全部参与 fake-quant 前向）；耗时仅供参考，非性能结论。
- 首次启动时误用 base 环境致 pytest 收集失败，清理残留目录后在 `smolvla_eval` 下重跑，不影响结果。

## 4. 结论

1. Vision 全量化（72 Linear + 24 QK/PV，FP8 E4M3 PoT + 1% outlier）联合接入**一次跑通**，全部验收 Gate（288 scale / 96 manifest / 216 runtime / 72 weight）PASS，episode success。
2. 联合量化下两类算子稀疏度与各自单独实验**一致**（Linear ~48.6%、MatMul ~57.6%）， pooled **54.58%** 可作为「Vision 全量化」的 S\|MMM bit-zero headline。
3. static weight bit 54.41% 与 09-22 实验逐位复现，证明 Linear 量化路径与历史完全一致，联合接入未引入回归。
4. 单 episode 仅作工程验证，不作精度结论。

## 5. 问题与后续

- 原始 CSV/JSON 已随本次回填提交副本（`docs/` 下 `preflight_run1.json`、`coverage_summary_run1.json`、`result_run1.json`），可独立复核；正式数据在 `outputs/2026-09-25_phaseI_vision-joint-smoke/`。
- 通过后建议：全模型分组件 W8/W4 混合精度评测；多 episode 方差估计；与 eager raw 基线的正式量化掉点对比。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-25 | 创建联合验证实验 | |
| 2026-09-25 | 回填 VJ1：一次跑通全链路 PASS（288 scale / 216 runtime / 72 weight / SR=100%），分组稀疏与交叉验证见 §3 | |
