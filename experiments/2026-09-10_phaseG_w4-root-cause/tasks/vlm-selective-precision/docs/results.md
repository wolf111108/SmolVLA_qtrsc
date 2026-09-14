# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 vlm-selective-precision
- **状态**：done（draft / running / done / aborted）
- **最后更新**：2026-09-14

---

## 1. 摘要

在 Expert 固定为 raw FP（不量化）、只量化 VLM 的前提下，定位 VLM 内部 attention 与 MLP 两个子模块对 W4 权重量化的敏感度。结论：**VLM 的 MLP 比 attention 对 W4 更敏感**——只把 MLP 三个投影压到 W4 掉 51pp（90%→39%），只把 attention 四个投影压到 W4 掉 18pp（90%→72%）。VLM 全 W4（复用 G1-C）则直接崩到 21%。说明 W4 的精度损失主要来自 MLP 的 gate/up/down 权重。

## 2. 总结果表

| 组 | config | 基准 | Success Rate | Δ vs baseline | episodes | 输出目录 | 备注 |
|---|---|---|---|---|---|---|---|
| baseline | g5a_vlm_fp8_control | 90.0% | 90.0% | — | 100 | vlm-selective-precision | VLM 全 FP8，Expert raw FP |
| attn-W4 | g5b_vlm_attn_w4_mlp_fp8 | 90.0% | 72.0% | -18pp | 100 | vlm-selective-precision | VLM attention 4 个 proj W4 |
| mlp-W4 | g5c_vlm_attn_fp8_mlp_w4 | 90.0% | 39.0% | -51pp | 100 | vlm-selective-precision | VLM MLP 3 个 proj W4 |
| all-W4 | G5-D（复用 G1-C） | 90.0% | 21.0% | -69pp | 100 | component-localization | VLM 全 W4 |

## 3. 分组结果与分析

### 3.1 VLM 内部组件敏感度（attention vs MLP）

| 取值 | Success Rate | Δ vs control | 备注 |
|---|---|---|---|
| control（VLM 全 FP8） | 90.0% | — | Expert raw FP |
| attention-W4 | 72.0% | -18pp | q/k/v/o_proj → W4 |
| mlp-W4 | 39.0% | -51pp | gate/up/down_proj → W4 |
| VLM all-W4 | 21.0% | -69pp | 复用 G1-C |

分析：

- **MLP 是 VLM 内部 W4 掉精度的主要来源**：mlp-W4（39%）显著低于 attention-W4（72%），两者合计 51+18 ≈ 69pp，与 all-W4 的 -69pp 一致，说明两个子模块的损失近似可加。
- attention-W4 掉 18pp 仍属显著（超过经验噪声带），因此 attention 的 4 个投影也不能轻易压 W4，但相对 MLP 更耐受。
- 该实验用 per_tensor 权重粒度（weight_quant_granularity=per_tensor），后续可对比 per_output_channel 是否能缓解 MLP-W4 的崩溃（见 weight-granularity 子实验，Gate 2 已拦停 groupwise）。

## 4. 结论

- VLM 内部 W4 敏感度排序：**MLP > attention**，MLP-W4 是 W4 掉精度的主导因子。
- 单独 attention-W4（72%）或单独 MLP-W4（39%）都不满足 accuracy-preserving 目标；VLM 全 W4（21%）更不可接受。
- 因此 accuracy-preserving 的 W4 机会不在 VLM 内部，而应在 **Expert**（G1-D Expert-W4 = 84%，在噪声带内）——这也是 Phase H 选 S1=Expert-W4 作 accuracy-preserving anchor 的原因。

## 5. 问题与后续

- 未尝试 per_output_channel / groupwise 粒度（Gate 2 拦停 groupwise），W4 的粒度缓解方向待 Phase I 再评估。
- 本实验 Expert 固定 raw FP；与 Phase H 的 S1（Expert-W4 + VLM raw）方向相反，两者共同框定「W4 放 Expert、VLM 保持 FP8」的最优精度分配。
- 下一步：Phase H（accuracy-preserving sparsity）已在 `experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/` 开展。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-13 | 创建文档 | |
| 2026-09-14 | 回填 G5 三组 + G5-D 结果（90/72/39/21） | |
