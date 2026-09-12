# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 weight-error-audit
- **状态**：done（draft / running / done / aborted）
- **最后更新**：2026-09-11

---

## 1. 摘要

对全部 224 个物理 Linear 在 FP8 / W4 两套协议下做离线数值审计（8 个校准 batch 作前向数据，各层独立 scale）。核心结论：**W4 量化误差在 expert/vlm、各 layer、各 operator 间分布均匀，无「少量极端敏感 site」**——expert vs vlm 的 weight_nmse 为 0.0097 vs 0.0128（~1.3×），唯一异常点是 `vlm.layers.0.self_attn.v_proj`（output_nmse=0.0376，为次高 ~2×）。这排除了「单组件权重误差极大导致 F3 goal 崩溃」的简单假设，引导后续优先 G2（tensor-wise 粒度）与 G3（闭环累积）。

## 2. 总结果表

本 task 无 rollout（无 SR）。交付 5 个 CSV（见 §7 输出目录）：

| 交付物 | 内容 |
|---|---|
| `linear_error_stats.csv` | 224 层 × 2 协议全部 17 个误差字段 |
| `linear_error_ranked.csv` | W4 按 output_nmse 降序 |
| `component_summary.csv` | expert vs vlm 聚合 |
| `operator_summary.csv` | q/k/v/o/gate/up/down 聚合 |
| `layer_summary.csv` | (component, layer) 聚合 |

## 3. 分组结果与分析

### 3.1 component 维度（expert vs vlm）

| component | count | w4 weight_sqnr_db | w4 weight_nmse | w4 output_nmse | w4 output_sqnr_db |
|---|---:|---:|---:|---:|---:|
| expert | 112 | 20.13 | 0.0097 | 0.0074 | 21.46 |
| vlm | 112 | 18.99 | 0.0128 | 0.0120 | 19.34 |

分析：expert 与 vlm 的 W4 误差同量级（nmse 差 ~1.3–1.6×），**不构成「expert 主导」**。vlm 略差，与 hidden 维度更大（960 vs 720）、tensor-wise scale 更难覆盖行间分布一致。

### 3.2 operator 维度

| op_type | w4 weight_nmse | w4 output_nmse |
|---|---:|---:|
| down_proj | 0.0109 | 0.0086 |
| gate_proj | 0.0107 | 0.0091 |
| k_proj | 0.0122 | 0.0083 |
| o_proj | 0.0114 | **0.0119** |
| q_proj | 0.0124 | 0.0085 |
| up_proj | 0.0106 | 0.0100 |
| v_proj | 0.0106 | 0.0115 |

分析：o_proj / v_proj 的 output_nmse 略高（attention 输出投影），其余均匀；无 operator 级崩溃。

### 3.3 layer 维度（浅层 vs 深层）

无单调趋势。唯一异常：`vlm.layers.0` 的 output_nmse=0.0143（其余 vlm 层 0.010–0.013），其中 `vlm.layers.0.self_attn.v_proj` 以 output_nmse=0.0376 居 ranked 第 1（为第 2 名 ~2 倍）。ranked top10 几乎全是 vlm 的 o_proj/v_proj（浅层 0–3 层为主）。

### 3.4 量化健康度（W4 整体）

- `unique_quant_codes` ≈ 15（接近 int4 满格 16），`saturation_ratio` = 0 → scale 合理、无饱和；
- `weight_cosine` ≈ 0.993、`weight_sqnr_db` ≈ 19–20 dB → int4 量化的正常水平；
- `protected_ratio` ≈ 0.02（1% outlier 通道 ∪ 元素保护的放大后比例）。

## 4. 结论

1. **W4 数值误差是全局均匀的**，不存在集中在 expert、某 operator 或某层的「敏感 site」（唯一温和异常 `vlm.layers.0.v_proj` 也仅 ~2× 偏离）。
2. 由此，F3 goal 90%→21% 的崩溃**不能被局部权重误差解释**，更可能来自：tensor-wise W4 scale 粒度（→ G2）或 open-loop 误差累积（→ G3），而非 G1 能定位的单一组件。
3. G1 仍值得跑：G0 是数值误差层、G1 是闭环 SR 层，E1 已证明二者可背离；G1 直接回答「恢复哪个组件 FP 后 SR 回升最多」。

## 5. 问题与后续

- 审计脚本修复记录：`quant_awo` 对 int/fp 均返回 code（非去量化值），首版误将 code 当权重导致 nmse 虚高 ~1/scale²；已修正为 `w_sim = code * scale`；
- 建议优先启动 G1（component-localization），并视 G1 结果决定 G2/G3 顺序。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-11 | 创建文档，回填 G0 审计结果（5 CSV，含 code→去量化 bug 修复） | zyzhao |
|---|---|---|

分析：

## 4. 结论

<!-- 对 §1「实验目的」中每个问题的直接回答；可执行的结论（保留/否决某配置） -->

-

## 5. 问题与后续

<!-- 实验中发现的异常、失败 run、待验证问题、下一步实验（链接到新实验子目录） -->

-

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-11 | 创建文档 | |
