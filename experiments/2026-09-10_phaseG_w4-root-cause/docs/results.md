# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-10_phaseG_w4-root-cause
- **状态**：running（draft / running / done / aborted）
- **最后更新**：2026-09-12

---

## 1. 摘要

<!-- 3-5 句话：做了什么、核心数字、最重要的结论 -->

Phase G 目标：归因 Phase F 中 F3（Linear W4）四 suite 平均 83.0% → 48.5%、goal 90% → 21% 的退化。G0（离线审计）已完成：W4 误差全局均匀、无敏感 site；G1（组件定位，goal×100ep×4 config）已完成：**崩溃完全来自 VLM 侧 W4**（仅 VLM W4 = 21% ≈ 全量化 21%；仅 Expert W4 = 84% ≈ FP8 anchor 88%）——数值误差仅 1.3× 之差但闭环 SR 差 63pp，指向 VLM 的闭环敏感度而非误差大小。下一 Gate：G2 weight-granularity（检验 tensor-wise 粒度根因，需先实现 `weight_quant_granularity` 字段）。

## 2. 总结果表

<!-- 每个 config 一行；数值带单位；与 baseline 的差值单独列；各 task 明细见其 docs/results.md -->

| task | 问题 | Goal SR | 状态 |
|---|---|---:|---|
| G0 weight-error-audit | 数值误差集中于何处 | —（无 rollout） | **done**（误差均匀，无敏感 site） |
| G1-A fp8_all | F1 anchor | **88.0%** | **done** |
| G1-B w4_all | F3 anchor | **21.0%** | **done** |
| G1-C w4_vlm_only | VLM 主导？ | **21.0%** | **done**（→ VLM 主导） |
| G1-D w4_expert_only | Expert 主导？ | **84.0%** | **done**（→ Expert 无强贡献） |
| G2 weight-granularity | tensor-wise 是根因？ | | pending（**需先实现 `weight_quant_granularity` 代码字段**，G1 后最高优先） |
| G3 action-horizon | 开环放大？ | | pending（G1 后降为次要：组件归因已干净） |
| G4 outlier-mask-diagnosis | mask 漂移？ | | conditional |

## 3. 分组结果与分析

<!-- 按实验变量分组展开：每组建一个小节，含数据表、图、简要分析 -->

### 3.1 G1 组件归因（libero_goal，100 ep/config）

| 取值 | Success Rate | 备注 |
|---|---:|---|
| A fp8_all（anchor） | 88.0% | F1 复现（F1 goal 90%） |
| B w4_all（F3 anchor） | 21.0% | F3 复现（完全一致） |
| C 仅 vlm.* W4 | 21.0% | = B，崩溃模式逐 task 复现（t1/t2/t4/t7→0） |
| D 仅 expert.* W4 | 84.0% | ≈ A，−4pp 噪声带内 |

分析：VLM-only W4 完整复现 F3 崩溃，Expert-only W4 几乎无损 → **VLM 量化贡献 ≥95% 退化**。与 G0 交叉：vlm/expert 权重 NMSE 仅 1.3×，但闭环 SR 差 63pp → 根因是 VLM 对均匀 W4 噪声的闭环敏感度，非误差大小。详见 `tasks/component-localization/docs/results.md`。

## 4. 结论

<!-- 对 §1「实验目的」中每个问题的直接回答；可执行的结论（保留/否决某配置） -->

1. **Q1（哪一部分造成）：VLM**。仅 VLM W4 即 21%（=F3），仅 Expert W4 84%（≈FP8）。
2. **Q2（W4 scale 粒度）：G0 已给出必要条件**——误差均匀无热点，排除「局部敏感 site」假设；tensor-wise 粒度是否根因待 G2 检验（G1 交叉证据支持：VLM hidden 960 > expert 720，tensor-wise scale 更难覆盖）。
3. **Q3（action horizon）：降为次要**。组件归因已干净，开环放大假设不再是主要矛盾；G3 保留但优先级降低。
4. **Q4（outlier mask）：维持 conditional**。G0 显示 saturation=0、unique_codes≈15、protected≈2%，量化健康度正常，无 mask 异常证据。

## 5. 问题与后续

<!-- 实验中发现的异常、失败 run、待验证问题、下一步实验（链接到新实验子目录） -->

- **G2 阻塞项**：`weight_quant_granularity`（per_tensor / per_output_channel / group 128/64/32）字段**尚未在代码中实现**（当前 w_interval 为全 tensor 单一标量）——需先开发，方案已调研待确认；
- G1-B/F3、G1-A/F1 的独立校准复现性检验通过（21%=21%、88% vs 90% 在噪声带内）；
- 下一步：实现 `weight_quant_granularity` → G2 四档对比 → 视结果决定 G3/G4。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-11 | 创建文档 | |
| 2026-09-12 | 回填 G1 四 config 结果（88/21/21/84），结论：VLM 主导；G2 升为最高优先（需先实现 granularity 字段） | zyzhao |
