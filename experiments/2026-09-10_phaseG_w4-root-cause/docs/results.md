# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-10_phaseG_w4-root-cause
- **状态**：done（draft / running / done / aborted）
- **最后更新**：2026-09-14

---

## 1. 摘要

<!-- 3-5 句话：做了什么、核心数字、最重要的结论 -->

Phase G 目标：归因 Phase F 中 F3（Linear W4）四 suite 平均 83.0% → 48.5%、goal 90% → 21% 的退化。G0（离线审计）：W4 误差全局均匀、无敏感 site；G1（组件定位）：**崩溃完全来自 VLM 侧 W4**（仅 VLM W4=21% ≈ 全量化 21%；仅 Expert W4=84% ≈ FP8 anchor 88%）；G2（粒度）：per-channel 仅 +2pp，Gate 2 拦停 groupwise，根因收敛为 VLM 对 W4 噪声的闭环敏感度。G5（VLM 内部）：**MLP 主导**——MLP-W4 -51pp（39%）远大于 attention-W4 -18pp（72%），两者近似可加。G6（Expert-FP8 背景）升级 `linear.overrides` 验证该结论鲁棒性：**四组全部完成，SR = 90/69/41/21，三项判据（|ΔL|≤6pp）全过** → 结论对 Expert 精度背景**鲁棒**。

**Phase G 全部闭环实验已完成（2026-09-19 结题）。** G0/G1/G2/G5/G6 共 12 个 Goal×100 闭环 run（含 2 次复用）与 2 项离线审计；G3 已降为次要、G4 保持 conditional，均未启动。

## 2. 总结果表

<!-- 每个 config 一行；数值带单位；与 baseline 的差值单独列；各 task 明细见其 docs/results.md -->

| task | 问题 | Goal SR | 状态 |
|---|---|---:|---|
| G0 weight-error-audit | 数值误差集中于何处 | —（无 rollout） | **done**（误差均匀，无敏感 site） |
| G1-A fp8_all | F1 anchor | **88.0%** | **done** |
| G1-B w4_all | F3 anchor | **21.0%** | **done** |
| G1-C w4_vlm_only | VLM 主导？ | **21.0%** | **done**（→ VLM 主导） |
| G1-D w4_expert_only | Expert 主导？ | **84.0%** | **done**（→ Expert 无强贡献） |
| G2 weight-granularity | tensor-wise 是根因？ | **23.0%**（vs 21.0%，Δ+2pp） | **done**（per-channel 失败，Gate 2 拦停 groupwise 闭环；根因收敛为 VLM 对 W4 噪声的闭环敏感度） |
| G3 action-horizon | 开环放大？ | | pending（G1 后降为次要：组件归因已干净） |
| G4 outlier-mask-diagnosis | mask 漂移？ | | conditional |
| G5 vlm-selective-precision | VLM 内 attention vs MLP | 90/72/39/21（control/attn-W4/mlp-W4/all-W4） | **done**（→ MLP 主导，MLP-W4 -51pp、attention-W4 -18pp） |
| G6 vlm-selective-expert-fp8 | Expert-FP8 背景鲁棒？ | 90/69/41/21（control/attn-W4/mlp-W4/all-W4） | **done**（→ 三项判据全过，结论鲁棒） |

## 3. 分组结果与分析

<!-- 按实验变量分组展开：每组建一个小节，含数据表、图、简要分析 -->

### 3.1 G1 组件归因（libero_goal，100 ep/config）

![G1 component localization](figures/phaseG_G1_component_localization.png)

| 取值 | Success Rate | 备注 |
|---|---:|---|
| A fp8_all（anchor） | 88.0% | F1 复现（F1 goal 90%） |
| B w4_all（F3 anchor） | 21.0% | F3 复现（完全一致） |
| C 仅 vlm.* W4 | 21.0% | = B，崩溃模式逐 task 复现（t1/t2/t4/t7→0） |
| D 仅 expert.* W4 | 84.0% | ≈ A，−4pp 噪声带内 |

分析：VLM-only W4 完整复现 F3 崩溃，Expert-only W4 几乎无损 → **VLM 量化贡献 ≥95% 退化**。与 G0 交叉：vlm/expert 权重 NMSE 仅 1.3×，但闭环 SR 差 63pp → 根因是 VLM 对均匀 W4 噪声的闭环敏感度，非误差大小。详见 `tasks/component-localization/docs/results.md`。

> G1 与 G5 两张图均使用 **Goal 绿色系** 的四档明度（由浅到深）。其中后三档 `#a1d99b / #31a354 / #006d2c` 与 Phase F 中 Goal 柱在 F1/F2/F3 使用的颜色**完全一致**，因此 F 与 G 两套图的绿色语言可以直接对读。

### 3.2 G5 VLM 内部 selective precision（Expert raw FP）

![G5 VLM internals](figures/phaseG_G5_vlm_internals.png)

| 取值 | Success Rate | Δ vs control | 备注 |
|---|---:|---:|---|
| control（VLM 全 FP8） | 90.0% | — | Expert raw FP |
| attention-W4（q/k/v/o） | 72.0% | -18pp | |
| mlp-W4（gate/up/down） | 39.0% | -51pp | |
| VLM all-W4（复用 G1-C） | 21.0% | -69pp | |

分析：**MLP 是 VLM 内部 W4 掉精度的主要来源**——MLP-W4（39%）显著低于 attention-W4（72%），51+18 ≈ 69pp 与 all-W4 的 -69pp 近似可加。说明 W4 的精度损失主要来自 MLP 的 gate/up/down 权重。详见 `tasks/vlm-selective-precision/docs/results.md`。

> 图上四根柱均为 Goal 绿色系，由浅到深对应对 W4 的敏感度递增（control → attention → MLP → all），与 Phase F 四 suite 图的配色体系一致。

### 3.3 G6 VLM selective precision（Expert-FP8 背景）

| 组 | config | 期望对照 | Success Rate | Δ vs G5 | 状态 |
|---|---|---|---:|---:|---|
| G6-A | g6a_all_fp8_control | G5-A 90% | **90.0%** | 0.0 | done |
| G6-B | g6b_vlm_attn_w4_expert_fp8 | G5-B 72% | **69.0%** | −3.0 | done |
| G6-C | g6c_vlm_mlp_w4_expert_fp8 | G5-C 39% | **41.0%** | +2.0 | done |
| G6-D | g6d_vlm_all_w4_expert_fp8 | G5-D 21% | **21.0%** | 0.0 | done |

**L 值对照（相对各自 control 的损失）：**

| 指标 | G6（Expert FP8） | G5（Expert raw FP） | Δ | 判定（\|Δ\|≤6pp） |
|---|---:|---:|---:|:--:|
| L_attn | **21.0** | 18.0 | +3.0 | ✅ |
| L_mlp | **49.0** | 51.0 | −2.0 | ✅ |
| L_all | **69.0** | 69.0 | 0.0 | ✅ |

分析：为验证 G5「MLP > Attention 敏感性」在 Expert 也量化 FP8 的实际部署背景下是否仍成立，升级 `model_wrapper.py` 新增 `linear.overrides`（component/module-aware precision routing）。framework Gate 0-5 全 PASS（routing 单测 8、四组路由 count 224/0·160/64·176/48·112/112、raw Linear bit-exact、四组 calibration 各 864 scale、task0×1 各 SR=100%）。

**结论：三项判据全部通过** —— 最大偏差仅 +3pp（远小于 ±5.7pp 噪声带），且 **L_mlp / L_attn = 49/21 = 2.33×** 仍为高度主导（G5 为 2.83×）；可加性成立（21+49=70 ≈ L_all 69）。**G5 的核心结论对 Expert precision background 鲁棒。** 详见 `tasks/vlm-selective-expert-fp8/docs/results.md`。

## 4. 结论

<!-- 对 §1「实验目的」中每个问题的直接回答；可执行的结论（保留/否决某配置） -->

1. **Q1（哪一部分造成）：VLM**。仅 VLM W4 即 21%（=F3），仅 Expert W4 84%（≈FP8）。
2. **Q2（W4 scale 粒度）：G0 已给出必要条件**——误差均匀无热点，排除「局部敏感 site」假设；tensor-wise 粒度是否根因待 G2 检验（G1 交叉证据支持：VLM hidden 960 > expert 720，tensor-wise scale 更难覆盖）。
3. **Q3（action horizon）：降为次要**。组件归因已干净，开环放大假设不再是主要矛盾；G3 保留但优先级降低。
4. **Q4（outlier mask）：维持 conditional**。G0 显示 saturation=0、unique_codes≈15、protected≈2%，量化健康度正常，无 mask 异常证据。
5. **G5（VLM 内部）：MLP 主导**。MLP-W4 -51pp（39%）远大于 attention-W4 -18pp（72%），两者近似可加（-69pp）→ W4 精度损失集中在 MLP gate/up/down。
6. **G6（Expert-FP8 背景）：结论鲁棒。** framework 升级通过全部静态/等价/校准 Gate；四组闭环 SR = 90/69/41/21，三项判据（|ΔL| ≤ 6pp）全过（+3 / −2 / 0），且 L_mlp 仍为 L_attn 的 2.33×。「VLM MLP 的 W4 敏感性 ≫ Attention」不依赖 Expert 是否为 raw FP。

## 5. 问题与后续

<!-- 实验中发现的异常、失败 run、待验证问题、下一步实验（链接到新实验子目录） -->

- **G2 阻塞项**：`weight_quant_granularity`（per_tensor / per_output_channel / group 128/64/32）字段**尚未在代码中实现**（当前 w_interval 为全 tensor 单一标量）——需先开发，方案已调研待确认；
- G1-B/F3、G1-A/F1 的独立校准复现性检验通过（21%=21%、88% vs 90% 在噪声带内）；
- 下一步：实现 `weight_quant_granularity` → G2 四档对比 → 视结果决定 G3/G4；
- **G5 已闭环**：VLM 内 W4 敏感度 MLP > Attention，accuracy-preserving 的 W4 机会不在 VLM 内部，而在 Expert（G1-D=84%）——指向 Phase H 选 S1=Expert-W4 作 anchor；
- **G6 已闭环**：四组 SR = 90/69/41/21，三项 L 值判据全过 → G5 结论对 Expert-FP8 背景鲁棒。**Phase G 归因链条完整闭合**（G0 误差均匀 → G1 VLM 主导 → G2 粒度非主因 → G5 VLM 内 MLP 主导 → G6 对部署背景鲁棒）。
- **未启动项**：G3（action-horizon）已降为次要；G4（outlier-mask-diagnosis）保持 conditional。二者均未跑闭环，如需可基于现有 config 直接启动。
- **跨实验延续**：Phase H 已把「Expert-W4」作为 S1 部署候选，H3 100ep 得 S0=89.0% / S1=85.0%（Δ −4pp），与该归因链条一致。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-11 | 创建文档 | |
| 2026-09-12 | 回填 G1 四 config 结果（88/21/21/84），结论：VLM 主导；G2 升为最高优先（需先实现 granularity 字段） | zyzhao |
| 2026-09-14 | 回填 G5（90/72/39/21，MLP 主导）与 G6（Gate 0-5 PASS，g6a=90%） | zyzhao |
| 2026-09-19 | **状态改 done**；G6 四组闭环回填（90/69/41/21）与 L 值判据（三项全过）；结论：G5 对 Expert-FP8 背景鲁棒，归因链条闭合 | zyzhao |
| 2026-09-15 | 新增 G1 / G5 柱状图（`scripts/plot_phaseG_bars.py`）：G1 复用 Phase F 四 suite 调色板，G5 用 Goal 绿色四档深浅 | zyzhao |
