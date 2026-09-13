# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 weight-granularity（G2）
- **状态**：done（draft / running / done / aborted）
- **最后更新**：2026-09-13

---

## 1. 摘要

<!-- 3-5 句话：做了什么、核心数字、最重要的结论 -->

G2-P0 数值 preflight 已完成（112 个 VLM Linear，per-tensor vs per-output-channel W4）。五项 PASS 条件全部通过，但改善幅度仅 5.7%。G2-B 闭环已完成（含一次前向 bug 修复重跑）：**per-channel Goal SR = 23.0%，vs per-tensor anchor（G1-C）21.0%，Δ +2pp —— 落在 Gate 1 的「≤30%：per-channel 基本失败」档**。结论：tensor-wise scale 粒度不是 F3 崩溃的主因；5.7% 的数值改善几乎不转化为闭环收益，与 G0（数值误差与闭环 SR 解耦）交叉印证。按 Gate 2（G2-B < 50%），不启动 groupwise 闭环 rollout，转向 groupwise offline audit 或 VLM selective-precision。

## 2. 总结果表

<!-- 每个 config 一行；数值带单位；与 baseline 的差值单独列 -->

| 组 | granularity | Goal SR | Δ vs G2-A(21%) | 状态 | 输出目录 |
|---|---|---:|---:|---|---|
| G2-A（复用 G1-C） | per_tensor | 21.0% | — | done/reused | `tasks/component-localization/g1c_w4_vlm_only/` |
| G2-B | per_output_channel | **23.0%** | **+2.0pp** | **done**（含 bug 修复重跑） | `tasks/weight-granularity/g2b_per_channel_vlm/` |
| G2-B (invalid) | per_output_channel | 0.0% | — | **invalid**（x_sim 未去量化 bug，归档） | `tasks/weight-granularity/g2b_per_channel_vlm.invalid_xsim_bug/` |
| G2-P0（离线） | 两者对比 | —（无 rollout） | — | done | `tasks/weight-granularity/preflight/{per_tensor,per_channel}/` |
| G2-C/D/E | groupwise | — | — | **halted by Gate 2**（仅允许 offline audit） | — |

## 3. 分组结果与分析

<!-- 按实验变量分组展开：每组建一个小节，含数据表、图、简要分析 -->

### 3.1 G2-P0 preflight（离线数值对比）

| 指标 | per_tensor | per_output_channel |
|---|---:|---:|
| mean weight_nmse | 1.144e-02 | **1.078e-02**（−5.7%） |
| nonfinite scales | 0 | 0 |
| saturation >1% 层数 | 0 | 0 |
| NMSE 改善层数 | — | 112/112（100%） |
| scale shapes | scalar | 960×48 / 320×32 / 2560×32（与 out_features 一致） |

分析：per-channel 在**每一层**都严格改善（形式 PASS），但幅度很小。结合 G0 结论（VLM nmse 0.0128 仅 ~1.3× expert，且 expert W4 闭环无损 84%），**数值 NMSE 量级本身不足以解释 21% vs 84% 的闭环差距**——G2-B 闭环是关键检验：若 5.7% 的数值改善换来 ≥15pp SR 恢复，说明 VLM 闭环处于误差悬崖边缘；若 SR 几乎不动（≤30%），则 per-channel 失败，根因转向 VLM selective-precision / activation-conditioned 方向（setup §8.8）。

### 3.2 G2-B 闭环（VLM per-channel W4，goal×100ep）

| task | G1-A fp8 | G2-A per_tensor(=G1-C) | G2-B per_channel |
|---:|---:|---:|---:|
| t0 | 100 | 100 | 90 |
| t1 | 100 | 0 | 0 |
| t2 | 90 | 0 | 0 |
| t3 | 80 | 30 | 50 |
| t4 | 100 | 0 | 0 |
| t5 | 90 | 0 | 0 |
| t6 | 70 | 10 | 0 |
| t7 | 90 | 0 | 0 |
| t8 | 100 | 50 | 80 |
| t9 | 60 | 0 | 10 |
| **AVG** | **88.0** | **21.0** | **23.0** |

分析：崩溃组（t1/t2/t4/t7）在 per-channel 下仍为 0；仅 t3（30→50）、t8（50→80）部分恢复；t6/t5 反而略降（噪声带内）。整体 Δ +2pp，远低于 Gate 1 的 15pp 显著阈值。

## 4. 结论

1. **per-channel 失败（Gate 1「≤30%」档）**：23.0% vs 21.0%（Δ +2pp < 15pp 阈值），tensor-wise scale 粒度不是 F3 崩溃主因；
2. **数值-闭环解耦再获确证**：P0 的 5.7% NMSE 改善几乎不转化为闭环收益，与 G0/G1 的「数值误差与 SR 解耦」结论一致；
3. **崩溃任务集对 scale 粒度不敏感**：t1/t2/t4/t7 在 per-tensor/per-channel 下都是 0%——这些任务失败的原因不在 weight scale 分辨率；
4. 可执行结论：W4 在 VLM 上即使配 per-channel 也不可用；按 setup §8.8，下一步进入 groupwise offline audit（仅数值，不烧 rollout）或直接转向 VLM selective-precision / 部署候选 M1（VLM FP8 + Expert W4，Expert W4 已证无害 84%）。

## 5. 问题与后续

- **已修复 bug（重要审计记录）**：G2-B 首跑 0% 系 `quant_forward_pot_ao_outlier_channel` 中 `x_sim` 未乘回 `a_interval`（`quant_awo` 返回缩放 code 而非去量化值；标量路径靠 M_aw 整数补偿在末尾统一乘回，张量路径漏掉）。坏结果归档于 `g2b_per_channel_vlm.invalid_xsim_bug/`，冒烟测试已补层输出量级检查（3b）防复发；
- Gate 2 生效：G2-B < 50%，groupwise 不进入闭环，只允许 offline numerical audit（G128/64/32 vs channel 的 NMSE 差距）；
- 后续优先级建议：① groupwise offline audit（快，无 rollout）；② VLM selective-precision（attention vs MLP、浅 vs 深）；③ 部署候选 M1 验证（VLM FP8 + Expert W4 + MatMul FP8 四 suite）；
- Phase G 父实验可考虑收敛：G0（数值均匀）+ G1（VLM 主导）+ G2（非粒度）已把根因锁定为「VLM 对 W4 量级噪声的闭环敏感度本身」，继续在 scale 粒度上投入的边际收益低。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-11 | 创建文档 | zyzhao |
| 2026-09-12 | 按修订版 setup 重建（VLM-only + 两级 Gate）；回填 G2-P0 结果（PASS，NMSE 改善 5.7%） | zyzhao |
| 2026-09-13 | 回填 G2-B：首跑 0%（x_sim bug，已修复归档）；重跑 **23.0%**（Δ+2pp，per-channel 失败）；Gate 2 拦停 groupwise 闭环 | zyzhao |
