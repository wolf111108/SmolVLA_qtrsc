# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 vlm-selective-expert-fp8
- **状态**：running（draft / running / done / aborted）
- **最后更新**：2026-09-14

---

## 1. 摘要

component-aware Linear precision routing 升级（`linear.overrides`）已完成，G6 四组配置（Expert-FP8 背景下的 VLM selective precision）已生成。框架 Gate 0-4 全部通过：routing 单测 8 个 PASS、四组路由 count 符合预期（G6-A 224/0、B 160/64、C 176/48、D 112/112，MatMul 64）、224 Linear raw 模式与 F.linear bit-exact（max|diff|=0）、四组 calibration smoke 各产出 864 scale 文件且无 NaN/Inf。尚未跑 Gate 5（task0×1）与 Gate 6（Goal×100），无闭环 SR 结果。

## 2. 总结果表

| 组 | config | 基准 | Success Rate | Δ vs baseline | episodes | 输出目录 | 备注 |
|---|---|---|---|---|---|---|---|
| G6-A | g6a_all_fp8_control | G5-A 90% | 待跑 | — | 100 | vlm-selective-expert-fp8 | calibration 完成 |
| G6-B | g6b_vlm_attn_w4_expert_fp8 | G5-B 72% | 待跑 | — | 100 | vlm-selective-expert-fp8 | calibration 完成 |
| G6-C | g6c_vlm_mlp_w4_expert_fp8 | G5-C 39% | 待跑 | — | 100 | vlm-selective-expert-fp8 | calibration 完成 |
| G6-D | g6d_vlm_all_w4_expert_fp8 | G5-D 21% | 待跑 | — | 100 | vlm-selective-expert-fp8 | calibration 完成 |

## 3. 分组结果与分析

<!-- 按实验变量分组展开：每组建一个小节，含数据表、图、简要分析 -->

### 3.1 Framework Gate（Gate 0-4）

| Gate | 内容 | 结果 |
|---|---|---|
| Gate 0 | py_compile | PASS |
| Gate 1 | routing resolver 单测 | 8 PASS（含 target typo fail-fast） |
| Gate 2 | 四组 routing count | PASS（224/0、160/64、176/48、112/112，MatMul 64） |
| Gate 3 | raw Linear equivalence | PASS（224 Linear，max|diff|=0） |
| Gate 4 | calibration smoke | PASS（四组各 864 scale，无 NaN/Inf） |
| Gate 5 | task0×1 smoke | PASS（四组各 SR=100%，无 crash/NaN） |

分析：

- 升级后 no-override 行为不变（G6-A 224 FP8 与 base config 一致），raw forward 数值 bit-exact，向后兼容性成立。
- 四组 scale 隔离正确（per_site），VLM W4 与 Expert FP8 不会共享 scale。
- 尚未进入闭环 SR 阶段，precision routing 的 accuracy 影响待 Gate 5/6 验证。

### 3.2 VLM selective precision（Expert-FP8 背景）

待 Gate 6（Goal×100）完成后回填 SR，对照 G5（90/72/39/21）计算 L_attn/L_mlp/L_all。

## 4. 结论

- framework 升级本身通过全部静态/等价/校准 Gate，可安全进入 rollout。
- 尚无闭环 SR 结论，G5「MLP > Attention 敏感性对 Expert-FP8 背景是否鲁棒」待 Gate 6 回答。

## 5. 问题与后续

- 环境存在 torchcodec/FFmpeg 探测 warning（自动 fallback pyav），不影响 calibration，但需在 rollout 日志中留意。
- 下一步：Gate 5（task0×1 四组 smoke）→ Gate 6（Goal×100）。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-14 | 创建文档 | |
| 2026-09-14 | 回填 Gate 0-4 结果（单测 8、路由、raw equiv、calibration） | |
| 2026-09-14 | 回填 Gate 5（task0×1 四组 SR=100%，无 crash/NaN） | |
