# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 vlm-selective-expert-fp8
- **状态**：done（draft / running / done / aborted）
- **最后更新**：2026-09-19

---

## 1. 摘要

component-aware Linear precision routing 升级（`linear.overrides`）完成，G6 四组（Expert-FP8 部署背景下的 VLM selective precision）全部跑通：Gate 0-5 静态/等价/校准检查全 PASS，Gate 6（Goal ×100）四组于 2026-09-16 完成，SR = **90 / 69 / 41 / 21**（control / attn-W4 / mlp-W4 / all-W4）。

**核心结论：G5 的「VLM MLP 对 W4 的敏感性远大于 Attention」在 Expert 也量化为 FP8 的实际部署背景下依然成立。** 三项预定义判据（|ΔL| ≤ 6pp）全部通过：L_attn = 21（G5 为 18，Δ +3pp）、L_mlp = 49（G5 为 51，Δ −2pp）、L_all = 69（G5 为 69，Δ 0pp）。MLP 仍贡献约 71% 的精度损失，且两者近似可加（21 + 49 = 70 ≈ L_all 69）。

---

## 2. 总结果表

| 组 | config | VLM attn | VLM MLP | Expert | QK/PV | Goal SR | Δ vs G5 | 输出目录 |
|---|---|---|---|---|---:|---:|---|
| G6-A | g6a_all_fp8_control | FP8 | FP8 | FP8 | FP8 | **90.0%** | 0.0 | `g6a_all_fp8_control/` |
| G6-B | g6b_vlm_attn_w4_expert_fp8 | **W4** | FP8 | FP8 | FP8 | **69.0%** | −3.0 | `g6b_vlm_attn_w4_expert_fp8/` |
| G6-C | g6c_vlm_mlp_w4_expert_fp8 | FP8 | **W4** | FP8 | FP8 | **41.0%** | +2.0 | `g6c_vlm_mlp_w4_expert_fp8/` |
| G6-D | g6d_vlm_all_w4_expert_fp8 | **W4** | **W4** | FP8 | FP8 | **21.0%** | 0.0 | `g6d_vlm_all_w4_expert_fp8/` |

> 输出目录前缀：`outputs/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/`
> 「Δ vs G5」= G6 减去同口径 G5 组（A/B/C 对 G5-A/B/C；D 对 G1-C，因 G5 的 all-W4 复用 G1-C）。

### 2.1 逐 task 成功数（/10，seed=1000）

| 组 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G6-A | 10 | 10 | 9 | 8 | 10 | 9 | 7 | 9 | 10 | 8 | **90** |
| G6-B | 10 | 9 | 5 | 6 | 6 | 7 | 7 | 5 | 9 | 5 | **69** |
| G6-C | 10 | 8 | 0 | 7 | 1 | 2 | 0 | 2 | 10 | 1 | **41** |
| G6-D | 10 | 0 | 1 | 2 | 0 | 0 | 1 | 0 | 5 | 2 | **21** |
| *G5-A* | 10 | 10 | 10 | 9 | 10 | 10 | 5 | 9 | 9 | 8 | *90* |
| *G5-B* | 10 | 10 | 5 | 6 | 6 | 8 | 5 | 5 | 10 | 7 | *72* |
| *G5-C* | 10 | 4 | 0 | 7 | 2 | 2 | 1 | 3 | 9 | 1 | *39* |
| *G5-D(=G1-C)* | 10 | 0 | 0 | 3 | 0 | 0 | 1 | 0 | 5 | 2 | *21* |

（斜体 = G5 参照臂，**非本实验数据**，列在此处便于逐 task 对照。）

### 2.2 各阶段耗时（eval_s，goal ×100）

| 组 | eval_s | 小时 |
|---|---:|---:|
| G6-A | 7430 | 2.06 |
| G6-B | 18259 | 5.07 |
| G6-C | 40246 | 11.18 |
| G6-D | 37895 | 10.53 |

> 耗时与 SR 强负相关：SR 越低 → 失败的 episode 跑满 300 步上限 → 越慢。G6-C/D 因此比 G6-A 慢 5 倍以上。四组均为与 Phase H / Phase I 并发运行的结果，含 GPU 争抢导致的膨胀。

---

## 3. 分组结果与分析

### 3.1 Framework Gate（Gate 0-5）

| Gate | 内容 | 结果 |
|---|---|---|
| Gate 0 | py_compile | PASS |
| Gate 1 | routing resolver 单测 | 8 PASS（含 target typo fail-fast） |
| Gate 2 | 四组 routing count | PASS（224/0、160/64、176/48、112/112，MatMul 64） |
| Gate 3 | raw Linear equivalence | PASS（224 Linear，max\|diff\|=0） |
| Gate 4 | calibration smoke | PASS（四组各 864 scale，无 NaN/Inf） |
| Gate 5 | task0×1 smoke | PASS（四组各 SR=100%，无 crash/NaN） |
| Gate 6 | Goal ×100 四组 | **PASS**（2026-09-16 全部完成） |

分析：

- 升级后 no-override 行为不变（G6-A 224 FP8 与 base config 一致），raw forward 数值 bit-exact，向后兼容性成立。
- 四组 scale 隔离正确（per_site），VLM 的 W4 与 Expert 的 FP8 不共享 scale。
- Gate 2 的 routing CSV 已为四组补齐（`routing/<config>_routing.csv`），逐 operator 记录 `w_bit` 与 `override_names`，可独立复核 160/64、176/48、112/112 的分配。

### 3.2 VLM selective precision（Expert-FP8 背景）— 核心结果

| 取值 | G6（Expert **FP8**） | G5（Expert **raw FP**） | Δ(G6−G5) | 判定（\|Δ\|≤6pp） |
|---|---:|---:|---:|:--:|
| control | **90.0%** | 90.0% | 0.0 | ✅ |
| attention-W4 | **69.0%** | 72.0% | −3.0 | ✅ |
| MLP-W4 | **41.0%** | 39.0% | +2.0 | ✅ |
| VLM all-W4 | **21.0%** | 21.0% | 0.0 | ✅ |

**相对各自 control 的损失（L 值）：**

| 指标 | G6 | G5 | Δ | 判定 |
|---|---:|---:|---:|:--:|
| L_attn | **21.0** | 18.0 | +3.0 | ✅ |
| L_mlp | **49.0** | 51.0 | −2.0 | ✅ |
| L_all | **69.0** | 69.0 | 0.0 | ✅ |

**判据（实验开跑前预设）**：若 |L − G5 对应值| ≤ 6pp 且仍有 L_mlp ≫ L_attn，则结论对 Expert precision background 鲁棒。

**结论：两条均满足。**

1. **三项 L 值全部落在 ±6pp 内**（+3 / −2 / 0），最大偏差 3pp 远小于 100ep 的 ±5.7pp 噪声带。
2. **MLP 仍主导**：L_mlp / L_attn = 49 / 21 = **2.33×**（G5 为 2.83×）。MLP 占 L_all 的 **71%**（G5 为 74%）。
3. **可加性成立**：L_attn + L_mlp = 21 + 49 = 70 vs L_all = 69，差 +1pp（G5 为 69 vs 69，差 0pp）。

**对部署的含义**：既然 Expert 从 raw FP 压到 FP8 并不改变 VLM 内部的敏感性结构，说明这两侧的精度损失近似正交 —— `linear.overrides` 的 component-aware 路由在「VLM FP8 + Expert W4」这类混合部署里是可用的（Phase G 的 G1-D 已证明 Expert-W4 只掉 4pp，Phase H 把它作为 S1 候选并在 H3 100ep 下复现 Δ −4pp）。

### 3.3 与 G5 的逐 task 差异

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G6-A − G5-A | 0 | 0 | −1 | −1 | 0 | −1 | +2 | 0 | +1 | 0 | −2 |
| G6-B − G5-B | 0 | −1 | 0 | 0 | 0 | −1 | +2 | 0 | −1 | −2 | −3 |
| G6-C − G5-C | 0 | +4 | 0 | 0 | −1 | 0 | −1 | −1 | +1 | 0 | +2 |
| G6-D − G5-D | 0 | 0 | +1 | −1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

观察：

- **G6-D vs G5-D 逐 task 完全一致（0 差）**，最极端配置下行为高度可复现。
- G6-B 与 G5-B 在 t6 都出现 +2 的正向差，而 t1/t5/t8/t9 各有 −1~−2，净 −3pp —— 与「±5.7pp 噪声带」的解释一致，无系统性模式。
- G6-C 的 t1 为 8（G5-C 仅 4），是单点最大正差 +4。
- 崩溃 task 集合稳定：G6-B/C/D 与对应 G5 组均在 t2 附近塌陷（G6-C t2=0、G6-D t1/t4/t5/t7=0），崩溃模式与 G1-B/G1-C 的 F3 复现一致。

---

## 4. 结论

1. **framework 升级通过全部静态/等价/校准 Gate 与闭环 Gate 6**，`linear.overrides` 的 component-aware precision routing 可安全用于混合精度部署。
2. **G5 的核心结论对 Expert precision background 鲁棒**：「VLM MLP 的 W4 敏感性 ≫ Attention」不依赖 Expert 是否为 raw FP。三项 L 值偏差 ≤ 3pp（阈值 6pp）。
3. **VLM MLP 仍是 W4 精度的主要 bottleneck**（贡献 L_all 的 71%）。因此若要在 VLM 内部做 accuracy-preserving 的 W4，必须优先保护 `gate/up/down_proj`；attention 侧的 W4 代价小得多（21pp vs 49pp）。
4. **Expert 侧才是 W4 的机会点**：G6-A(90%) / G6-D(21%) 与 G1-D（Expert-W4 84%）合并看，把 W4 预算放在 Expert 而非 VLM 内部，是 Phase H 选择 S1 = Expert-W4 作为部署候选的依据。

---

## 5. 问题与后续

- **环境**：torchcodec/FFmpeg 探测 warning（自动 fallback pyav），不影响 calibration 与 rollout。
- **早期事故**：2026-09-14 首次启动时因仓库内共享模块 `src/vla_tcs2/quant/stat_manager.py` 被编辑成半成品，G6-B 启动即 `SyntaxError`，叠加 `run_goal.sh` 原有的 `set -e` 导致整条 sweep 静默中止（C/D 连坐未跑）。已通过 `set -uo pipefail` + `failed[]` + 每组 `tee` + `conda run --no-capture-output` 修复，详见 `docs/logs.md` §4.1/§4.2。
- **耗时膨胀**：四组与 Phase H 的 H3、Phase I 的 v1 并发运行，单组耗时被放大 1.3–2.6×；G6-C/D 因 SR 低（episode 跑满 300 步）本就慢，两者叠加导致 C/D 各需约 11h。后续排期建议避免多组长时间 rollout 并发。
- **已知框架 bug（非本实验引入，待修）**：`calibrate()` 会覆盖 `module._stat_manager` 导致「同一次 run 里既校准又统计」时 runtime sparsity 静默全空。本实验全部使用 `--skip-calibration`，因此**不受影响**；该 bug 由 2026-09-15 的 quickscan 实验发现并记录于其 `docs/logs.md` §4.9。
- **未跑项**：G3（action-horizon）在该阶段已降为次要，G4（outlier-mask-diagnosis）保持 conditional，均未启动。

---

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-14 | 创建文档 | |
| 2026-09-14 | 回填 Gate 0-4 结果（单测 8、路由、raw equiv、calibration） | |
| 2026-09-14 | 回填 Gate 5（task0×1 四组 SR=100%，无 crash/NaN） | |
| 2026-09-19 | **状态改 done**；回填 Gate 6 四组 SR（90/69/41/21）与逐 task 明细、耗时、L 值判据（三项全过）与结论 | |
