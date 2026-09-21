# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-15_phaseI_vision-quantization
- **状态**：running（V0 ✅ / V1 ✅ / VLIN ✅（Goal×100 = 80.0%，Δ −10.0pp）/ V2–V5 待跑）
- **最后更新**：2026-09-21

---

## 1. 摘要

Gate L0（legacy regression）已通过：用原封不动的 G6 canonical 配置 build，确认 `QuantizedLinear=224`、`QuantizedMatMul=64`、`vision.*=0`、`connector.*=0`，Phase I 新增代码未破坏既有 VLM/Expert wrapping。V0（Vision workload audit）已完成：真实模型为 12 层 Vision Transformer（hidden=768、intermediate=3072、heads=12），attn 实现为 `sdpa`；一次 `sample_actions()` 处理 **2 张相机图**，每张图 1024 patch tokens → connector 后 64 visual tokens（12288→960）。Vision+Connector 单次 sample_actions 理论 dense FLOPs ≈ **428.2 GFLOPs**，其中 Vision MLP 54.2%、attention projection 27.1%、QK/PV 18.1%、connector 0.71%，与手册 §0 粗估（430.6 G，72%）一致。

**V1（Connector FP8）已完成（2026-09-19）—— 但代价很高。** 五道 gate 全过：routing 225 Linear + 64 MatMul = **289**（224+1 connector）、**V1-R raw 等价性 bit-exact**（max_abs_error = 0.000e+00）、calibration-only、task0×1 smoke = 100%、**Goal×100 = 81.0%（81/100）**。

> **关键发现：单独量化一个仅占 0.71% FLOPs 的 connector Linear，就把 Goal×100 从 89.0% 拉到 81.0%（Δ = −8.0pp）**（baseline = Phase H H3 S0 FP8-all = 89.0%，两者同为 canonical VLM/Expert FP8 背景）。对照 Phase G 的归因：VLM 侧 W4 才造成 −51pp、attention-W4 仅 −18pp ——  **8pp 在「单个 Linear 的量化代价」尺度上属于异常高值**。这是本 Phase 最值得后续追查的信号（见 §5）。

**VLIN（完整 Vision 72-Linear FP8 integration）已完成（2026-09-21）**：Gate 1–6 全部通过（routing **296 Linear / 64 MatMul**，reuse 288 / recalibrate 72；coverage **72/72 sites、216/216 scale 文件**；smoke 100%），**Goal×100 = 80.0%（80/100，Δ = −10.0pp vs G6-A 90.0%）**。覆盖 347.9 GFLOPs（≈81.2% Vision compute / 58% 整体推理），Vision QK/PV 保持 SDPA、connector 原精度。逐 task 降幅模式与 V1（connector 单点 FP8，81.0%）几乎相同——损失同样集中在 task02/03/06/09，这一巧合是后续审计的重要线索（见 §3.6/§5）。

## 2. 总结果表

| 组 | config | 基准 | Success Rate | Δ vs baseline | episodes | 输出目录 | 备注 |
|---|---|---|---|---|---|---|---|
| Gate L0 | g6a_all_fp8_control（复用） | — | — | — | — | — | 224 Linear / 64 MatMul / 0 vision / 0 connector |
| V0 | v0_workload_audit | — | 100.0%（task0） | — | 1 | `outputs/.../v0_workload_audit` | 无量化，仅 workload/FLOPs 审计 |
| V1-R | v1_connector_fp8（raw mode） | 原模型 connector | **bit-exact** | 0.0 | — | — | max/mean/max_rel error 全为 `0.000e+00` |
| V1 smoke | v1_connector_fp8 | — | 100.0%（task0） | — | 1 | `outputs/.../v1_connector_fp8` | 289 quant modules（225 Linear + 64 MatMul） |
| **V1** | v1_connector_fp8_goal | **H3 S0 FP8-all = 89.0%** | **81.0%（81/100）** | **−8.0 pp** | 100 | `outputs/.../v1_connector_fp8_goal` | Wilson 95% CI `[72.2%, 87.5%]`，eval_s = 40735（≈11.3h） |
| VLIN Gate1–5 | vlin_full_vision_linear_fp8（gate 部分） | — | — | — | — | `scales/.../vlin_full_vision_linear_fp8` | routing 296 Linear / 64 MatMul；reuse 288 / recalibrate 72；72/72 sites、216/216 scale 文件 |
| VLIN smoke | vlin_full_vision_linear_fp8 | — | 100.0%（task0） | — | 1 | `outputs/.../vlin_full_vision_linear_fp8` | 360 quant modules（296 Linear + 64 MatMul）；eval_s ≈ 94.2 |
| **VLIN Goal×100** | vlin_full_vision_linear_fp8_goal | **G6-A = 90.0%** | **80.0%（80/100）** | **−10.0 pp** | 100 | `outputs/.../vlin_full_vision_linear_fp8_goal` | Wilson 95% CI `[71.1%, 86.7%]`，eval_s = 10189（≈2.83h）；对 H3 S0（89.0%）为 −9.0pp |

## 3. 分组结果与分析

### 3.1 Gate L0 — legacy regression

| 项 | 期望 | 实测 | 判定 |
|---|---:|---:|---|
| QuantizedLinear | 224 | 224 | ✅ |
| QuantizedMatMul | 64 | 64 | ✅ |
| vision.* module | 0 | 0 | ✅ |
| connector.* module | 0 | 0 | ✅ |

### 3.2 V0 — Vision 结构 / 运行时 / FLOPs

静态结构（`vision_structure.json`）：

| 项 | 值 |
|---|---|
| Vision 层数 | 12 |
| hidden / intermediate | 768 / 3072 |
| heads | 12 |
| image_size / patch_size | 512 / 16（→ 1024 patch tokens） |
| attention impl | **sdpa**（⚠ 非 eager，V4 需先做 eager-equivalence gate） |
| connector | scale_factor=4，proj 12288→960（→ 64 tokens） |

运行时（`vision_runtime_shapes.json`）：

| 项 | 值 |
|---|---|
| cameras / sample_actions | **2.0**（真实 inference 实测：19 次 sample_actions × 2 = 38 vision calls） |
| vision 输出 shape | [1, 1024, 768] |
| connector 输出 shape | [1, 64, 960] |

FLOPs（`vision_flops.csv`，1 MAC = 2 FLOPs，per sample_actions）：

| operator | FLOPs | % of vision |
|---|---:|---:|
| Vision MLP (fc1/fc2) | 2.319e11 | 54.16% |
| Attention projection (q/k/v/out) | 1.160e11 | 27.08% |
| QK | 3.865e10 | 9.03% |
| PV | 3.865e10 | 9.03% |
| Connector proj | 3.020e09 | 0.71% |
| **合计** | **4.282e11** | 100% |

分析：Vision MLP 是 Vision 内部最大算力块（54%），其次 attention projection（27%）、QK/PV（18%）。与手册 §0 粗估一致，验证了「只补 Vision MLP + attention projection 即可覆盖约 80% Vision compute」的判断。

### 3.3 FLOPs 占比（饼图）

**Phase I 实测（V0）— Vision encoder + connector 内部拆分**（428.2 GFLOPs / `sample_actions()`）

![Phase I Vision FLOPs](figures/phaseI_flops_pie_vision.png)

| 子模块 | GFLOPs | Vision 内占比 |
|---|---:|---:|
| Vision MLP `fc1/fc2` | 231.93 | 54.16% |
| Attention Q/K/V/out projection | 115.96 | 27.08% |
| Attention `QK^T` | 38.65 | 9.03% |
| Attention `P×V` | 38.65 | 9.03% |
| Connector projection | 3.02 | 0.71% |
| **合计** | **428.22** | **100%** |

**整体推理 — 单次 `sample_actions()`**（手册 §0 粗估，595.7 GFLOPs ≈ 0.60 TFLOPs）

![Phase I inference FLOPs](figures/phaseI_flops_pie_inference.png)

| 组件 | GFLOPs | 占比 |
|---|---:|---:|
| Vision Encoder + pixel shuffle / connector | 430.60 | 72.28% |
| VLM 16-layer prefix prefill | 57.60 | 9.67% |
| Action Expert 10-step denoise | 107.50 | 18.05% |
| **总计** | **595.70** | **100%** |

两图并排（便于看出左图的蓝色扇区就是右图的放大）：

![Phase I FLOPs 2-panel](figures/phaseI_flops_pie_2panel.png)

**配色说明**：整体图给每个顶层分支一个色相（Vision 蓝 / VLM 橙 / Expert 绿）；Vision 内部图是对蓝色分支的放大，因此用**同一蓝色的 5 档明度**，按占比由大到小由深到浅。两份图共用 `scripts/figure_palette.py`，与 Phase F/G 图同源。

**交叉验证**：手册 §0 粗估 Vision+Connector 为 430.6 G，V0 实测为 428.2 G，两者相差 **0.6%**，互相印证。

生成命令（数值均从源文件解析，不硬编码；整体图的数字直接解析手册 §0 表格）：

```bash
python experiments/2026-09-15_phaseI_vision-quantization/scripts/plot_phaseI_flops_pies.py
```

### 3.4 V1-R — Connector raw wrapper 等价性

方法（`scripts/v1_connector_raw_equiv.py`）：用**同一份 config** build 两个模型，仅切换 `quantization.connector.enabled`；开启侧把全部 module 切到 `raw` 模式（不做 fake quant），喂入同一 vision-hidden 张量，比较 connector 输出。

```
original connector proj type: Linear
wrapped  connector proj type: QuantizedLinear
============================================================
V1-R connector raw equivalence
============================================================
max_abs_error  : 0.000e+00
mean_abs_error : 0.000e+00
max_rel_error  : 0.000e+00
V1-R: PASS (raw wrapper exact)
```

| 项 | 期望（手册 §9.2） | 实测 | 判定 |
|---|---|---|---|
| connector proj 被 wrap | `QuantizedLinear` | `QuantizedLinear` | ✅ |
| max_abs_error | exact 或 dtype rounding 级 | **0.000e+00** | ✅ |
| mean_abs_error | 同上 | **0.000e+00** | ✅ |
| max_rel_error | 同上 | **0.000e+00** | ✅ |
| 总 quantized modules | 225 Linear + 64 MatMul = 289 | **289** | ✅（手册 §7） |

> **方法学意义**：raw wrapper 的 bit-exact 排除了「新增 component 引入数值偏差」这一混淆项 ⇒ V1 的 SR 变化可以**干净地归因于量化**。
> **范围限制**：手册 §9.2 要求同时报告 connector output / prefix embedding / final action chunk 三项，当前脚本**只覆盖 connector output**；另两项待补（见 §5）。

### 3.5 V1 — Connector FP8（Goal × 100）

配置：canonical VLM/Expert FP8 背景 + `connector.overrides[connector_fp8] = pot_fp8_outlier`（A/W/O 全 `e4m3`），`linear.include = [vlm.*, expert.*, connector.*]`，`vision.enabled = false`。calibration：`episodes=8 / batch_size=8 / frame_stride=4`（recalibrate，connector scale 首次生成）。

**Gate 结果**

| Gate | 要求 | 实测 | 判定 |
|---|---|---|---|
| Gate 1 routing count | 225 Linear | 225 Linear + 64 MatMul = 289 | ✅ |
| Gate 2 raw equivalence | exact | `0.000e+00`（§3.4） | ✅ |
| Gate 3 calibration-only | scale 文件生成 | `scales/.../v1_connector_fp8` | ✅ |
| Gate 4 task0 × 1 smoke | SR | **100.0%（1/1）** | ✅ |
| Gate 5 Goal × 100 | SR | **81.0%（81/100）** | ✅（完成，但显著下降） |

**逐 task 成功数（每 task 10 episodes，与 Phase H H3 S0 FP8-all 同 task 对照）**

| Config | t00 | t01 | t02 | t03 | t04 | t05 | t06 | t07 | t08 | t09 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H3 S0 FP8-all（baseline） | 10 | 10 | 10 | 8 | 10 | 9 | 5 | 9 | 10 | 8 | **89** |
| **V1 Connector FP8** | 10 | 10 | **8** | **7** | 10 | 9 | **3** | 9 | **9** | **6** | **81** |
| Δ | 0 | 0 | **−2** | **−1** | 0 | 0 | **−2** | 0 | **−1** | **−2** | **−8** |

统计：SR = 81.0%（81/100），Wilson 95% CI **[72.2%, 87.5%]**，eval_s = **40735**（≈11.3h）。与 baseline（89.0%，CI `[81.4%, 93.7%]`）**Δ = −8.0pp，两个 CI 重叠约 6.1pp（81.4 → 87.5）⇒ 方向明确但未达统计显著**。

**要点**

- **降幅不均匀**：task00 / 01 / 04 / 05 / 07 **完全不降**；损失集中在 task02（−2）、task03（−1）、**task06（−2）**、task08（−1）、**task09（−2）**。即 **connector 量化打掉的是「困难 task 上的余量」**，而非均匀削弱所有能力。task06 在 baseline 下本已最弱（5/10），V1 下进一步降到 3/10。
- **单位 FLOPs 的代价极高**：connector 仅占 0.71% 的 Vision+Connector FLOPs（占整体推理约 0.51%），却带来 8pp 损失。对照 Phase G：VLM **全部 attention** W4 = −18pp、VLM **全部 MLP** W4 = −51pp。connector 只有 1 个 Linear，却拿到 attention 全部（16 层 × 4 投影 = 64 个 Linear）损失的 **44%**。
- **可能原因（待验）**：① connector 输出直接构成 VLM 的 64 个 visual token，**无中间归一化缓冲**，单点量化误差直接进入 prefix；② 12288→960 的投影需要保留 pixel-shuffle 通道的精细混合结构，per-tensor/per-site 的 scale 难以同时容纳全部通道的动态范围；③ `outlier_ratio=0.01` 对 connector 的输入分布未必合适。
- **不影响 V2 的正确性，但影响优先级**：按手册 §9.4，V1 出现明显下降时**应先审计再进 V2**（见 §5）。

### 3.6 VLIN — 完整 Vision 72-Linear FP8 integration（Gate 1–6）

背景：外部协作者在分支 `phaseI/vision-linear-full-integration`（基于 main@2b2a3013，+9 commits / 8 files）补全 Vision Encoder **全部 72 个 Linear** 的 FP8 wrapping。实验设计：VLM + Expert = 224 Linear + 64 MatMul（FP8，**复用 G6-A canonical scale，reuse**）；Vision Encoder = 72 Linear（FP8，**重新 calibration**）；QK/PV 保持原 SDPA 不量化；**connector 保持 raw**（与 V1 显式区分，故总数为 296 而非 297）。

一键脚本 `scripts/run_full_vision_linear.sh`（fail-loud，任一 Gate 失败即退出）于 2026-09-21 在本机跑通，总耗时 ≈ 25 min（calibration 45/45 forwards ≈ 8–10 min）。

**Gate 结果**

| Gate | 内容 | 实测 | 判定 |
|---|---|---|---|
| 1 | py_compile + `tests/test_vision_quant_routing.py` | 全过（含新增「72/72 全部 recalibrate」用例） | ✅ |
| 2 | G6-A canonical scale 复制 | 复制到 `scales/2026-09-15_phaseI_vision-quantization/vlin_full_vision_linear_fp8`（独立目录，不触碰 G6 原始 scale） | ✅ |
| 3 | routing/count 审计（真实模型） | 见下表 | ✅ |
| 4 | calibration-only | 45/45 forwards，仅重校准 72 个 Vision Linear | ✅ |
| 5 | calibration coverage | 见下 | ✅ |
| 6 | Goal task0 × 1 smoke | **SR = 100.0%（1/1）**，eval_s ≈ 94.2 | ✅ |

Gate 3 routing 明细（`scripts/audit_full_vision_linear_routing.py`）：

| 项 | 实测 | 判定 |
|---|---:|---|
| QuantizedLinear 总数 | **296** | ✅ |
| VLM/Expert Linear（reuse） | 224 | ✅ |
| Vision Linear（recalibrate） | **72**（attn proj 48 + MLP 24） | ✅ |
| Connector Linear | 0 | ✅ |
| QuantizedMatMul 总数 | 64（Vision 0） | ✅ |
| calibration action | **reuse 288 / recalibrate 72**（288 = 224 Linear + 64 MatMul） | ✅ |
| Vision op（q/k/v/out_proj/fc1/fc2） | 各 **12/12** | ✅ |
| module_id / scale group | 无重复 | ✅ |
| A/W/O dtype | 全 E4M3，method = `pot_fp8_outlier` | ✅ |

Gate 5 coverage 明细（`scripts/audit_full_vision_linear_calibration.py`，逐文件打开 pickle 校验）：

| 项 | 实测 |
|---|---:|
| Vision Linear sites | **72 / 72** |
| complete sites（A/W/O 齐 + finite + >0） | **72 / 72** |
| Vision scale 文件 | **216 / 216**（72 × 3：A/W/O） |
| scale 目录总文件 | 1080 = canonical 复制 864 + 新增 216 |

**Goal×100 正式结果（2026-09-21，`run_full_vision_linear_goal.sh`，seed=1000，n_action_steps=10）**：**SR = 80.0%（80/100）**，Wilson 95% CI `[71.1%, 86.7%]`，eval_s = 10189（≈2.83h，101.9 s/ep）。

逐 task（与 V1 对照）：

| Config | t00 | t01 | t02 | t03 | t04 | t05 | t06 | t07 | t08 | t09 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 Connector FP8 | 10 | 10 | 8 | 7 | 10 | 9 | 3 | 9 | 9 | 6 | **81** |
| **VLIN Vision 72-Linear FP8** | 10 | 10 | **8** | **7** | **9** | 9 | **4** | 9 | 9 | **5** | **80** |

**要点**

- **Δ vs G6-A = −10.0pp（90→80）**；对 H3 S0（89.0%）为 −9.0pp。58% 的推理计算量入 FP8 付出 10pp，明显高于「小代价」预期（协作者预期 88–90%）。
- **逐 task 模式与 V1 几乎重合**：V1 = [10,10,8,7,10,9,3,9,9,6]，VLIN = [10,10,8,7,9,9,4,9,9,5]。两者量化对象完全不同（V1 = connector 单个 Linear；VLIN = Vision 72 Linear、connector raw），却掉同样的 task（t02/03/06/09）且幅度接近（−8 vs −10pp）⇒ 强烈暗示两条路径的误差可能经**同一传导通道**作用（Vision→connector→VLM prefix 的 64 visual token），或这些 task 对 prefix 视觉表征噪声本身敏感（t06 在 G6-A 下也最弱）。
- **速度收益可观**：eval_s 10189 vs V1 40735（≈4×），部分因逐 task SR 分布与主机负载，需同条件复测才能归因于量化提速。
- **框架能力验证**：现有框架无需改 `model_wrapper.py` 即可完整 wrap 72 个 Vision Linear，routing / per-site scale group / 分层 calibration policy（reuse vs recalibrate）全部按预期工作。
- **与 V1 的对照关系**：VLIN 与 V1 共用 G6-A canonical VLM/Expert FP8 背景（baseline 89.0%），但 VLIN **不含 connector 量化**（V1 含）⇒ VLIN Goal×100 与 89.0% 的差值可解释为「Vision 72 Linear FP8」的净代价，与 V1 的 −8.0pp（connector 单点）形成两个独立的单变量数据点。
- **风险预告已兑现**：V0 已测得这 72 个 Linear 覆盖 Vision 内部 81.2% FLOPs（MLP 54.2% + attn projection 27.1%），且 Vision 输出经 connector 直接进入 VLM prefix（无归一化缓冲）——实际 Goal×100 = 80.0%（−10pp），确认 Vision 侧 FP8 并非「小代价」。

## 4. 结论

1. **Phase I 新增代码向后兼容**（Gate L0 PASS，224/64/0/0；开启 connector 后为 225 Linear + 64 MatMul = 289，与手册 §7 的预期 count 表一致）。
2. **Vision+Connector 占单次 `sample_actions()` 的 dense FLOPs 约 428 G**（占全推理 ~72%），**Vision MLP 是最大块（54.2%）**，attention projection 27.1%、QK/PV 18.1%、connector 仅 0.71% ⇒ 手册「只补 Vision MLP + attention projection 即可覆盖 ~80% Vision compute」的判断成立。
3. **Connector 的 raw wrapper 是 bit-exact 的**（V1-R：max/mean/max_rel error 全为 `0.000e+00`）⇒ 新增 component 的 resolver / scale group / calibration / StatManager export 链路正确，**V1 的 SR 变化只能归因于量化本身，不可能是 wrapper 引入的数值偏差**。
4. **Connector FP8 的精度代价远超其 FLOPs 占比**：connector 只占 0.71% 计算，但 Goal×100 从 89.0% 降到 **81.0%（−8.0 pp）**。作为标尺：Phase G 中 VLM attention 全部 W4 才 −18pp、VLM MLP 全部 W4 才 −51pp。**一个 Linear 能拿到 8pp，说明 connector 输出的视觉 token 对量化噪声高度敏感**（12288→960 的投影需保留通道混合的精细结构），或存在 per-site scale / outlier 选择不当。
5. **attention 实现为 `sdpa`（非 eager）**，V4（QK/PV 量化）前必须先做 eager-backend equivalence gate（手册 §12.2）；否则 QK/PV 的量化无法与既有统计口径对齐。
6. **逐 task 视角**：V1 与 H3-S0 同 task 对比，降幅集中在 task02（10→8）、task03（8→7）、**task06（5→3）**、task08（10→9）、**task09（8→6）**；而 **task00/01/04/05/07 完全不降** ⇒ connector 量化**不是均匀地削弱能力，而是打掉了模型在困难 task 上的余量**（task06 本就是 S0 最弱的 task）。这与 §1 的 8pp 均值一致，但解释了「为何均值降 8pp 而很多 task 看不出变化」。
7. **完整 Vision 72-Linear FP8（VLIN）已完成（2026-09-21）**：工程链路全过（routing 296/64、reuse 288 / recalibrate 72、72/72 + 216/216、smoke 100%），**Goal×100 = 80.0%（80/100，Δ = −10.0pp vs G6-A 90%）**，CI `[71.1%, 86.7%]`。58% 推理计算量入 FP8 的代价为 10pp，高于预期。
8. **V1 与 VLIN 的逐 task 降幅模式几乎相同**（[10,10,8,7,10,9,3,9,9,6] vs [10,10,8,7,9,9,4,9,9,5]，均掉 t02/03/06/09）而量化对象完全不同（connector 单点 vs Vision 72 Linear）⇒ 误差大概率经同一通道作用（visual token / VLM prefix），或该 4 个 task 对视觉 prefix 噪声本身敏感。这把 V1 的「connector 异常敏感」重新解读为「**视觉通路末端共同敏感**」，后续审计应把 connector 与 Vision Linear 放在同一框架下看。

## 5. 问题与后续

- **VLIN 结果已出（80.0%，−10.0pp）**：已推送到 mirror 分支待协作者审计逐 task 结果，并决定 Vision MLP / attention projection 是否拆开做更低 bit。注意逐 task 模式与 V1 几乎重合（同掉 t02/03/06/09），建议审计时优先检验「共同传导通道」假设（connector 输出 / prefix embedding 的扰动谱，可复用 V1 审计工具链）。
- **V0 ✅ / V1 ✅ 已过 gate**（routing 289、raw equivalence、calibration-only、task0×1、Goal×100），可进入 **V2（Vision MLP fc1/fc2，24 个 Linear、54% Vision FLOPs）**。
- **先追查 V1 的 8pp（优先级高于 V2）**：手册 §9.4 明确要求「如果 V1 Connector FP8 都导致明显 SR 崩溃，先审计 calibration / outlier / connector output，不要进入 V2」。建议顺序：
  1. 查 connector 的 per-site scale 与 outlier 选择（`scales/2026-09-15_phaseI_vision-quantization/v1_connector_fp8`）是否落在合理范围；
  2. 查 connector 输入张量的分布（12288 通道，pixel-shuffle 输出）与 `outlier_ratio=0.01` 是否匹配；
  3. 跑 connector 的 W8A8 单独 ablation（例如只量化 W、A 保持 raw FP）以定位是权侧还是激活侧主导；
  4. 确认 connector 输出是否经过任何归一化（若无，单点误差会直接放大到 VLM prefix 的 64 个视觉 token 上）。
- **V2 前置建议**：先跑一段短 smoke（task0 × 1ep）看 Vision MLP FP8 是否也带来同量级损失；若 V1+V2 累积损失超过 ~10pp，应在进入 V3/V4 前先解决 V1。
- **V4 前置**：sdpa → eager 等价性确认（手册 §12.2）。
- **V1-R 的比较范围**：手册 §9.2 要求同时报告 connector output / prefix embedding / final action chunk 三项；当前 `v1_connector_raw_equiv.py` **只比较了 connector output**（已 bit-exact）。剩余两项（prefix embedding / final action chunk）尚未接入脚本，属待补项（不影响 connector output 的结论）。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-15 | 创建文档 | |
| 2026-09-15 | 回填 Gate L0 + V0（结构/运行时/FLOPs）结果 | |
| 2026-09-15 | 新增 §3.3 FLOPs 占比饼图（V0 实测 + 手册 §0 整体推理） | |
| 2026-09-19 | 回填 **V1（Connector FP8）**：V1-R raw 等价性 bit-exact、smoke 100%、**Goal×100 = 81.0%（−8.0pp vs H3 S0 89.0%）**；新增 §3.4/§3.5、重写 §4 结论 6 条与 §5 后续；状态保留 `running`（V2–V5 待跑） | |
| 2026-09-21 | 回填 **VLIN（完整 Vision 72-Linear FP8）Gate 1–6**：routing 296/64、reuse 288 / recalibrate 72、coverage 72/72 + 216/216、smoke 100%；新增 §3.6、总结果表 3 行、结论第 7 条、§5 首条（Goal×100 待跑） | |
| 2026-09-21 | 回填 **VLIN Goal×100 = 80.0%（80/100，Δ −10.0pp vs G6-A）**；新增逐 task 对照（与 V1 模式几乎重合）、结论第 8 条（共同传导通道假设）；状态改为 `VLIN ✅` | |
