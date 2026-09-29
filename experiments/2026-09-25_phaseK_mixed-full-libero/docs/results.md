# 结果记录（Results）

> 本文档在实验**过程中与结束后**持续更新。章节为固定结构，不可删除；无内容写「无」。
> 图片放 `docs/figures/`，文中用相对路径 `![](figures/xxx.png)` 引用。

- **实验名称**：2026-09-25_phaseK_mixed-full-libero
- **状态**：running（draft / running / done / aborted）
- **最后更新**：2026-09-29

---

## 1. 摘要

沿用前序混合精度配置（Vision+VLM Linear **WINT8** / Expert Linear **WINT4**；三组 Linear 的 A/O 与全部 88 个 QK/PV 的 A/B/O 为 **FP8 E4M3 PoT**；`outlier_ratio=0.01`，W `per_tensor`、scale `per_site`），在标准 LIBERO 多 suite 上评测量化主路径的成功率、runtime/static 稀疏度与 dense-equivalent 计算量。

截至 2026-09-29 已完成 **Spatial** 与 **Object** 两个 suite（各 100 baseline + 100 quant episodes）：

- **Spatial**：B0 raw **86/100 (86.0%)** → M0 mixed-PTQ **81/100 (81.0%)**，Δ = **−5.0 pp**（退化集中在 task 1/2/5）。
- **Object**：B0 raw **89/100 (89.0%)** → M0 mixed-PTQ **94/100 (94.0%)**，Δ = **+5.0 pp**（反向，全部来自逐 episode 翻转）。
- **两 suite 合计 200 episodes：B0 = M0 = 175/200 = 87.5%，Δ = 0.0 pp。**

两个 suite 方向相反、量级相同，且在合计口径下完全抵消（0.0 pp），**不支持「该配置造成系统性掉点」的结论**；每 task 仅 10 episodes、单 seed，1 次翻转即 10 pp，差异落在采样噪声带内。

**Goal 按决策跳过**（已由同配置单 suite 实验覆盖，SR 87→83，见 §5）；**Long 的 baseline 不重跑**，改用历史同协议审计数据 69.0 作参照（见 §5 风险说明）；**Long quant 正在运行**（2026-09-29 15:21 启动）。

<!-- 3-5 句话：做了什么、核心数字、最重要的结论 -->

## 2. 总结果表

<!-- 每个 config 一行；数值带单位；与 baseline 的差值单独列 -->

| Suite | B0 SR | M0 SR | Δ pp | B0/M0 episodes | 输出目录 |
|---|---:|---:|---:|---|---|
| Spatial | **86.0** | **81.0** | **−5.0** | 100/100（完成） | [原始输出](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/libero_spatial/) |
| Object | **89.0** | **94.0** | **+5.0** | 100/100（完成） | [原始输出](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/libero_object/) |
| Goal | —（跳过） | —（跳过） | — | 0/0 | [不适用](../../../../outputs/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/) |
| Long | 69.0（历史参照，非本轮） | 运行中 | 待定 | 0/100 | [quant 原始输出](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/libero_10/) |
| **Spatial+Object 合计** | **87.5**（175/200） | **87.5**（175/200） | **0.0** | 200/200 | [汇总目录](../../../../outputs/2026-09-25_phaseK_mixed-full-libero/) |

> 注 1：合计行是**人工按原始 `result.json` 累加**得出（本实验按决策 D 不执行 `summarize.py`，故无 `success_summary.csv`）。
> 注 2：Goal 行的「不适用」链接指向已完成的单 suite 前置实验，仅作参照，**不是本轮产出**。
> 注 3：Long 的 69.0 来自 2026-08-27 审计运行，**未钉 checkpoint 版本，非严格同协议**，不可与 M0 直接相减作为严格结论（见 §5）。

## 3. 分组结果与分析

<!-- 按实验变量分组展开：每组建一个小节，含数据表、图、简要分析 -->

### 3.1 逐 task 对照与成/败翻转

**Spatial（`libero_spatial`）**

| task_id | B0 /10 | M0 /10 | Δ | succ→fail | fail→succ |
|---|---:|---:|---:|---:|---:|
| 0 | 10 | 10 | 0 | 0 | 0 |
| 1 | 10 | 8 | **−2** | 2 | 0 |
| 2 | 9 | 8 | **−1** | 1 | 0 |
| 3 | 10 | 10 | 0 | 0 | 0 |
| 4 | 5 | 5 | 0 | 1 | 1 |
| 5 | 9 | 7 | **−2** | 2 | 0 |
| 6 | 9 | 9 | 0 | 1 | 1 |
| 7 | 8 | 8 | 0 | 1 | 1 |
| 8 | 7 | 7 | 0 | 1 | 1 |
| 9 | 9 | 9 | 0 | 1 | 1 |
| **合计** | **86** | **81** | **−5** | **10** | **5** |

逐 episode 完全一致的 task 仅 **2/10**（task 0、3）；task 4 / 6 / 7 / 8 / 9 均出现**一次双向翻转**（净 0），说明「SR 相同」不等于「逐 episode 相同」。

**Object（`libero_object`）**

| task_id | B0 /10 | M0 /10 | Δ | succ→fail | fail→succ |
|---|---:|---:|---:|---:|---:|
| 0 | 10 | 9 | −1 | 1 | 0 |
| 1 | 10 | 10 | 0 | 0 | 0 |
| 2 | 9 | 10 | +1 | 0 | 1 |
| 3 | 7 | 9 | **+2** | 0 | 2 |
| 4 | 9 | 10 | +1 | 0 | 1 |
| 5 | 6 | 7 | +1 | 0 | 1 |
| 6 | 10 | 10 | 0 | 0 | 0 |
| 7 | 9 | 10 | +1 | 0 | 1 |
| 8 | 9 | 9 | 0 | 0 | 0 |
| 9 | 10 | 10 | 0 | 0 | 0 |
| **合计** | **89** | **94** | **+5** | **1** | **6** |

逐 episode 完全一致的 task **4/10**（task 1、6、8、9）。Object 的整体 +5 pp 由**单方向**的 fail→succ 翻转主导（6 次对 1 次），且 succ→fail 仅出现在 task 0。

### 3.2 Gate 核验（两 suite 均 PASS）

| 项 | 要求 | Spatial | Object |
|---|---|---|---|
| 量化站点 | 384（296 Linear + 88 MatMul） | ✅ 384 | ✅ 384 |
| `coverage_summary.json` | PASS | ✅ PASS | ✅ PASS |
| runtime 稀疏行 | 3736 | ✅ 3736 | ✅ 3736 |
| static weight 行 | 296 | ✅ 296 | ✅ 296 |
| outlier 行 | 5040 | ✅ 5040 | ✅ 5040 |
| scale 文件 | 1152，且与 `calibrate/scale_audit.json` 逐项一致 | ✅ | ✅ |
| 精度路由 | Vision/VLM Linear `(e4m3,int8,e4m3)`、Expert Linear `(e4m3,int4,e4m3)`、MatMul `(e4m3,e4m3,e4m3)` | ✅ 逐 module 断言 | ✅ |
| 执行路径 | B0 `raw` / M0 `quant_forward`，其余元数据完全相同 | ✅ `execution.json` 仅 `mode` 不同 | ✅ |
| 结果完整性 | 10 tasks × 10 episodes，`pc_success` 与逐 task 计数一致 | ✅ | ✅ |
| 计算量覆盖 | 所有量化位点 / flow step 调用数与 sparsity role 一致 | ✅ | ✅ |

两 suite 的 `execution.json` 除 `mode` 外完全相同：`linear_sites=296`、`matmul_sites=88`、`vision_backend=explicit_eager`、`vlm_expert_backend=injected_eager`。

### 3.3 runtime 稀疏（native 口径，S\|MMM v1，各 100 episodes）

| 分组 | Spatial element | Spatial bit | Object element | Object bit |
|---|---:|---:|---:|---:|
| Vision | 6.3433% | **54.6951%** | 6.2715% | **54.7433%** |
| VLM | 1.6338% | **53.8021%** | 1.7805% | **53.8986%** |
| **Vision+VLM pooled** | 6.0247% | **54.6346%** | 5.9676% | **54.6862%** |
| Expert | 2.8196% | **54.0735%** | 2.9511% | **54.1066%** |

按算子类型细分（element / bit）：

| 分组 | Spatial Linear | Spatial MatMul | Object Linear | Object MatMul |
|---|---|---|---|---|
| Vision | 4.0156% / 48.5663% | 7.5089% / 57.7641% | 3.9677% / 48.5830% | 7.4251% / 57.8282% |
| VLM | 0.0189% / 51.9779% | 4.6317% / 57.1884% | 0.0194% / 51.9978% | 5.0497% / 57.4272% |
| Expert | 0.0189% / 52.3305% | 5.5015% / 55.7425% | 0.0186% / 52.3085% | 5.7591% / 55.8284% |

观察：

- **bit 稀疏度跨 suite 极稳定**（54.07%–54.74%，四组 × 两 suite 的极差 < 0.7 pp），说明 fp8 S\|MMM 口径的位稀疏主要由**数据分布与格式**决定，与具体任务 suite 几乎无关。
- Linear 的 element 稀疏极低（VLM/Expert 约 0.019%）：A/O 走 per-site scale，e4m3 下恰好取到 0 的元素很少；稀疏主要体现在 exponent 场（bit 口径，约 48.5%–52.3%）。
- MatMul element 稀疏（4.63%–7.51%）显著高于 Linear，与 09-23 / 09-25 前序 MatMul 通路结论一致。

### 3.4 static weight 稀疏（sign-aware，INT8/INT4 分开报告）

两 suite **完全相同**（同一份权重、同一 scale，仅 rollout 数据不同）：

| 分组 | 权重格式 | element 稀疏 | bit 稀疏 |
|---|---|---:|---:|
| Vision + VLM | INT8 | 0.7580% | **63.3772%** |
| Expert | INT4 | 8.6806% | **73.7238%** |

INT4 Expert 的 element 零率（8.68%）比 INT8（0.76%）高一个量级，bit 稀疏高约 10.3 pp——符合「位宽越窄、可表示幅值档位越少、落入 0 的权重越多」的预期。**不把 INT 口径与 FP8 S\|MMM 口径混当成同一格式指标。**

### 3.5 outlier 浮点旁路（实测占比）

| suite | pooled 被保护元素 / 总元素 | 行数 |
|---|---:|---:|
| Spatial | **1.5218%** | 5040 |
| Object | **1.5222%** | 5040 |

设计值 `outlier_ratio=0.01`；实测略高于 0.01，因为权重侧的 runtime mask 定义为「activation channel mask ∪ weight 自身 outlier mask」，两个 ~1% 掩码取并集后接近 2%。**这些元素在 normal 路径走浮点旁路、不参与量化**；`module_sparsity.csv` 的 `*_native` 列已将其排除，避免高估稀疏度。

### 3.6 dense-equivalent 计算量

每 generation 的算法计算量由网络结构决定，**两 suite、两组完全一致**：**598.43 GFLOPs/generation**。

| 组件 | FLOPs / generation | 占比 |
|---|---:|---:|
| Vision | 427.62 G | **71.5%** |
| Expert | 108.59 G | 18.1% |
| VLM | 57.60 G | 9.6% |
| connector | 3.02 G | 0.5% |
| other（patch embed / action-state-time proj） | 1.60 G | 0.3% |
| **all** | **598.43 G** | 100.0% |

分 suite 总量（与 rollout 长度相关）：

| suite | 组 | generations | all 总 TFLOPs | Vision | VLM | Vision+VLM | Expert |
|---|---|---:|---:|---:|---:|---:|---:|
| Spatial | baseline | 1297 | 776.2 | 554.6 | 74.7 | 629.4 | 140.8 |
| Spatial | quant | 1412 | **844.99** | 603.80 | 81.34 | 685.13 | 153.33 |
| Object | baseline | 1564 | 936.0 | 668.8 | 90.1 | 758.9 | 169.8 |
| Object | quant | 1507 | **901.84** | 644.42 | 86.81 | 731.23 | 163.64 |

> ⚠️ 上表 Vision / VLM / Vision+VLM / Expert 四列**不可相加**：`Vision+VLM` 是前两列的和、`all` 是全部含 connector/other 的和。正式 FLOPs 仅取自 `compute.csv` / `compute_summary.json`，**禁止直接累加 `workload.csv` 的 MACs**（其 Linear activation/output、MatMul A/B/O 是重复的角色行）。
>
> 这是**算术密集算子的 dense-equivalent 算法计算量**，不含 bias/softmax/norm/elementwise/embedding/data-movement/量化统计开销/outlier 额外 GEMM；**不是 GPU 实际指令数，也不是稀疏后剩余计算量或硬件加速比**。

### 3.7 分析

1. **两个 suite 的 Δ 方向相反且量级相同**（Spatial −5.0 pp、Object +5.0 pp），合计 200 episodes 下 Δ = **0.0 pp**。这是「采样噪声主导」的最直接证据。
2. **翻转结构不对称**：Spatial 以 succ→fail 为主（10 vs 5），Object 以 fail→succ 为主（1 vs 6）。若量化真有系统性精度损失，不应出现这种与 suite 相关的方向反转。
3. **工作负载侧完全稳定**：runtime bit 稀疏跨 suite 极差 < 0.7 pp，static weight 稀疏逐位相同，计算量/generation 完全固定。说明**量化配置本身的行为是确定的**，SR 波动来自闭环仿真与 10-episode 采样分辨率。
4. **耗时**：M0 显著慢于 B0（Spatial 13008 s vs 648 s；Object 7266 s vs 649 s），这是 `quant_forward` 逐算子假量化 + 稀疏统计（`chunk_size=4194304` 分块归约）的开销，**不代表真实 INT/FP8 硬件速度**。
5. 本实验**未做 component-wise 消融**，不能把任何 Δ 归因于 Vision/VLM W8 或 Expert W4 中的哪一部分。

<!-- 按实验变量分组展开：每组建一个小节，含数据表、图、简要分析 -->

## 4. 结论

<!-- 对 §1「实验目的」中每个问题的直接回答；可执行的结论（保留/否决某配置） -->

1. Phase K 混合精度配置在 Spatial 与 Object 上均跑通全部 Gate（384 sites / 1152 scales / 3736 runtime rows / 296 weight rows / 5040 outlier rows / scale 审计逐项一致 / 精度路由逐 module 断言），**无工程性问题**。
2. 成功率：Spatial **86.0 → 81.0**（−5.0 pp）、Object **89.0 → 94.0**（+5.0 pp）、两 suite 合计 **87.5 → 87.5**（**0.0 pp**）。
3. 因此**既不声称「无损」，也不否决该配置**：在当前 100 ep/task、单 seed 协议下，该配置的 SR 差异与同 checkpoint 的采样波动同量级，**没有可检测的系统性掉点**。
4. 工作负载特征：runtime S\|MMM bit 稀疏 Vision+VLM ≈ **54.6%**、Expert ≈ **54.1%**（跨 suite 稳定）；static weight 稀疏 INT8 ≈ **63.4%**、INT4 ≈ **73.7%**；outlier 浮点旁路实测 ≈ **1.52%**；算法计算量 **598.43 GFLOPs/generation**，其中 Vision 占 71.5%。
5. 该结果可作为后续**多 seed 复现**与 **component-wise 消融**的对照基线；本实验本身不足以支撑统计显著性结论。

## 5. 问题与后续

<!-- 实验中发现的异常、失败 run、待验证问题、下一步实验（链接到新实验子目录） -->

1. **Goal suite 跳过（协议偏离）**：已由前序单 suite 实验 [`2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal`](../2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/docs/results.md) 覆盖（同 checkpoint、同配置：B0 **87%** → M0 **83%**，Δ **−4.0 pp**，退化集中在 task 9）。两实验的 `src/` 框架代码在 `4cdccc4..b8115e2` 之间**零改动**，配置逐项一致。**本轮不再重跑**，引用时须标注为跨轮引用、脚本版本不同。
2. **Long baseline 不重跑（协议偏离，有风险）**：改用 `outputs/table2_repro_audit/06_simulator/mj332_A_na10_ep10_*`（2026-08-27，FP，100 ep）。该运行的 `chunk_size=50` / `n_action_steps=10` / `num_steps=10` / `seed=1000` / `pretrained_path=lerobot/smolvla_libero` 与本轮逐项一致，但其 `pretrained_revision=None`（**未钉 checkpoint 版本**），且同协议同 seed 下仍与本轮实测存在偏差（Spatial 85 vs 86、**Object 95 vs 89，差 6 点**）。**故 Long 的 69.0 只能作近似参照，不可与 M0 相减作为严格结论。**
3. **未执行 `summarize.py`（决策 D）**：因跳过 Goal 后其硬编码四 suite 与 `assert episodes==400` 会失败，故不生成根级 `summary.json` 与 6 张汇总 CSV，也不修改该脚本（以保持 `run_arm.py` 的 `source_fingerprint` 完整、不影响后续 stage）。本文件中的合计行均为**人工按原始 `result.json` 累加**。
4. **待补**：Long quant 完成后回填 §2/§3；若要严格结论，建议补跑 Long baseline 与多 seed 重复。
5. **未做**：多 seed 方差估计；component-wise 消融（Vision/VLM W8 与 Expert W4 各自贡献）；与其它 suite 基线的直接相减（协议/轮次不同）。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-25 | 创建文档 | |
| 2026-09-29 | 回填 Spatial（86→81，−5.0 pp）与 Object（89→94，+5.0 pp）全量结果：总表、逐 task 与翻转、Gate 核验、runtime/static 稀疏、outlier、dense-equivalent 计算量；记录 Goal 跳过、Long 用历史参照与不执行 summarize 三项偏离 | lfwang |
