# 结果记录（Results）

- **实验名称**：2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal
- **状态**：done（B0/M0 各 100 episodes 跑通，`check_quant` 全 PASS）
- **最后更新**：2026-09-25

## 1. 摘要

Phase K 正式评测于 2026-09-25 在 commit `4cdccc4`（运行期间 `worktree.patch` 为 0 字节，工作区干净）上按 `run_all.sh` 顺序一次跑通：prepare（`snapshot_download` 固定 checkpoint `31d453f7…`）→ fresh calibration（`reuse=0`、`recalibrate=384`，产出 **1152 个 scale**）→ M0-smoke（task0×1ep，**SR=100%**）→ B0 eager raw Goal 100ep（**SR=87%**）→ M0 mixed `quant_forward` Goal 100ep（**SR=83%**）→ 汇总 **PASS**。

配置：Vision 与 VLM 的 Linear 用 **WINT8**，Expert 的 Linear 用 **WINT4**；三部分的 Linear A/O 与全部 QK/PV MatMul 的 A/B/O 用 **FP8 E4M3 PoT**；`outlier_ratio=0.01`、`weight_quant_granularity=per_tensor`、scale 粒度 `per_site`。**整数权重 scale 保持连续 calibrated 值**：1152 个 scale 中只有 856 个要求 PoT（A/O 与 MatMul），296 个 `w_scale` 不要求 PoT。

**主结论（同 checkpoint、同 eager 路径、同协议）：混合精度 PTQ 把整体 Goal SR 从 87%（B0）降到 83%（M0），Δ = −4.0 pp。** 退化集中在单个 task（task 9 由 7/10 降到 4/10），task 4、5 各 −1，task 6 反向 +1，其余 7 个 task 逐 episode 完全一致；每 task 仅 10 episodes，1 次翻转即 10 pp，故该 4 pp 差异不足以支撑「显著掉点」的结论。

## 2. 总结果表

| 组 | 执行模式 | Goal成功数/100 | SR | Δ vs B0 | eval_s |
|---|---|---|---|---|---|
| B0 eager raw（未量化基线） | `raw` | 87 | **87.0%** | — | 4756.8 |
| M0 Vision+VLM W8 / Expert W4；AFP8 PoT | `quant_forward` | 83 | **83.0%** | **−4.0 pp** | 10910.7 |
| M0-smoke（task0×1ep，不计入正式 100ep） | `quant_forward` | 1 | 100% | — | 116.2 |

两组 `avg_max_reward` 与 `pc_success` 一致（0.87 / 0.83），奖励即为 0/1 成功信号。两组的 `execution.json` 除 `mode` 外完全相同：`linear_sites=296`、`matmul_sites=88`、`vision_backend=explicit_eager`、`vlm_expert_backend=injected_eager`。

## 3. 分组结果与分析

### 3.1 逐 task 对照（`task_success.csv`）

| task_id | B0 成功/10 | M0 成功/10 | Δ pp |
|---|---:|---:|---:|
| 0 | 10 | 10 | 0 |
| 1 | 10 | 10 | 0 |
| 2 | 8 | 8 | 0 |
| 3 | 7 | 7 | 0 |
| 4 | 10 | 9 | −10 |
| 5 | 10 | 9 | −10 |
| 6 | 5 | 6 | +10 |
| 7 | 10 | 10 | 0 |
| 8 | 10 | 10 | 0 |
| 9 | 7 | 4 | **−30** |
| **合计** | **87** | **83** | **−4.0** |

−4 pp 与 task 9 单点的 −3 成功几乎同量级；task 2、3、6 在 B0 本来就只有 5–8/10，说明该协议下同 checkpoint 的单 task 波动本就在 ±1–2 成功的量级。

### 3.2 Gate 核验（`check_quant` 全 PASS）

| 项 | 要求 | 实测 |
|---|---|---|
| 量化站点 | 384 | 384 ✅（296 Linear + 88 MatMul，`module_id` 集合逐项断言） |
| scale 文件 | 1152 | 1152 ✅（888 Linear 的 w/a/o + 264 MatMul 的 A/B/O），全部记录 SHA256 |
| scale 与校准一致 | 完全一致 | ✅ `quant/scale_audit.json == calibrate/scale_audit.json`（逐项 `entries==previous` 断言） |
| PoT 断言 | A/O 与 MatMul 必须为 2 的幂 | 856/856 ✅（`math.frexp(v)[0]==0.5`）；296 个 `w_scale` 为有限正值 |
| runtime 行 | 3736 | 3736 ✅（Vision 216 + VLM 320 + Expert 3200） |
| weight 行 | 296 | 296 ✅ |
| runtime 位宽 | 4 bits/element（native 口径） | ✅ 全部行满足 `total_bits_native == 4 × total_elements_native` |
| 精度路由 | 逐 module 断言 | ✅ Vision/VLM Linear `(e4m3,int8,e4m3)`、Expert Linear `(e4m3,int4,e4m3)`、MatMul `(e4m3,e4m3,e4m3)` |
| 结果完整性 | 10 tasks × 10 episodes | ✅ 100 episodes，`pc_success` 与逐 task 计数一致 |

### 3.3 runtime 稀疏（native 口径，S\|MMM v1，100 episodes 汇总）

| 分组 | element 稀疏 | S\|MMM bit 稀疏 | bit 分子 / 分母 |
|---|---:|---:|---|
| Vision Linear | 4.0385% | **48.58%** | 958,157,580,695 / 1,972,146,339,840 |
| Vision MatMul | 7.4033% | **57.69%** | 2,272,088,216,144 / 3,938,301,079,420 |
| VLM Linear | 0.0192% | **52.01%** | 144,976,914,976 / 278,771,544,960 |
| VLM MatMul | 5.3555% | **57.60%** | 86,496,676,730 / 150,170,634,168 |
| **Vision+VLM pooled** | 5.9833% | **54.61%** | 3,461,719,388,545 / 6,339,389,598,388 |
| Expert Linear | 0.0191% | **52.33%** | 363,679,053,038 / 695,012,876,800 |
| Expert MatMul | 5.9228% | **55.87%** | 405,508,435,918 / 725,808,594,288 |
| **Expert pooled** | 3.0349% | **54.14%** | 769,187,488,956 / 1,420,821,471,088 |

- Linear 分组的 element 稀疏极低（VLM/Expert 约 0.019%）：Linear 的 A/O 走 per-site scale，e4m3 下**恰好取到 0** 的元素很少；稀疏主要落在 exponent 场（bit 口径），约 48–52%。
- MatMul 分组 element 稀疏 5.36–7.40%、bit 稀疏 55.87–57.69%，与 09-23 QK/PV-only 及 09-25 joint-smoke 实验的 MatMul 通路（57.60–57.68%）一致。
- Vision pooled bit 54.65%、Vision+VLM pooled bit 54.61%，与 joint-smoke 单 episode 的 54.58% 一致（差异来自 100 episodes 采样）。
- Expert runtime 行按 `phase=denoise, flow_step=0..9` 展开（3200 行）；Vision/VLM 固定 `phase=prefill, flow_step=-1`。

### 3.4 static weight 稀疏（`weight_sparsity_static.csv`，INT8/INT4 分组报告）

| 分组 | 权重格式 | element 稀疏 | bit 稀疏 | 零元素 / 总元素 | 稀疏位 / 总位 |
|---|---|---:|---:|---|---|
| Vision | INT8 | 0.8041% | **63.16%** | 682,961 / 84,934,656 | 429,149,913 / 679,477,248 |
| VLM | INT8 | 0.7332% | **63.50%** | 1,153,158 / 157,286,400 | 798,954,416 / 1,258,291,200 |
| **Vision+VLM** | INT8 | 0.7580% | **63.38%** | 1,836,119 / 242,221,056 | 1,228,104,329 / 1,937,768,448 |
| Expert | INT4 | 8.6806% | **73.72%** | 8,526,220 / 98,222,080 | 289,652,083 / 392,888,320 |

INT4 Expert 的 element 零率（8.68%）比 INT8 的 0.73–0.80% 高约一个量级，bit 稀疏（73.72%）比 INT8（63.38%）高约 10 pp，符合「位宽越窄、可表示幅值档位越少、落入 0 的权重越多」的预期。按要求 INT8 与 INT4 **分开报告**，不把 INT 口径与 FP8 S\|MMM 口径混当成同一格式指标。

### 3.5 outlier 浮点旁路（实测占比，`outlier_sidepath.csv`）

`outlier_ratio` 设计值为 0.01；实测「被保护元素 / 总元素」（元素口径）按 component 与 role 汇总：

| 角色 | Vision | VLM | Expert |
|---|---:|---:|---:|
| Linear activation | 0.9404% | 0.9495% | 0.9631% |
| Linear output | 0.9404% | 0.9606% | 0.9639% |
| Linear weight runtime mask | 1.9330% | 1.9499% | 1.9764% |
| MatMul A | 1.0110% | 0.8299% | 0.9398% |
| MatMul O | 1.0110% | 0.8299% | 0.9398% |
| MatMul B（权重操作数） | 2.4482% | 2.1424% | 2.3145% |
| **component 汇总** | **1.1682%** | **1.6486%** | **1.8408%**（pooled **1.5220%**） |

`weight_runtime_mask` 定义为「activation channel mask 与 weight 自身 outlier mask 的广播并集」，两个 ~1% 掩码取并集后接近 2%，因此权重侧实测占比高于 0.01；MatMul 的 B 同理。**这些被保护元素在 normal 路径上走浮点旁路、不参与量化**；`module_sparsity.csv` 的 `*_native` 列已把该部分排除，避免高估稀疏度（另有 `*_reported` 列保留含保护的原始计数）。

### 3.6 分析

1. 混合精度 PTQ 的整体掉点（−4 pp）在量级上小于历史「Expert W4 单独量化」类配置常见的损失；但本实验未做 component-wise 消融，**不能把 4 pp 归因于 Vision/VLM W8 或 Expert W4 中的哪一部分**。
2. 逐 task 看，退化是单点式的（task 9），不是整体性下降，7/10 task 逐 episode 完全一致。结合每 task 10 episodes 的采样分辨率，本次只能说明「没有出现灾难性崩坏」，不足以证明「无损」。
3. 工作负载侧，Vision+VLM 与 Expert 的 runtime bit 稀疏几乎相同（54.61% vs 54.14%），说明把 Vision/VLM 设为 W8、Expert 设为 W4 主要改变的是**静态权重的可压缩性**（63.38% vs 73.72%）与精度权衡；runtime exponent 稀疏度主要由 A/O 的 FP8 与数据分布决定。

## 4. 结论

1. Phase K 配置（Vision+VLM W8 / Expert W4，A/O 与 QK/PV 全 FP8 E4M3 PoT，1% outlier）在 LIBERO Goal 100 episodes 上跑通，覆盖与一致性 Gate（384 sites / 1152 scales / 3736 runtime rows / 296 weight rows / scale 审计一致）**全 PASS**。
2. 同 checkpoint、同 eager 路径、同协议下：B0 raw **87%** vs M0 mixed-PTQ **83%**，**Δ = −4.0 pp**；退化集中在 task 9（−3 成功），另有 task 4/5 各 −1、task 6 +1。
3. runtime S\|MMM bit 稀疏：Vision+VLM **54.61%**、Expert **54.14%**；static weight 稀疏：Vision+VLM INT8 **63.38%**、Expert INT4 **73.72%**；outlier 浮点旁路实测 pooled 1.52%。
4. 由于 100 episodes × 单 seed 的采样不确定性，本次**既不声称「无损」，也不声称「显著掉点」**；该结果作为后续多 seed 复现与 component-wise 消融的对照基线。

## 5. 问题与后续

- **证据副本已随本次回填提交**（本目录 `docs/` 下）：`summary.json`、`task_success.csv`、`coverage_summary_{smoke,quant}.json`、`result_{smoke,baseline,quant}.json`、`execution_{baseline,quant}.json`、`scale_audit_quant.json`、`quantization_manifest_quant.csv`、`module_sparsity_quant.csv`、`weight_sparsity_static_quant.csv`、`workload_quant.csv`、`outlier_sidepath_quant.csv`、`run_log_filtered.txt`（去掉 tqdm 刷屏的 `run.log`）、`prepared.json`、`commit.txt`。原始产物（含 7.0 MB 完整 `run.log`、`worktree.patch`、`packages.txt`、`source_configs/`、`baseline/`、`smoke/` 与 `calibrate/`）在 `outputs/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/`（`outputs/` 被 .gitignore 忽略）；scale 文件在 `scales/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/quant/`。
- 未做：多 seed / 多种子方差估计；component-wise 消融（Vision/VLM W8 与 Expert W4 各自贡献）；与历史其它 suite 成功率的直接相减（协议不同，不可相减）。
- 已知环境噪声（不影响结果）：`torchcodec` 载入失败（FFmpeg 8/7/6/5 分别缺 `libavutil.so.60/59/58/57`，最终 `libtorchcodec_core4.so: undefined symbol: torch_dtype_float4_e2m1fn_x2`）——本实验 `max_episodes_rendered=0`，视频关闭，无影响；HF Hub 未认证警告；`torch_dtype` deprecated 警告；`lerobot_eval.py:372` 的 `np.bool` DeprecationWarning。
- 复跑前必须先归档 `outputs/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/` 与 `scales/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/`，否则 `run_all.sh` 会拒绝执行。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-25 | 创建混合精度Goal评测实验 | |
| 2026-09-25 | 回填 B0/M0 各 100ep 正式结果（SR 87% → 83%，Δ −4.0 pp）、Gate 核验、runtime/weight 稀疏与 outlier 旁路实测，并附 `docs/` 证据副本 | |
