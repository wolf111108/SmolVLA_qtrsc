# VLA_tcs2 实验数据与条件总结（截至 2026-09-03）

> 来源：`doc/logs/` 全部日志（2026-08-22 ~ 2026-09-04）+ `outputs/` 产物。
> 用途：一页纵览所有跑出的数据与实验条件，供写报告/续实验/交接引用。

---

## 0. 实验条件（自包含说明，供项目外读者）

### 0.1 这个项目在做什么

对 **SmolVLA**（一个 4.5 亿参数的视觉-语言-动作模型，VLA）做**训练后量化**（PTQ, Post-Training Quantization）：把训练好、原本以 FP32/BF16 浮点运行的模型权重和激活压缩成低精度格式（INT16/12/8、FP8），观察机器人操作成功率掉多少。目标是找到"几乎不掉分"的最低精度组合，降低推理硬件成本。

**评测方法**：模型在 **LIBERO** 仿真环境里操作机器人完成任务（抓碗、开抽屉等），统计**成功率 SR**（Success Rate）。LIBERO 有 4 个子任务集（suite）：Spatial / Object / Goal / Long（Libero-10），每个 suite 含 10 个任务。

### 0.2 被量化的模型

| 项 | 值 | 说明 |
|---|---|---|
| checkpoint | `lerobot/smolvla_libero` | 社区公开的 SmolVLA LIBERO 微调版（内部代号 A 类），量化研究的默认底座 |
| 为什么选它 | 五个公开 checkpoint 对比中总分最高（见 §3） | 论文官方 checkpoint 无法复现 87.3%，故以自测 FP 为对照 |
| 结构 | 450M 参数：SmolVLM2-500M 视觉语言模型（16 层，专家宽度 0.75）+ 动作专家 | 输入两张相机图像 + 8 维机器人状态 + 语言指令，输出 7 维动作序列 |

### 0.3 量化了什么、没量化什么

SmolVLA 内部有几百个矩阵乘法算子，本项目的量化范围（对每个实验都一样，除 matmul 实验外）：

| 范围 | 算子 | 数量 | 说明 |
|---|---|---|---|
| ✅ 量化 | Transformer 的 7 种投影层：注意力 q/k/v/o_proj + FFN gate/up/down_proj | 224 个 Linear | 分布在视觉语言模型 + 动作专家两部分的每个 Transformer 层里 |
| ✅ 量化（仅 #6 实验） | 注意力里的两个矩阵乘：QK^T（query·key）和 PV（概率·value） | 共享 1 对 | 见 §0.7 的 matmul 说明 |
| ❌ 不量化 | 图像编码器（SigLIP）、视觉-语言连接器、动作头（含最终输出投影 action_out_proj） | — | 动作头输出速度场，精度最关键，保持浮点 |

**术语**：每层量化涉及三个张量——**a**（输入激活）、**w**（权重）、**o**（输出）；配置里 `a_bit/w_bit/o_bit` 分别指定三者的精度（如 `12/12/12` = 全 int12，`e4m3` = FP8 格式）。

### 0.4 两种关键量化技术（读懂主表必需）

1. **outlier 保护**（`outlier_ratio=0.01`）：激活值分布有长尾——少数通道的数值特别大，会把整层的量化 scale 撑大，导致其余 99% 的"正常"值只剩几个量化级别可用（这是 int12 从 95% 崩到 7% 的根因）。解法：**数值最大的 1% 通道保留浮点不量化**，其余 99% 用正常 scale 量化。代价是少量计算不走低精度通路。
2. **PoT scale**（Power-of-Two，`pot_fp8_outlier` 方法）：量化 scale 通常 是任意浮点数，反量化时要做乘法；PoT 把 scale 强制为 2 的幂（2^k），乘法变成移位，硬件更省。风险是 scale 被"向上取整"损失表示精度，实验证明该损失可忽略。

### 0.5 校准协议（量化前必须做）

量化 scale 需要先用真实数据统计各层数值范围，这一步叫**校准**：

| 项 | 值 |
|---|---|
| 校准数据 | `HuggingFaceVLA/libero` v3.0（即该 checkpoint 的训练数据集） |
| 采样 | 随机抽 8 条演示轨迹（seed 42），每 4 帧取 1 帧 → 约 356 帧，batch_size 8 |
| 校准前向 | 用模型的**推理接口** `predict_action_chunk`（与部署一致，非训练前向） |
| 产物 | 每层 3 个 scale 存 pickle 文件（`scales/<实验名>/`），可复用（`--reuse`） |

### 0.6 评测协议

| 项 | 值 | 说明 |
|---|---|---|
| 环境 | LIBERO `libero_object` suite | 10 个桌面物体操作任务 |
| 规模 | 每任务 10 episodes = 100 集/实验 | "ep10 快测"口径，用于快速筛选；正式结论需 ep50 |
| 推理参数 | `n_action_steps=10`：模型每次预测 50 步动作块、执行前 10 步后重新观测（闭环重规划） | 该参数对 SR 影响巨大（+11pp，见 §4），对比时必须一致 |
| 随机种子 | seed 1000 | |
| FP 对照 | 同协议浮点运行 = **93.8%** | 所有 Δ 都相对这个数 |

### 0.7 matmul 量化的特别说明（#6 实验）

SmolVLA 的注意力实现里，所有层共用同一个模型级的注意力接口函数，无法逐层替换。因此 QK^T/PV 的量化器是**全模型共享的一对**，其 scale 由所有层激活的 absmax 聚合而成（保守但安全）。这是框架的架构限制，不是刻意设计。

### 0.8 软硬件环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA H100 NVL（主评测机 h100） |
| conda 环境 | `smolvla_eval`：Python 3.12、PyTorch 2.7.1+cu118、mujoco **3.3.2**、lerobot |
| MuJoCo 版本为什么重要 | 3.8.1 会系统性压低 SR（个别任务 0/10 → 3.3.2 下 8/10），所有最终数据统一 3.3.2 | 
| 渲染 | `MUJOCO_GL=egl`（无头服务器 GPU 渲染） |

---

## 1. 量化实验主表（libero_object, ep10 快测口径）

**怎么读这张表**：每行一个量化配置，SR 是 100 集机器人操作的成功率，Δ 是相对浮点模型（93.8%）的变化。"a/w/o 格式"列 = 激活/权重/输出三者的数值格式（`12/12/12` 表示三者都是 12 位整数；`e4m3` 是 8 位浮点 FP8 的标准格式）。#2 → #3 的对比就是 §0.4 所说 outlier 保护的作用；#5 → #6 是 PoT scale 与 matmul 的增量验证。

| # | 配置 | a/w/o 格式 | outlier | method | SR | Δ vs FP | 结果目录 |
|---|---|---|---|---|---:|---:|---|
| 0 | FP baseline（不量化） | 浮点 | — | — | **93.8%** | — | `verify_libero` |
| 1 | int16 | 16/16/16 | — | per_tensor* | 89.0% | -4.8 | `smolvla_int16_quant_test` |
| 2 | int12 纯 | 12/12/12 | — | per_tensor* | **7.0%** | **-86.8** | `smolvla_int12_quant_test` |
| 3 | int12+outlier | 12/12/12 | 0.01 | outlier* | **95.0%** | +1.2 | `smolvla_int12_outlier_quant_test` |
| 4 | int8+outlier | 8/8/8 | 0.01 | outlier* | **92.0%** | -1.8 | `smolvla_int8_outlier_quant_test` |
| 5 | PoT-FP8+outlier（linear） | e4m3 全 | 0.01 | pot_fp8_outlier | **93.0%** | -0.8 | `smolvla_fp8_pot_outlier_quant_test` |
| 6 | **PoT-FP8+outlier（linear+matmul）** | e4m3 全 | 0.01 | pot_fp8_outlier | **93.0%** | **-0.8** | `smolvla_fp8_pot_outlier_matmul_quant_test` |
| 7 | int8 纯 / int12 mixed | — | — | — | ⬜ 未跑 | — | — |

\* #1–4 在 `method` 注册表重构前跑，config 未写 `method`（复现需补 `method: outlier` / `per_tensor`）。method = 代码注册表里的量化方法名（§0.4 的两种技术对应 `outlier` / `pot_fp8_outlier`，无保护的基础量化是 `per_tensor`）。

### 1.1 逐 task 明细（成功数/10）

| 配置 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | 合计 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| int16 | 10 | 10 | 8 | 8 | 10 | 7 | 10 | 8 | 8 | 10 | 89 |
| int12 纯 | 0 | 1 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 2 | 7 |
| int12+outlier | 10 | 10 | 10 | 10 | 10 | 8 | 10 | 8 | 9 | 10 | 95 |
| int8+outlier | 10 | 10 | 9 | 9 | 10 | 7 | 10 | 8 | 9 | 10 | 92 |
| PoT-FP8 linear | 10 | 10 | 10 | 8 | 10 | 8 | 10 | 8 | 9 | 10 | 93 |
| PoT-FP8 L+M | 10 | 10 | 8 | 10 | 10 | 8 | 10 | 9 | 9 | 9 | 93 |

### 1.2 PoT scale 审计（#5 vs #6）

| 类型 | count | k 范围 | 备注 |
|---|--:|---|---|
| a_scale (linear) | 112 | [-7, -4] | 两次一致 |
| w_scale (linear) | 112 | [-12, -9] | |
| o_scale (linear) | 112 | [-8, -4] | |
| A_scale (matmul) | 2 | [-8, -4] | 共享 qk/pv 各 1 |
| B_scale (matmul) | 2 | [-7, -5] | |
| O_scale (matmul) | 2 | [-6, 0] | 含 k=0（=1.0，softmax 概率通路） |

全部 2 的幂（342 文件，bitwise frexp 验证通过）。

### 1.3 核心结论

1. **outlier 通道保护（1%，absmax 排序）是关键开关**：int12 从 7%→95%（+88pp），int8 达 92%；
2. **PoT scale 无额外损失**：FP8 e4m3 + 2^k scale 与 FP 持平（-0.8pp），硬件 scale 乘法器可省为移位；
3. **matmul 量化零代价**：linear+matmul 与 linear-only 完全同分（93.0%），attention QK/PV 也量化后无退化——全算子（除 vision/action head）FP8 化可行；
4. 机制：per-tensor scale 被激活长尾通道撑大（down_proj_0 top1% 通道占 56.7% 能量），1% 通道走 FP 后 normal 部分 scale 恢复。

---

## 2. FP baseline 多口径总表（A 类 `lerobot/smolvla_libero`）

| 协议 | Spatial | Object | Goal | Long | 平均 | 出处 |
|---|---:|---:|---:|---:|---:|---|
| 3.8.1, na=1, ep10 | 81 | 60 | 77 | 59 | 69.2% | 五模型 benchmark（8-25） |
| **3.3.2, na=1, ep10** | 81 | 78 | 77 | 60 | **74.0%** | Phase 6 首轮（8-26） |
| **3.3.2, na=10, ep50（正式）** | 86.0 | 93.8 | 87.8 | 74.2 | **85.45%** | P3（9-1 回填） |
| 3.3.2, na=10, ep10 | 85.0 | 95.0 | 89.0 | 69.0 | 84.5% | P2 |
| 3.3.2, na=1, ep50 | 85.2 | 78.4 | 78.4 | 55.8 | 74.45% | P1 |
| verify_libero ep50 | 84.8 | 95.2 | 87.8 | 70.0 | 84.45% | 8-28（量化工程 eval 链路自测） |

> 注意口径不可混用：五模型对比表（§3）全部是 3.8.1 + na=1；量化实验全部 na=10。

---

## 3. 五公开 checkpoint 对比（mujoco 3.8.1, na=1, ep10，历史 benchmark）

| Suite | 论文 | lerobot (A) | k1000dai (D) | HuggingFaceVLA (B) | tiantianx (C) |
|---|---:|---:|---:|---:|---:|
| Spatial | 90 | **81** | 64 | 65 | 76 |
| Object | 96 | 60 | **82** | 71 | 59 |
| Goal | 92 | **77** | 70 | 72 | 63 |
| Long | 71 | **59** | 46 | 37 | 40 |
| **平均** | **87.3** | **69.2** | 65.5 | 61.2 | 59.5 |

- 结论：无一接近论文；A 类最高定为量化 baseline；
- hfvla_ckpts100k 因 2.2B-Instruct 底座 + 无 state 训练被审计关闭（不可修）；
- MuJoCo 3.3.2 是决定性变量：A 类 Spatial Task5 从 0/10 → 8/10。

---

## 4. 消融矩阵：n_action_steps × n_episodes（A 类, mj332）

| 数据点 | na | ep | Spatial | Object | Goal | Long | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 1 | 10 | 81 | 78 | 77 | 60 | 74.0% |
| P1 | 1 | 50 | 85.2 | 78.4 | 78.4 | 55.8 | 74.45% |
| P2 | 10 | 10 | 85.0 | 95.0 | 89.0 | 69.0 | 84.5% |
| P3 | 10 | 50 | 86.0 | 93.8 | 87.8 | 74.2 | **85.45%** |

归因：**na 1→10 贡献 +10.5~11.0pp（主因）**；ep 10→50 贡献 +0.45~0.95pp（噪声级）。chunk 执行/闭环重规划是关键协议。na=30/50 补点待跑。

---

## 5. Meta-World MT50（第二基准，FP 基线）

checkpoint `lerobot/smolvla_metaworld`，50 任务 × 10 ep，seed 1000，metaworld 3.0.0 / mujoco 3.8.1：

| 难度 | 复现 | 论文 0.45B | 差异 |
|---|---:|---:|---:|
| Easy（28） | 79.6% | 82.5% | -2.9 |
| Medium（11） | 38.2% | 41.8% | -3.6 |
| Hard（6） | 56.7% | 45.0% | +11.7 |
| Very Hard（5） | 42.0% | 60.0% | -18.0 |
| 四难度算术平均 | **54.1%** | 57.3% | -3.2 |

（500 集加权整体 64.0%；勿与算术平均口径混用。）

---

## 6. SQNR 与张量分布分析（量化机制证据）

### 6.1 SQNR（int16 vs int12，按 kind 平均 dB）

| kind | int12 | int16 | Δ |
|---|---:|---:|---:|
| activation | 51.85 | 58.56 | -6.7 |
| **weight** | **38.77** | 57.09 | **-18.3** |
| **output** | **36.20** | 52.22 | **-16.0** |

重灾区：down_proj（output 20.2dB）、k_proj weight；activation 几乎无损。

### 6.2 通道级能量分析（112 层平均，outlier 机制依据）

| kind | ch_absmax p50/max | top1% 通道能量 | absmax 策略命中 |
|---|---:|---:|---:|
| **activation** | 0.323 | **13.7%**（down_proj_0 达 56.7%） | 12.5%（近最优） |
| output | 0.428 | 5.8% | 4.8% |
| weight | 0.909 | 1.1%（分布平坦） | 1.0% |

→ outlier 集中在 activation 通道；weight 通道保护基本无效也无必要；absmax 排序≈能量排序最优。

---

## 7. 代码架构现状（2026-09-03）

- **可插拔注册表**：`quant/scale_methods.py`（SCALE_METHODS）+ `quant/quant_methods.py`（QUANT_METHODS），各 6 条目（per_tensor / outlier / pot_fp8_outlier × linear+matmul），一个 `method` 名驱动校准+推理；
- **matmul**：`QuantizedMatMul` 流程控制化；注入为**模型级共享一对 qk/pv**（`get_attention_interface` 是模型级方法，架构限制）；
- **SQNR 日志**：`VLA_SQNR_LOG=1` 打开（默认关）；
- 关键提交：`ced2850`（linear 注册表化）、`b4a3ddb`（pot_fp8_outlier）、matmul 迁移待提交。

---

## 8. 在跑 / 待办实验

| 项 | 状态 |
|---|---|
| ~~PoT-FP8 linear+matmul~~ | ✅ **93.0%（-0.8pp），2026-09-03 19:43 完成** |
| 纯 int8 对照（单调性） | ⬜ |
| int8/12+outlier ep50 验证 | ⬜ |
| na=30 / na=50 消融补点 | ⬜ 脚本已备（`run_phase6_mj332_A_foursuite_na{30,50}_ep50.sh`） |
| PoT-FP8 四 suite ep50 全量 | ⬜（若要做正式报告） |
| int8/int12 outlier config 补 `method` 字段 | ⬜（重构前结果复现需要） |
| `None` scale 边界（pot_fp8 前向） | ⬜ 已知未修 |
| x/w 拆分 ratio、按层差异化 ratio | ⬜ 记录未实施 |

---

## 9. 结果文件索引

| 内容 | 位置 |
|---|---|
| 量化实验结果 | `outputs/experiments/<配置名>/result.json`、`summary.md`、`pot_scale_audit.{csv,json}`、`per_task_sr.csv` |
| FP 四 suite ep50 | `outputs/verify_libero/libero_*/eval_info.json` |
| 消融 P0–P3 | `outputs/table2_repro_audit/06_simulator/` + 8-26/9-1 日志 |
| 五模型对比 | `outputs/hfvla_vs_baselines.md`、`outputs/checkpoint_baselines_summary.md` |
| MT50 | `outputs/metaworld_mt50_lerobot_smolvla/eval_info.json` |
| SQNR | `outputs/quant_sqnr/*.jsonl` |
| 张量分布 | `outputs/tensor_dump_analysis/`、`outputs/tensor_dump_channel_analysis/` |
| 可视化 | `outputs/figures/quant_sr_vs_bitwidth.png`、`ablation_matrix_bar.png` |
