# VLA-TCS2

面向 VLA（Vision-Language-Action）模型的**量化评测与硬件建模框架**，首个 workload 为 **SmolVLA** + **LIBERO** 机器人仿真基准（已扩展 Meta-World 作为第二基准）。

> 本 README 只介绍项目背景与架构；实验设置、日志与结果数据分别在 [`experiments/`](experiments/)、[`doc/logs/`](doc/logs/)、[`outputs/`](outputs/) 下，不在此重复。
>
> 实验导航见 [§6 实验索引](#6-实验索引)；量化精度与稀疏度统计的完整口径见 [`doc/manual/SmolVLA_quantized_sparsity_research_manual.md`](doc/manual/SmolVLA_quantized_sparsity_research_manual.md)。

---

## 1. 项目背景与目标

本项目旨在搭建一套可扩展的 VLA 实验框架，长期目标包括：

1. 在 LIBERO / Meta-World 等机器人仿真环境中完成闭环 VLA evaluation，并尽可能复现论文结果；
2. 对 VLA 中的 Linear / MatMul 等核心计算引入 **PTQ（Post-Training Quantization）**；
3. 统计运行时 activation / weight / output 的数值范围、zero ratio、bit sparsity 等特征；
4. 对比量化前后的 Success Rate，量化精度损失；
5. 最终将模型 workload 映射到硬件模型，分析 latency / memory traffic / throughput / sparsity speedup；
6. 形成一套可复用到其他 VLA / VLM policy 的测试框架。

核心原则：**先搞清仿真与 baseline 复现，再做量化。**

### 参考论文

- **SmolVLA**: A vision-language-action model for affordable and efficient robotics（arXiv:2506.01844）

---

## 2. 项目架构

### 2.1 顶层目录

```text
VLA_tcs2/
├── main.py                  # 传统单配置入口（load config → build → calibrate → eval）
├── pyproject.toml           # Python 包配置（setuptools, src 布局）
├── README.md                # 本文件（项目背景与架构）
│
├── src/vla_tcs2/            # ★ 量化框架源码（见 §2.3）
├── experiments/             # ★ 实验管理：一切新实验的唯一入口（见 experiments/README.md 与 §6）
├── scales/                  # 校准产出的 scale pickle，按实验名分目录（不入库）
├── outputs/                 # 评测原始产出（result.json / eval_info.json / coverage / sparsity / reports / figures）
├── doc/                     # ★ 项目文档（见 §2.2）
├── docs/                    # Phase D/E 时期遗留文档（不新增内容）
├── configs/experiments/     # Phase D/E 时期的 YAML（不再新增；新配置放 experiments/<exp>/configs/）
├── scripts/                 # 仓库级通用脚本（评测 / 审计 / 汇总 / 绘图）
├── tests/                   # 单元测试
├── checkpoints/             # 本地模型 checkpoint（大文件，不纳入文档）
├── lerobot_current/         # 主 LeRobot（LIBERO 评测环境，固定 commit；提供 `lerobot` 包）
├── envs/                    # 环境冻结文件（pip freeze / conda yml）
└── er.md                    # 本地临时汇报草稿（.gitignore 忽略，不入库）
```

### 2.2 文档目录 `doc/`

```text
doc/
├── guides/
│   ├── handoff.md                    # 项目交接文档（最完整的背景与结论）
│   ├── rebuild_manual.md             # 新设备环境重建手册（含 84% 验收标准）
│   └── experiment_results_summary.md # 历史结果汇总（Phase D/E 时期为主）
├── manual/
│   ├── SmolVLA_quantized_sparsity_research_manual.md  # ★ 量化 + 稀疏度研究总手册（口径 / 公式 / 阶段结论）
│   └── metaworld_manual.md           # Meta-World 评估手册（第二基准接入）
├── logs/                             # 每日工作日志（按日期_机器命名），README.md 为索引
└── phaseE_worklog.md                 # Phase E 工作记录
```

### 2.3 量化框架源码 `src/vla_tcs2/`

| 模块 | 职责 |
|---|---|
| `model_wrapper.py` | 加载预训练 SmolVLA policy；按组件把选中 `nn.Linear` 替换为 `QuantizedLinear`、注入注意力 `QuantizedMatMul`（QK/PV）；安装 phase / flow_step 运行时钩子；提供 `switch_quantization_mode_all` 模式切换。主要入口：`ModelWrapper.build/set_mode`，内部 `_wrap_smolvla_linear_layers`、`_wrap_smolvlm_vision_linear_layers`、`_inject_smolvla_quantized_matmul` |
| `quant_linear.py` | 量化线性层，支持 `raw` / `scale_inspection` / `quant_forward` 三种模式与 `a/w/o` 三组精度规格、outlier 保护 |
| `quant_matmul.py` | 量化矩阵乘（attention 的 `Q@K^T` 与 `P@V`），三种模式 + outlier 保护 + mixed-precision |
| `calibration.py` | 校准流程：判定 reuse/recalibrate → 绑定 `QuantStatManager` → `scale_inspection` 前向 → 保存 pickle scale |
| `eval.py` | 进程内 LIBERO 评测（复用 lerobot `eval_policy_all`，支持量化模型直接评测） |
| `vision_attention.py` | 可选的 SmolVLM 视觉注意力适配器（eager/SDPA 等价实现），使 Vision 的 QK/PV 可被替换为量化算子；不修改 transformers 全局 attention registry |
| `runtime_context.py` | 基于 `ContextVar` 的 phase / flow_step 运行上下文，由钩子自动设置，供稀疏度与 workload 统计打标；**不改变模型行为** |
| `quant/quant_spec.py` | 统一精度描述（`int4/8/16`、`e4m3`、`e5m2`、`e2m1`(FP4)、`bf16`、`none`）与相应假量化/scale 工具 |
| `quant/scale_methods.py` | 可插拔的**校准侧** scale 估计方法，由 `quantization.scale_method`（或按模块覆盖）按名称选择 |
| `quant/quant_methods.py` | 可插拔的**评测侧**量化前向方法（`pot_ao_outlier` / `pot_fp8_outlier` 等），含四路径 outlier 分解 |
| `quant/stat_manager.py` | scale 注册与 pickle 落盘；runtime / static weight 稀疏度、outlier 旁路、unit sparsity 的收集与 CSV 导出 |
| `quant/utils.py` | STE / scale 工具函数 |
| `quant/test_methods.py` | 噪声/敏感性测试用的前向方法族（gaussian/rms 等，用于敏感度分析实验） |
| `hardware/` | 可插拔算子事件采集（`capture.py`）、dense GEMM core 解析估算（`backends/dense.py`）、JSONL 离线重放（`trace.py`）、报告汇总（`report.py`）；默认关闭 |
| `stats/` | 运行时统计（预留） |

### 2.4 量化流水线

```text
load config (YAML)
      ↓
ModelWrapper.build()        # 加载 policy → 替换 Linear → 注入 QK/PV MatMul → 安装 phase/flow_step 钩子
      ↓
set_mode("scale_inspection")# 校准模式：前向收集 Linear 的 w/a/o 与 MatMul 的 A/B/O 统计
      ↓
calibrate()                 # 前向采样 → save_all_scales (pickle 到 scales/<exp>/quant/)
      ↓
set_mode("quant_forward")   # 量化推理（基线组则保持 "raw"）
      ↓
evaluate()                  # LIBERO rollout → success rate + 稀疏度 / outlier / 计算量统计
      ↓
result.json · coverage_summary.json · sparsity/*.csv · compute*.csv
```

三种模块模式：`raw`（原样浮点）· `scale_inspection`（校准收集）· `quant_forward`（量化前向）。

> **基线隔离**：基线组与量化组必须走**同一条显式 attention 路径**，以隔离 backend 差异；
> 实验内由 `run_arm.py` 显式设置模式，不依赖 `main.py` 的默认模式选择（量化开启时默认 `quant_forward`）。
>
> **校准与评测分离**：scale 由校准阶段落盘，评测阶段读取并**逐项校验 SHA256 与校准审计一致**；
> 新增精度必须 fresh calibration，不得沿用旧 scale（`calibration_policy` 控制）。

### 2.5 SmolVLA 模型结构与量化范围

#### 模型结构

SmolVLA 由三层嵌套组成（源码：`lerobot_current/src/lerobot/policies/smolvla/`）：

```text
SmolVLAPolicy (PreTrainedPolicy 子类)           ← 顶层，负责 I/O / 归一化
└── model = VLAFlowMatching                     ← 核心，含 action head + flow matching
    ├── vlm_with_expert = SmolVLMWithExpertModel
    │   ├── vlm = SmolVLM2-500M-Video-Instruct
    │   │   ├── model.vision_model   (SigLIP 视觉编码器)
    │   │   ├── model.connector      (视觉→语言 模态投影 + 重采样)
    │   │   └── model.text_model     (SmolLM2 LLM，16 层)
    │   └── lm_expert                (action expert，16 层，隐藏宽 0.75×)
    ├── state_proj / action_in_proj / action_out_proj
    └── action_time_mlp_in / action_time_mlp_out
```

各组件职责：

| 组件 | 说明 |
|------|------|
| `SmolVLAPolicy` | 顶层策略类，负责归一化、processor（tokenize / 图像编码）、动作反解 |
| `VLAFlowMatching` | 核心类，持有 action head，实现 flow matching 的训练 / 采样 |
| `SmolVLMWithExpertModel` | VLM backbone + action expert 的组合模块 |
| `vlm`（SmolVLM2-500M） | 冻结的视觉-语言 backbone：SigLIP 视觉编码器 + SmolLM2 LLM |
| `lm_expert` | SmolVLA 的核心创新——从 VLM 配置派生、宽度 0.75× 的 action expert |

action head 各层（均位于 `VLAFlowMatching` 顶层）：

| 层 | 输入 → 输出 | 角色 |
|----|------------|------|
| `state_proj` | `max_state_dim` → `hidden_size` | 本体状态 → token 空间 |
| `action_in_proj` | `max_action_dim` → `expert_hidden_size` | 噪声动作 → expert 维度 |
| `action_out_proj` | `expert_hidden_size` → `max_action_dim` | 输出 flow matching 速度场 $v_t$ |
| `action_time_mlp_in/out` | `2×hidden` → `hidden` → `hidden` | 融合 timestep 与动作（SiLU 激活） |

关键机制：**cross-attention 交织**——`num_vlm_layers=16`、`num_expert_layers=16`、`self_attn_every_n_layers=2`。VLM 层做 self-attention 生成 prefix KV cache；expert 层大部分做 cross-attention，其 query 来自「噪声动作 + timestep」、key/value 来自 VLM 的 prefix cache。推理时 `sample_actions` 从纯噪声出发，经 `euler_integrate` 做 ODE 迭代去噪，得到最终 7 维动作。

#### 量化范围与站点清单（已实现）

`ModelWrapper._wrap_smolvla` 按**组件**组织 wrapping，每个组件可独立配置精度：

| 组件 | 覆盖算子 | Linear 站点 | QK/PV MatMul 站点 | 默认 |
|---|---|---:|---:|---|
| `vlm`（SmolLM2 `text_model`，16 层） | `q/k/v/o_proj` + `gate/up/down_proj` | 112 | 32 | ✅ 开启 |
| `expert`（`lm_expert`，16 层，宽 0.75×） | `self_attn` + `mlp` | 112 | 32 | ✅ 开启 |
| `vision`（SigLIP `vision_model`，12 层） | `q/k/v/out_proj` + `fc1/fc2` | 72 | 24 | 🔲 opt-in |
| **合计** | — | **296** | **88** | **384 sites** |

- **QK/PV MatMul 已实现量化**（attention 的 `Q@K^T` 与 `P@V`），由 `quantization.quantize_matmul` 控制、**默认 `false`**；Phase I 起所有正式实验均开启。
- **Vision 与 connector 为 opt-in**。`vision.enabled=true` 本身**不会** wrap 任何算子，必须再显式打开 `vision.linear.attn_proj`（48 个）/ `vision.linear.mlp`（24 个）与 `vision.matmul.enabled`（24 个 QK/PV）；connector 由 `quantization.connector.enabled` 单独控制（`module_id=connector.layer.0.connector_proj`）。三者默认全关时保持既有 **224 个 VLM/Expert Linear** 不变。
- **未量化**：action head（`state_proj` / `action_in_proj` / `action_out_proj` / `action_time_mlp_in|out`）位于 `VLAFlowMatching` 顶层，当前 `_wrap` 未触达；patch embedding 亦未覆盖。

> action head 是 flow matching 的**最终输出层**，直接产生速度场 $v_t$，对精度最敏感，目前保持 FP。
> 如需扩展，应把遍历对象从 `vlm_with_expert` 提到 `VLAFlowMatching` 顶层，并用 `include/exclude` 收敛范围。

**`module_id` 命名约定**（稀疏度 / 计算量 CSV 与精度 `overrides` 的 selector 都用它）：

| 形态 | 示例 |
|---|---|
| VLM / Expert Linear | `vlm.layers.3.self_attn.o_proj`、`expert.layers.7.mlp.down_proj` |
| Vision Linear | `vision.layers.0.self_attn.k_proj`、`vision.layers.5.mlp.fc2` |
| MatMul（注意是单数 `layer`） | `vision.layer.0.qk`、`expert.layer.15.pv` |
| Connector | `connector.layer.0.connector_proj` |

#### 精度与量化方法

`quant/quant_spec.py` 统一描述算子精度；Linear 侧分别有 `a/w/o` 三个 spec，MatMul 侧有 `A/B/O` 三个 spec。

| 维度 | 可选值 / 配置项 | 说明 |
|---|---|---|
| 精度 kind | `int` / `fp` / `bf` / `none` | 权重可用 INT，激活与输出可用 FP8 |
| FP 格式 | `e4m3`（默认）/ `e5m2` / `e2m1`(FP4) / `fp16` / `bf16` | FP4 为 E2M1 幅值最近邻假量化 |
| scale 粒度 | `per_tensor` / `per_component` / `per_layer` / `per_site` | Linear 与 MatMul 分别由 `linear_scale_granularity` / `matmul_scale_granularity` 控制 |
| 权重分组 | `weight_quant_granularity` + `weight_group_size` | — |
| outlier 保护 | `outlier_ratio`（如 `0.01`）+ `outlier_mask` | 超阈值元素走**浮点旁路**，不参与量化 |
| 量化方法 | `pot_ao_outlier`（Linear）/ `pot_fp8_outlier`（MatMul） | `pot_*` 表示 A/O（及 MatMul 的 A/B/O）scale 强制为 **2 的幂（PoT）**；**整数权重 scale 保持连续校准值，不强制 PoT** |
| mixed precision | `quantization.linear.overrides[]` | 每条含 `name` + `target`（`component` / `module_id` / `module_type`）+ `config`，按顺序匹配 |
| 校准策略 | `calibration_policy`（`reuse` / `recalibrate`），可全局及按组件/模块设置 | 新增精度必须 `recalibrate` |

> **当前主配置（Phase K）**：Vision+VLM Linear 用 **WINT8**、Expert Linear 用 **WINT4**；三组 Linear 的 A/O 与全部 88 个 QK/PV 的 A/B/O 用 **FP8 E4M3**；`outlier_ratio=0.01`、权重 `per_tensor` / scale `per_site`。全量开启时 384 sites 对应 `888 + 264 = 1152` 个 scale 文件。
>
> 注意：`WINT8` / `WINT4` 只描述**权重**；**不能把该配置称为「所有 scale 均 PoT」**（296 个 `w_scale` 是连续值）。

#### 稀疏度与计算量统计口径

统计由 `QuantStatManager` 在 `quant_forward` 前向中随算收集；phase / flow step 由 `runtime_context.py` 自动打标（`phase ∈ {prefill, denoise}`；Vision/VLM 记 `prefill / flow_step=-1`，Expert 记 `denoise / flow_step=0..9`）。

| 维度 | 口径 | 输出 |
|---|---|---|
| runtime element sparsity | `zero_elements / total_elements` | `sparsity/module_sparsity.csv`、`sparsity/workload.csv` |
| runtime bit sparsity | **S\|MMM v1**：E4M3 原始编码为 `S EEEE MMM`，只统计 `S MMM`，**排除 exponent 与 hidden leading 1**；按 4 bit/element 计 | 同上 |
| static weight sparsity | INT **sign-aware** 口径；INT8 与 INT4 **分开报告** | `sparsity/weight_sparsity_static.csv` |
| outlier 旁路 | 「被保护元素 / 总元素」，按组件与 tensor role 汇总 | `sparsity/outlier_sidepath.csv` |
| 覆盖率审计 | 站点数 / runtime 行 / weight 行 / scale 数与精度路由逐项断言 | `coverage_summary.json` |
| 计算量 | **dense-equivalent** MACs：Linear `out.numel()×in_features`、QK/PV `out.numel()×A.shape[-1]`、Conv `out.numel()×kH×kW×C/groups`；`FLOPs = 2×MACs` | `compute.csv`、`compute_summary.json` |

三条必须遵守的约束：

1. **native 与 reported 两套计数并存**：`*_native` 排除 outlier 浮点旁路人为置零，以反映真实量化路径的稀疏度；`*_reported` 保留含保护的原始计数。跨实验比较必须统一口径。
2. **`workload.csv` 不可直接累加 MACs**——其 Linear `activation/output`、MatMul `A/B/O` 是重复的角色行；正式 FLOPs 只来自 `compute.csv`。
3. **位稀疏率 ≠ 压缩率 ≠ 加速比**，不可乘 FLOPs 得到节省量。

同理，outlier 保护路径仍以浮点执行，本框架的统计结果**不能**解释为「全部运算均为 INT/FP8 硬件实现」，也不用于宣称实际芯片速度或能耗。

---

## 3. 环境说明

项目使用单一 conda 环境：

- **`smolvla_eval`** —— LIBERO / Meta-World 评测主环境（mujoco **3.3.2**，torch 2.7.1+cu118，transformers 5.5.4）。

> 注：此前存在 `smolvla_eval`（mujoco 3.8.1）与 `smolvla_eval_mj332`（3.3.2）两套环境，
> 已于 2026-08-27 合并——删除 3.8.1 环境，将 mj332 改名为 `smolvla_eval`。

依赖与跨机器复现步骤见 [`doc/guides/rebuild_manual.md`](doc/guides/rebuild_manual.md)。

### 3.1 运行环境变量

| 变量 | 用途 |
|---|---|
| `PYTHONPATH` | 需包含 `<repo>/src`；`lerobot` 由 `lerobot_current/` 提供 |
| `CUDA_VISIBLE_DEVICES` | 指定单卡；实验脚本通常接受 `GPU=<idx>` 并转发到该变量 |
| `MUJOCO_GL` | 无头环境必须为 `egl`（仓库脚本已默认设置） |

> **必须先激活环境**：`lerobot` 只安装在 `smolvla_eval` 中。若在 `base` 下直接跑实验脚本，
> `prepare` 这类只依赖 `huggingface_hub` / `yaml` 的阶段会「看似成功」，但 `calibrate`
> 会在 `import lerobot` 处失败，并写出**错误环境的 `packages.txt`**，污染版本证据。
> 开跑前用 `which python` 确认，或让脚本先做 import 断言。
>
> **GPU 独占**：单卡被多个长任务共用时显存与吞吐都会显著恶化。正式成功率评测请在空闲卡上
> 串行运行（实验设置文档中已写明「勿与其他长任务争抢」）。

### 3.2 已知环境噪声

以下警告在各实验 `run.log` 中普遍存在，已确认为**无害**：`torchcodec` 载入失败
（FFmpeg 缺 `libavutil.so.*`，自动回退 pyav，视频关闭时无影响）、HF Hub 未认证警告、
`torch_dtype` deprecated、`lerobot_eval.py` 的 `np.bool` DeprecationWarning、
`robosuite` 缺私有 macro 文件。

---

## 4. 快速开始

### 4.1 环境检查

```bash
conda activate smolvla_eval
cd ~/VLA_tcs2
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4.2 跑实验（推荐路径）

新实验一律走 `experiments/` 规范，每个实验自带 runner、分阶段控制与验收 Gate：

```bash
conda activate smolvla_eval
cd ~/VLA_tcs2
GPU=0 bash experiments/<exp_name>/scripts/run_all.sh
```

`run_all.sh` 的典型流程：

```text
记录 commit / worktree / git status / pip freeze / 源文件 SHA256
  → prepare（固定 checkpoint snapshot，两组共用）
  → calibrate（fresh 校准，产出全部 scale）
  → smoke（task0 × 1 episode 工程预检，不计入正式汇总）
  → 逐 suite / 逐组正式评测（baseline 与 quant）
  → summarize（汇总 + 覆盖率审计）
```

中断后**不要重跑 `run_all.sh`**（存在同名 `outputs/` 或 `scales/` 时会直接拒绝覆盖）。按实验
文档用分阶段命令续跑：

```bash
# baseline / quant 需显式指定 suite；prepare / calibrate / smoke 无 suite 参数
python experiments/<exp_name>/scripts/run_arm.py quant --suite libero_object
python experiments/<exp_name>/scripts/summarize.py
```

各阶段成功后写 `completed.json`，并拒绝覆盖已存在的目录。

### 4.3 传统单配置入口（Phase D/E 时期，仅用于历史 config）

```bash
python main.py --config configs/experiments/smolvla_fp_baseline.yaml   # FP 基线
python main.py --config configs/experiments/smolvla_int8.yaml          # 校准 + 评测
python main.py --config configs/experiments/smolvla_int8.yaml --skip-calibration
python main.py --config configs/experiments/smolvla_int8.yaml --skip-evaluation
```

> 注意：量化开启时 `main.py` 默认走 `quant_forward`，因此**不能**用它当 raw 基线；
> 基线请用实验自己的 `run_arm.py baseline`（强制 `raw` + 同一条显式 attention 路径）。

### 4.4 新建实验

```bash
bash experiments/new_experiment.sh 2026-09-27 phaseL my-topic
```

脚手架会建好 `configs/`、`scripts/`、`docs/` 骨架与 `outputs/<exp_name>/`。开跑前必须填写
`docs/experiment_setup.md`，尤其是「实验变量与分组」与「输出目录映射」两节。回填约定：
原始数据留在 `outputs/`，`results.md` 记指标与结论、`logs.md` 记过程与异常；原则上**不向
`docs/` 复制原始大 CSV**（个别实验为便于独立复核会另附证据副本，以各实验文档为准）。详见
[`experiments/README.md`](experiments/README.md)。

---

## 5. 文档导航

| 主题 | 位置 |
|---|---|
| 项目完整背景、结论与交接 | [`doc/guides/handoff.md`](doc/guides/handoff.md) |
| ★ 量化 + 稀疏度研究总手册（口径 / 公式 / 阶段结论） | [`doc/manual/SmolVLA_quantized_sparsity_research_manual.md`](doc/manual/SmolVLA_quantized_sparsity_research_manual.md) |
| 新设备环境重建手册 | [`doc/guides/rebuild_manual.md`](doc/guides/rebuild_manual.md) |
| Meta-World 评估手册 | [`doc/manual/metaworld_manual.md`](doc/manual/metaworld_manual.md) |
| 每日工作日志 | [`doc/logs/`](doc/logs/) |
| 实验管理规范（新实验入口） | [`experiments/README.md`](experiments/README.md) |
| 实验索引（Phase F→K） | 本文件 §6 |
| 硬件建模框架 | [Phase J 实验设置](experiments/2026-09-22_phaseJ_hardware-framework/docs/experiment_setup.md) |
| 结果汇总与对比分析 | [`outputs/reports/`](outputs/reports/) |
| 图表 | [`outputs/figures/`](outputs/figures/) |
| Table-2 严格复现审计记录 | [`outputs/table2_repro_audit/`](outputs/table2_repro_audit/) |

---

## 6. 实验索引

每个实验的权威描述在其 `docs/experiment_setup.md`，状态、结论与异常在其 `docs/results.md`
与 `docs/logs.md`；原始产出在 `outputs/<同名目录>/`（不入库）。下表仅做导航。

| 实验 | 主题 | 状态 |
|---|---|---|
| [`2026-09-08_phaseF_fp8pot-foursuite`](experiments/2026-09-08_phaseF_fp8pot-foursuite/docs/results.md) | outlier 保护的贡献（F1 vs F2）与 INT4 权重的跨 suite 代价（F1 vs F3），四 suite × 400 ep | done |
| [`2026-09-10_phaseG_w4-root-cause`](experiments/2026-09-10_phaseG_w4-root-cause/docs/results.md) | 把 W4 掉点分解到 component / quant-grid / closed-loop / outlier-mask | done |
| [`2026-09-13_phaseH_accuracy-preserving-sparsity`](experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/docs/results.md) | 精度保持前提下的稀疏度：FP 代码审计与 W4 的稀疏增益 | done |
| [`2026-09-15_phaseI_vision-quantization`](experiments/2026-09-15_phaseI_vision-quantization/docs/results.md) | Vision Encoder 逐段引入量化（V0…V5）与 VLIN 全量 72 Linear | running（V4–V5 待跑） |
| [`2026-09-15_sparsity-ratio-quickscan`](experiments/2026-09-15_sparsity-ratio-quickscan/docs/results.md) | 4 组配置的 element / bit 稀疏度 workload characterization | done |
| [`2026-09-22_quickscan_smmm-bit-sparsity`](experiments/2026-09-22_quickscan_smmm-bit-sparsity/docs/results.md) | 切换到 S\|MMM 位口径后的重测（与旧 `1.MMM` 口径对照） | done |
| [`2026-09-22_phaseI_vision-smmm-sparsity`](experiments/2026-09-22_phaseI_vision-smmm-sparsity/docs/results.md) | S\|MMM 口径扩展到含 Vision 的 VLIN workload | done |
| [`2026-09-22_phaseJ_hardware-framework`](experiments/2026-09-22_phaseJ_hardware-framework/docs/results.md) | 硬件建模首版：算子事件采集 + dense GEMM 解析估算 + JSONL 重放 | 软件单元验证完成；GPU rollout 待运行 |
| [`2026-09-23_phaseI_vision-attention-smoke`](experiments/2026-09-23_phaseI_vision-attention-smoke/docs/results.md) | Vision 12 层 QK/PV 接入验证（raw 等价性 + 单 episode 全链路） | done |
| [`2026-09-25_phaseI_vision-joint-smoke`](experiments/2026-09-25_phaseI_vision-joint-smoke/docs/results.md) | Vision 72 Linear + 24 QK/PV 联合启用的一次跑通验证 | done |
| [`2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal`](experiments/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/docs/results.md) | 混合精度（Vision+VLM W8 / Expert W4）Goal 100 ep 的精度与稀疏度 | done |
| [`2026-09-25_phaseK_mixed-full-libero`](experiments/2026-09-25_phaseK_mixed-full-libero/docs/experiment_setup.md) | 同一配置扩展到四 suite（各 400 ep）+ 元素/位稀疏度 + dense-equivalent 计算量 | running |

### 6.1 硬件建模（Phase J）

通过配置 `hardware.enabled: true` 在评测阶段采集注册的 Linear / QuantizedMatMul；输出
`hardware/trace.jsonl`、`operators.csv` 和 `summary.json`。量化 outlier 等未建模路径明确标为
unsupported。完整接口、公式、覆盖边界与运行命令见
[硬件框架实验设置](experiments/2026-09-22_phaseJ_hardware-framework/docs/experiment_setup.md)。

---

## 7. 修订记录

| 日期 | 修改内容 |
|---|---|
| 2026-09-26 | 校正量化范围（QK/PV MatMul 已实现、384 sites 清单、精度矩阵、`module_id` 命名）；新增稀疏度/计算量统计口径与实验索引；修正环境变量说明与失效文档链接 |
