# VLA-TCS2

面向 VLA（Vision-Language-Action）模型的**量化评测与硬件建模框架**，首个 workload 为 **SmolVLA** + **LIBERO** 机器人仿真基准（已扩展 Meta-World 作为第二基准）。

> 本 README 只介绍项目背景与架构；实验日志与结果数据分别在 [`doc/logs/`](doc/logs/)、[`outputs/`](outputs/) 下，不在此重复。

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
├── main.py                  # 量化流水线入口（load config → build → calibrate → eval）
├── pyproject.toml           # Python 包配置（setuptools, src 布局）
├── src/vla_tcs2/            # 量化框架源码
├── configs/experiments/     # 实验 YAML 配置
├── scripts/                 # LIBERO 评测 / 审计 / 汇总脚本
├── checkpoints/             # 本地模型 checkpoint（大文件，不纳入文档）
├── experiments/             # ★ 实验管理（新实验统一在此建子目录，见 experiments/README.md）
├── outputs/                 # 评测原始产出（eval_info.json / videos / reports / figures）
├── doc/                     # ★ 项目文档（见下，含每日工作日志 doc/logs/）
├── lerobot_current/         # 主 LeRobot（LIBERO 评测环境，固定 commit）
├── lerobot/                 # 旧 LeRobot（论文时期架构研究用）
└── envs/                    # 环境冻结文件（pip freeze / conda yml）
```

### 2.2 文档目录 `doc/`

```text
doc/
├── guides/
│   ├── handoff.md           # 项目交接文档（最完整的背景与结论）
│   └── rebuild_manual.md    # 新设备环境重建手册（含 84% 验收标准）
├── logs/                    # 每日工作日志（按日期命名）
└── manual/
    └── metaworld_manual.md  # Meta-World 评估手册（第二基准接入）
```

### 2.3 量化框架源码 `src/vla_tcs2/`

| 模块 | 职责 |
|---|---|
| `model_wrapper.py` | 加载预训练 SmolVLA policy，将选中的 `nn.Linear` 替换为 `QuantizedLinear`，注入注意力 `QuantizedMatMul`（QK/PV），并提供 `switch_quantization_mode_all` 模式切换 |
| `quant_linear.py` | 量化线性层，支持 `raw` / `scale_inspection` / `quant_forward` 三种模式，INT / FP8 / FP4 灵活精度 |
| `quant_matmul.py` | 量化矩阵乘（attention 的 `Q@K^T` 与 `P@V`），三种模式 + outlier 保护 + mixed-precision |
| `calibration.py` | 校准流程：判定 reuse/recalibrate → 绑定 `QuantStatManager` → `scale_inspection` 前向 → 保存 pickle scale |
| `eval.py` | 进程内 LIBERO 评测（复用 lerobot `eval_policy_all`，支持量化模型直接评测） |
| `quant/` | 量化基础设施子包：`quant_spec`（精度规格）、`utils`（STE/scale 工具）、`stat_manager`（scale 统计与落盘） |
| `hardware/` | 硬件建模代理（预留） |
| `stats/` | 运行时统计（预留） |

### 2.4 量化流水线

```text
load config (YAML)
      ↓
ModelWrapper.build()        # 加载 policy + 替换 Linear/MatMul
      ↓
set_mode("scale_inspection")# 校准模式，收集 scale
      ↓
calibrate()                 # 前向采样 → save_all_scales (pickle)
      ↓
set_mode("quant_forward")   # 量化推理（或 raw 浮点基线）
      ↓
evaluate()                  # LIBERO rollout → success rate
      ↓
save result.json
```

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

#### 当前量化范围（已实现）

`_wrap_smolvla_linear_layers`（`model_wrapper.py`）目前只量化以下 **Linear 层**：

| 模块 | 是否量化 | 说明 |
|------|---------|------|
| VLM `text_model` 的 `q/k/v/o_proj` + `gate/up/down_proj` | ✅ 量化 | 16 层 LLM 的 self-attn 与 MLP |
| `lm_expert` 的 `self_attn` + `mlp` | ✅ 量化 | 16 层 action expert |
| SigLIP `vision_model` | 🔲 可选（opt-in） | 视觉编码器；`quantization.vision.enabled=true` + `vision.linear.{mlp,attn_proj}=true` 才开启，默认全关闭 |
| `connector` | 🔲 可选（opt-in） | 视觉模态投影；`quantization.connector.enabled=true` 开启，默认关闭 |
| action head（`state/action_in/action_out_proj` + `action_time_mlp_*`） | ❌ 未量化 | 位于 `VLAFlowMatching` 顶层，当前 `_wrap` 未触达 |
| QK^T / PV matmul | ❌ 未量化 | 默认 `quantize_matmul: false` |

> 注：action head 是 flow matching 的**最终输出层**，直接产生速度场 $v_t$，对精度最敏感，
> 目前保持 FP。后续如需完整量化，可将 `_wrap` 的遍历对象从 `vlm_with_expert` 扩展到
> `VLAFlowMatching` 顶层，并用 `include/exclude` 控制 vision encoder / action head。

> Vision/Connector 量化（Phase I，默认关闭）：`_wrap_smolvlm_vision_linear_layers`
> 支持对 SmolVLM 视觉编码器（`vision_model.encoder.layers[i].self_attn.{q,k,v,out}_proj`
> 与 `.mlp.{fc1,fc2}`，12 层共 72 个 Linear）和 connector 投影
> （`connector.modality_projection.proj`，`module_id=connector.layer.0.connector_proj`）
> 做 opt-in 量化。二者由 `quantization.connector.enabled` 与 `quantization.vision.*`
> 独立控制，默认 `false`（保持既有 VLM/Expert 224 Linear 不变）。
>
> **关键 gate**：`vision.enabled=true` 本身**不会** wrap 任何 Linear，必须再显式打开
> `vision.linear.mlp`（24 个 fc1/fc2）和/或 `vision.linear.attn_proj`（48 个
> q/k/v/out_proj），两者默认 `false`。precision 通过 `quantization.linear.overrides`
> 的 `component`/`module_id` selector 区分；vision/connector 首次加入默认
> `calibration_policy=recalibrate`。详见
> `2026-09-15_phaseI_vision_quantization_experiment_manual.md`。

---

## 3. 环境说明

项目使用单一 conda 环境：

- **`smolvla_eval`** —— LIBERO / Meta-World 评测主环境（mujoco **3.3.2**）。

> 注：此前存在 `smolvla_eval`（mujoco 3.8.1）与 `smolvla_eval_mj332`（3.3.2）两套环境，
> 已于 2026-08-27 合并——删除 3.8.1 环境，将 mj332 改名为 `smolvla_eval`。

依赖与跨机器复现步骤见 [`doc/guides/rebuild_manual.md`](doc/guides/rebuild_manual.md)。

---

## 4. 快速开始

### 4.1 环境检查

```bash
conda activate smolvla_eval
cd ~/VLA_tcs2
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4.2 运行 FP 基线（不量化，验证 eval 链路）

```bash
python main.py --config configs/experiments/smolvla_fp_baseline.yaml
```

### 4.3 运行 int8 量化（校准 + 评测）

```bash
# 首次：完整校准 + 评测
python main.py --config configs/experiments/smolvla_int8.yaml

# 复用已有 scale 文件（跳过校准）
python main.py --config configs/experiments/smolvla_int8.yaml --skip-calibration

# 只校准，跳过评测
python main.py --config configs/experiments/smolvla_int8.yaml --skip-evaluation
```

---

## 5. 文档导航

| 主题 | 位置 |
|---|---|
| 项目完整背景、结论与交接 | [`doc/guides/handoff.md`](doc/guides/handoff.md) |
| 新设备环境重建手册 | [`doc/guides/rebuild_manual.md`](doc/guides/rebuild_manual.md) |
| Meta-World 评估手册 | [`doc/manual/metaworld_manual.md`](doc/manual/metaworld_manual.md) |
| 每日工作日志 | [`doc/logs/`](doc/logs/) |
| 实验管理规范（新实验入口） | [`experiments/README.md`](experiments/README.md) |
| 结果汇总与对比分析 | [`outputs/reports/`](outputs/reports/) |
| 图表 | [`outputs/figures/`](outputs/figures/) |
| Table-2 严格复现审计记录 | [`outputs/table2_repro_audit/`](outputs/table2_repro_audit/) |
