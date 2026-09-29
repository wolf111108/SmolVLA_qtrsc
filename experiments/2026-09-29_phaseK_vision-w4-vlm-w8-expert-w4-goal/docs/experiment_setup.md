# 实验设置（Experiment Setup）

- **实验名称**：2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal
- **状态**：draft
- **创建日期**：2026-09-29
- **负责人**：无
- **相关前序实验**：[Vision/VLM W8 + Expert W4 Goal](../../2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/docs/experiment_setup.md)

## 1. 实验目的

将前序配置的 Vision Encoder Linear 从 WINT8 降为 WINT4，保留 VLM WINT8、Expert WINT4，测试 LIBERO Goal SR。主变量只有 Vision Linear 权重位宽；校准沿用同一协议，但使用独立新 scale 目录，不能冒充复用了相同 scale。

本次只跑量化 SR，不重跑 baseline，不收集 runtime/static 稀疏度或计算量；保留精度 manifest、scale 审计与轻量模块调用计数，以确认量化实际执行。

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| 环境 | 沿用当前 Phase K 成功运行的本地环境 |
| 仓库 commit / 修改 | 启动时保存 commit.txt、worktree.patch、git_status.txt |
| 依赖 | 启动时保存 packages.txt；MuJoCo、LeRobot、torch 保持与前序一致 |
| GPU | `GPU=0` 等显式指定；请选空闲 GPU |

prepare 保存 src 与本实验 scripts/configs 的 SHA256，以及 resolved YAML 哈希；后续每 stage 校验。当前其他实验的 scripts/configs 与 src 不作修改，因此不会破坏其运行指纹。

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| checkpoint | lerobot/smolvla_libero，revision `31d453f7edd78c839a8bbc39744a292686daf0de` |
| processor | 固定完整 checkpoint snapshot，一并加载 normalizer/processor |
| 校准 | HuggingFaceVLA/libero，v3.0；8 episodes，batch=1，frame_stride=4，seed=42 |
| scale | 独立 fresh 校准，1152 个 scale；`scales/2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal/quant/` |
| 评测 | LIBERO Goal，显式 task_ids 0–9 |

## 4. 评测协议

- 正式：10 tasks ×10 episodes =100 episodes，seed=1000。
- 预检：Goal task0 ×1 episode，单独目录，不计入正式 SR。检验执行链路与覆盖，不要求预检必定成功才测正式 SR。
- chunk_size=50，n_action_steps=10，num_steps=10。
- eval.batch_size=1，max_parallel_tasks=1，use_async_envs=false；视频关闭。
- 保持 image/image2 → camera1/camera2 rename，输入与动作处理沿用当前框架。
- Vision explicit_eager；VLM/Expert injected_eager。不要将历史原生 attention baseline 与本实验视为严格同后端对照。

## 5. 实验变量与分组

| 组件 | Linear W | Linear A/O | QK/PV A/B/O | Linear 方法 |
|---|---|---|---|---|
| Vision Encoder | INT4 | E4M3 PoT | E4M3 PoT | pot_ao_outlier |
| VLM | INT8 | E4M3 PoT | E4M3 PoT | pot_ao_outlier |
| Expert | INT4 | E4M3 PoT | E4M3 PoT | pot_ao_outlier |

统一 outlier_ratio=0.01、weight_quant_granularity=per_tensor、scale_granularity=per_site；MatMul 方法 pot_fp8_outlier。

**PoT 仅指 Linear A/O、MatMul A/B/O 的 scale。296 个整数权重 scale 保持连续校准值，不要求 PoT；其余 856 个 scale 必须为 2 的幂。** Vision Encoder 量化范围是 72 Linear +24 QK/PV；patch embedding、connector、动作投影仍为 raw，不宣称 encoder 的所有算子都采用 INT4。

参考组不重跑：前序 Goal raw eager **87/100**、Vision W8/VLM W8/Expert W4 **83/100**。这些是历史参照，不是本轮测量；单 seed 与独立校准限制因果归因。本实验不自动合并历史原始数据，也不把独立校准误写为 scale 完全相同。

## 6. 运行命令

在仓库根目录、现有本地环境：

```bash
GPU=0 bash experiments/2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal/scripts/run_all.sh
```

顺序：prepare → calibrate → smoke → quant（100 ep）→ summarize。

整轮脚本拒绝覆盖已有 outputs/scales。中断续跑时保留已完成 stage，将失败 stage 整体移到本轮 outputs/failed/ 下，然后在同一环境和源代码下运行尚未完成阶段，例如：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
EXP=experiments/2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal
python "$EXP/scripts/run_arm.py" quant
python "$EXP/scripts/summarize.py"
```

不要因为 quant 中断就重新校准；calibrate 自身失败时，应整体归档其输出与 scale 目录，再从 calibrate 重新开始。不得混合不同代码/配置的阶段输出。

## 7. 输出目录映射

根：`outputs/2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal/`。

| 阶段 / config | 输出 | 状态 |
|---|---|---|
| prepare / quant.yaml | prepared.json、quant_resolved.yaml、版本信息 | pending |
| calibrate / quant.yaml | calibrate/scale_audit.json、completed.json | pending |
| smoke / quant.yaml | smoke/ | pending |
| quant / quant.yaml | quant/ | pending |
| summarize | summary.json、task_success.csv | pending |

smoke/quant 各含 config.yaml、result.json、eval_info.json、execution.json、quantization_manifest.csv、scale_audit.json、completed.json。Gate 验证 384 个位点（296 Linear +88 MatMul），W4/W8/W4 路由、FP8 格式及 PoT scale，所有位点实际被调用；正式结果必须有完整 10×10 episodes。每次评测 scale 哈希与本次校准一致。

按仓库规范原始数据保存在 outputs；结果写入 docs/results.md，进度与环境写入 docs/logs.md，不向 docs 复制原始大文件。

## 8. 风险与注意事项

- 一次 100ep 只能提供本协议的 SR 观测，不足以证明无损或显著退化。
- 基线后端漂移与 Vision W4 的影响需区分，优先参照前序相同 eager 路径。
- 不启用稀疏/计算量收集，以减少 SR 实验开销；轻量覆盖 hook 只计调用数，不保存张量。
- 不改变正在运行的其他实验。只新增本实验目录。
- 创建阶段不安装 PyTorch、不运行模型；真实校准、smoke 和 SR 由本地执行。
