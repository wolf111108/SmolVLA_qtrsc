# 实验设置（Experiment Setup）

- **实验名称**：2026-09-25_phaseK_mixed-full-libero
- **状态**：draft
- **负责人**：无
- **创建日期**：2026-09-25
- **相关前序实验**：[Goal 联合量化](../../2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/docs/experiment_setup.md)（87/100 → 83/100）

## 1. 实验目的

沿用当前 Vision+VLM W8、Expert W4、AFP8 PoT 配置，扩展至标准 LIBERO 四 suite，回答：各 suite 成功率如何、量化主路径的元素/位稀疏度如何、实际 rollout 的算法计算量如何分布。

四 suite 是 Spatial、Object、Goal、Long（`libero_10`），40 tasks；不包含训练任务集合 `libero_90`。基线和量化组分别 400 episodes；额外 1 episode 为 smoke，不纳入正式汇总。各 suite 在独立 Python 进程评测，统计不会跨 suite 污染。

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | 使用此前 Goal 成功运行的本地环境 |
| 仓库 revision / commit | 开跑时记录 `outputs/<实验名>/commit.txt`、`worktree.patch`、`git_status.txt` |
| LeRobot 路径与 commit | 开跑前在 logs.md 登记，沿用前序版本 |
| GPU | 本地指定 `GPU`，串行运行，勿与其他长任务争抢 |
| 关键依赖版本 | `packages.txt` 自动记录；沿用前序 MuJoCo/torch/transformers 版本 |

prepare 保存 `src/**/*.py`、本实验 scripts/configs 的 SHA256；每个后续 stage 校验指纹和 resolved YAML 哈希。可以回填 docs，但修改运行代码/配置必须开始新的一轮，不能混用已有输出。

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| checkpoint | `lerobot/smolvla_libero`；legacy_public；revision `31d453f7edd78c839a8bbc39744a292686daf0de`，与前序 Goal 完全相同 |
| 加载方式 | prepare 下载/解析完整 snapshot，包含 processor/normalizer，所有组使用同一 resolved 本地路径 |
| 评测 | libero_spatial / libero_object / libero_goal / libero_10，各 task 0–9 |
| 校准 | HuggingFaceVLA/libero v3.0；8 episodes，batch=1，frame_stride=4，seed=42 |
| scale | 独立 fresh 校准一次，四 suite 共用；`scales/<实验名>/quant/`；1152 个文件 |

禁止按 suite 重校准。各次量化运行的 scale 值与文件 SHA256 必须和 calibrate/scale_audit.json 相同。296 个整数权重 scale 为连续校准值；856 个 Linear A/O 与 MatMul A/B/O scale 必须为正且为 2 的幂。**WINT8/WINT4 不代表权重 scale 也采用 PoT。**

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | 4 suites × 10 tasks × 10 episodes，seed=1000；每组 400 episodes |
| 动作参数 | chunk_size=50，n_action_steps=10，num_steps=10 |
| 并行 | eval.batch_size=1，max_parallel_tasks=1，use_async_envs=false |
| 输入与归一化 | 保留前序 image/image2 → camera1/camera2 rename；加载 checkpoint processor |
| 视频 | 关闭 |
| backend | 两组均 Vision explicit_eager、VLM/Expert injected_eager |
| 指标 | 每 task/suite/整体 SR；逐 episode 成败翻转；runtime/static sparsity；dense-equivalent MACs/FLOPs |

这是当前量化协议的四 suite 扩展，**不是 n_action_steps=1 的严格论文 Table 2 复现**。不要与早期 n_action_steps=1 或不同 MuJoCo 版本的基线直接相减。

### 稀疏统计

- Runtime：native zero/total、native sparse_bits/total_bits，排除 protected FP sidepath 制造的人工零。FP8 位口径为 **S|MMM（符号+尾数，排除 exponent/hidden-1）**。
- 分 Vision、VLM、Vision+VLM pooled、Expert；再分 Linear A/O 与 QK/PV A/B/O。保留逐模块、phase、flow_step 明细。
- Static weight：INT8/INT4 sign-aware 口径分开报告。四 suite 的静态计数必须一致；整体统计仅取一份，不累加四次。
- 汇总先累加整数分子/分母，再求比率，不平均模块/suite 百分比。
- Outlier ratio 配置为 1%，实际被保护元素比例另表输出；MatMul B 是 Kᵀ/V 激活，不是静态权重。

### 计算量统计

实验级 `compute.py` 对每次真实执行的 nn.Linear（含 QuantizedLinear）、nn.Conv2d 和 QuantizedMatMul 挂一次 hook，仅读取 shape，不复制张量。只在 rollout 期间启用，不包含校准。

| 算子 | MACs / 单次调用 |
|---|---|
| Linear | `output.numel() × in_features` |
| QK/PV | `output.numel() × A.shape[-1]`（包含实际 batch/head 广播结果） |
| Conv2d / patch embedding | `output.numel() × kernel_H × kernel_W × in_channels/groups` |

`FLOPs = 2 × MACs`。覆盖 Vision、VLM、Expert，以及未量化的 patch embedding、connector、action/state/time 投影（后者归 other，保留完整模块路径）；导出未调用的模块清单。pixel shuffle 是数据重排，没有 MAC。

这是**算术密集算子的 dense-equivalent 算法计算量**：不含 bias、softmax、norm、elementwise、embedding lookup、数据搬运、量化/统计开销、outlier 实现额外 GEMM；不是 GPU 实际指令数，也不是稀疏后剩余计算量或硬件加速比。

每组输出 rollout 总量、各组件 FLOPs 占比、每 episode、每次完整 sample_actions generation 的平均量。一次 generation 已包括全部 10 次 denoise；不再乘 10，也不把 n_action_steps=10 当作 generation 次数。两组成功/失败轨迹长度可能不同，应同时看总量和每 generation 均值。

`workload.csv` 仍保留作诊断，但其 Linear activation/output、MatMul A/B/O 是重复的角色行，**禁止直接累加该文件的 MACs**。正式 FLOPs 仅来自 `compute.csv`。

## 5. 实验变量与分组

| 组 | config | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| B0 × 四 suite | baseline.yaml | raw | checkpoint、seed、eager、动作参数 | 保留 384 wrappers；收集 shape 计算量，不收集量化稀疏度 |
| M0 × 四 suite | quant.yaml | Vision/VLM Linear WINT8，Expert Linear WINT4；全部 Linear A/O 和 QK/PV A/B/O E4M3 PoT | 同上 | outlier_ratio=.01，per_tensor W、per_site scales；收集完整稀疏度及计算量 |

connector/patch embedding/action projections 保持 raw。新增脚本不改变框架量化前向。

Gate：384 sites（296 Linear、88 MatMul）；每正式量化 suite 3736 条唯一 runtime 稀疏行、296 条静态权重；1152 scales；所有 10 tasks ×10 episodes 完整；compute 覆盖所有量化位点/flow step，调用数与每个 sparsity role 一致；Vision 未量化部分、connector、other 必须有实际调用。smoke 先走相同 Gate。

## 6. 运行命令

在仓库根目录、前序本地环境执行：

```bash
GPU=0 bash experiments/2026-09-25_phaseK_mixed-full-libero/scripts/run_all.sh
```

执行顺序：prepare → fresh calibrate → Goal task0 ×1ep smoke → 按 Spatial/Object/Goal/Long 依次 baseline、quant → summarize。已有输出/scale 时拒绝整轮覆盖。

若中断，仅在**同一环境、同一代码/配置/scale**下续跑。已完成 stage 保留；失败 stage 的完整目录先移到本轮输出下的 `failed/`（不得只删报错标记），然后从失败 stage 继续。不要重复 prepare/calibrate。示例：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
EXP=experiments/2026-09-25_phaseK_mixed-full-libero
# 举例：Object quant 中断，归档其失败目录后重跑，再完成余下组。
python "$EXP/scripts/run_arm.py" quant --suite libero_object
python "$EXP/scripts/run_arm.py" baseline --suite libero_goal
python "$EXP/scripts/run_arm.py" quant --suite libero_goal
python "$EXP/scripts/run_arm.py" baseline --suite libero_10
python "$EXP/scripts/run_arm.py" quant --suite libero_10
python "$EXP/scripts/summarize.py"
```

独立 stage 使用同样 CLI：`prepare` / `calibrate` / `smoke` 无 suite 参数；`baseline` / `quant` 必须显式 `--suite`。stage 成功后写 completed.json；存在目录即拒绝覆盖。长时间预算请按前序 Goal 的基线约 79min、量化约 182min，再考虑 suite 轨迹长短估算；不是一小时 quickscan。

## 7. 输出目录映射

根目录为 `outputs/2026-09-25_phaseK_mixed-full-libero/`。

| config / 阶段 | 相对输出目录 | 状态 |
|---|---|---|
| prepare | prepared.json、baseline_resolved.yaml、quant_resolved.yaml、版本记录 | pending |
| quant.yaml / calibrate | calibrate/ | pending |
| quant.yaml / smoke | smoke/ | pending |
| baseline.yaml / Spatial | libero_spatial/baseline/ | pending |
| quant.yaml / Spatial | libero_spatial/quant/ | pending |
| baseline.yaml / Object | libero_object/baseline/ | pending |
| quant.yaml / Object | libero_object/quant/ | pending |
| baseline.yaml / Goal | libero_goal/baseline/ | pending |
| quant.yaml / Goal | libero_goal/quant/ | pending |
| baseline.yaml / Long | libero_10/baseline/ | pending |
| quant.yaml / Long | libero_10/quant/ | pending |

每 stage 保存 config.yaml、result.json、execution.json、compute.csv、compute_summary.json、completed.json；quant 另保存 scale_audit.json、coverage_summary.json 与 sparsity/ 下的五张原始 CSV。

根汇总：summary.json、success_summary.csv、task_success.csv（40 tasks +成败翻转）、sparsity_summary.csv、compute_summary.csv、outlier_summary.csv。整体静态权重取一份，runtime 计数跨四 suite 累加，计算量均值按 generation 加权。

按仓库规范：原始数据留在 outputs 同名目录；docs/results.md 回填表格并链接原始输出，docs/logs.md 回填版本、运行进度与失败记录，不向 docs 复制原始大 CSV。

## 8. 风险与注意事项

- 单 seed、每 task 10 episodes，不能由一次结果推断自然波动或统计显著性。成功数相同不代表逐 episode 相同，task_success.csv 明确记录双向翻转。
- 原始配置与前序一致；整数 W scale 不强制 PoT，不应写成所有 scale 均 PoT。
- S|MMM 与整数 sign-aware 指标不同；位稀疏率不等于压缩率或加速比，也不可直接将其乘 FLOPs 得到节省量。
- 这是模拟量化；本实验耗时包含统计开销，不用于宣称 INT8/INT4 硬件速度。
- 本地模型 smoke 验证由执行者完成。创建阶段仅做语法检查、标准库形状计数/汇总验证，不安装或下载 PyTorch。
