# Phase L — Goal INT8 activation / mixed INT4 weight comparison

## 目的

在同一 LIBERO Goal 协议下比较浮点 baseline、VLM/Expert A8W8O8、VLM A8W8O8 + Expert A8W4O8。两组量化均覆盖 VLM 与 Expert 的 QK/PV，并分别报告两个组件的 activation bit sparsity 和 weight bit sparsity。

## 环境与版本

- 使用仓库已有 SmolVLA / LIBERO CUDA 环境，安装依赖见根目录与 `doc/guides/rebuild_manual.md`。
- `run_all.sh` 记录 Git commit、worktree patch、`pip freeze`、源配置与完整日志。
- `prepare` 保存框架源码、LeRobot 源码、实验 Python 脚本及 YAML 的 SHA256；后续阶段及汇总拒绝源码/配置变化。
- 创建时仅执行 CPU 配置/统计回归测试。当前开发环境无 PyTorch、CUDA GPU；GPU 模型加载、校准和 rollout 尚未验证。

## 模型与数据

- 模型：`lerobot/smolvla_libero`，revision 固定为 `31d453f7edd78c839a8bbc39744a292686daf0de`。下载完整 snapshot，包括 processor/normalizer；三组共享该绝对路径。
- 校准：`HuggingFaceVLA/libero`，revision `v3.0`，按 seed 42 抽取 8 个 demonstrations，frame stride 4，batch size 1。
- 两个量化组使用相同抽样协议，分别 fresh calibration；独立 scale 目录，禁止复用彼此或历史实验尺度。数据集 `v3.0` 为版本标签，未进一步锁定 commit。

## 评测协议

- `libero_goal`，task IDs 0–9，每任务 10 episodes，每组 100 episodes；共 300 个正式 episodes。
- seed 1000，batch size 1，同步环境，`max_parallel_tasks=1`，不录视频。
- `n_action_steps=10`，`num_steps=10`。
- 先分别完成两组校准，再执行三组 task 0 × 1 episode smoke；通过后运行三组正式评估。smoke 与正式统计使用不同进程、不同目录。
- baseline 同样包装 Linear / 注入 QK/PV eager attention，显式 `raw` 模式；与量化组保持相同 attention 实现和预处理。
- 成功率从逐 episode boolean successes 累加，与 overall 校验；输出总成功率、逐任务成功率、相对同轮 baseline 的百分点差值。

## 实验变量与分组

| Arm | VLM Linear A/W/O | Expert Linear A/W/O | VLM + Expert QK/PV A/B/O | 模式 |
|---|---|---|---|---|
| `baseline` | 原浮点 | 原浮点 | 原浮点 | raw |
| `a8w8` | INT8 / INT8 / INT8 | INT8 / INT8 / INT8 | INT8 / INT8 / INT8 | quant_forward |
| `a8w4_expert` | INT8 / INT8 / INT8 | INT8 / INT4 / INT8 | INT8 / INT8 / INT8 | quant_forward |

- 本实验 VLM 指文本 backbone。**Vision tower、connector、action/state/time heads 均保持原浮点**。
- W4 仅作用于 Expert 的 Linear 权重；QK/PV 的 B（K/V）仍为 INT8 激活。
- `method=outlier`，`outlier_ratio=0.01`，采用仓库既有浮点保护旁路；Linear/MatMul 均为 per-site scale，Linear weight 为 per-tensor。A8/W8/O8 指常规量化路径，存在 outlier 浮点旁路。
- 覆盖：VLM 112 Linear + 32 MatMul；Expert 112 Linear + 32 MatMul；共 288 站点、864 个 scale。校准检查尺度有限且为正，eval 前后检查哈希一致。
- 正式与 smoke 均检查 manifest 实际精度；runtime CSV 应有 3520 行（VLM prefill/-1 320 行、Expert denoise 0–9 共 3200 行）；静态权重应有 224 行。

### 稀疏度定义

INT bit sparsity 复用 `QuantStatManager.compute_sparse_stats` 的 **integer sign-aware** 定义：对于 n 位补码，非负值统计 0 位、负值统计 1 位，包含符号位；分母为元素数 × n。每个编码的符号位均贡献一个 sparse bit。该指标与普通 zero-bit rate 分开理解。

| 指标 | 统计对象 | 聚合口径 |
|---|---|---|
| activation | Linear `activation` + QK/PV `A/B` | 使用 `sparse_bits_native / total_bits_native`，先求计数之和再相除 |
| linear_input / qk_inputs / pv_inputs | 上述激活的三个子集 | 分别输出，同样按计数加权 |
| static_weight | 各 Linear 完整权重量化编码 | 每层只统计一次；分母 VLM 8N，Expert 为 8N 或 4N |

- runtime 激活排除 `output/O`；各输入位置按实际调用、token 数、denoise step 累加。相同张量在不同算子输入处出现会分别计入，表示算子输入工作量，不能解释为去重后的存储占用。
- native 分子、分母均扣除被浮点旁路保护的位置：`sparse_bits_native = sparse_bits_reported - 8 × protected_elements`。额外报告实际 FP sidepath 元素比例。
- 每条 runtime 记录必须在 `outlier_sidepath.csv` 中有唯一对应项，且调用数、总元素数、保护元素数一致；缺失旁路记录时拒绝生成报告。
- 静态 weight 语义为 `full_weight_quantized_no_dynamic_outlier_mask`：不扣除运行时动态 outlier mask；QK/PV 的 B 不计为静态 weight。
- 两组件分别输出，不混算 VLM/Expert，不平均层级百分比。baseline 没有整数编码，其 bit sparsity 为 **N/A**。

## 运行命令

在仓库根目录、已有 SmolVLA/LIBERO 环境中运行：

```bash
GPU=0 bash experiments/2026-11-09_phaseL_int8-goal/scripts/run_all.sh
```

完整启动拒绝已有同名 outputs/scales，避免覆盖。`PYTHON=/path/to/python` 可选择解释器，`MUJOCO_GL` 默认 egl。

单阶段续跑示例（保留已完成阶段；失败阶段的输出及未完成校准的 scale 需先归档）：

```bash
python experiments/2026-11-09_phaseL_int8-goal/scripts/run_arm.py eval --arm a8w4_expert
python experiments/2026-11-09_phaseL_int8-goal/scripts/summarize.py
```

CPU 检查（仅需 Python + PyYAML；测试数据均为合成计数）：

```bash
python experiments/2026-11-09_phaseL_int8-goal/scripts/run_arm.py preflight
python experiments/2026-11-09_phaseL_int8-goal/scripts/test_experiment.py
```

## 输出目录映射

所有原始结果位于 `outputs/2026-11-09_phaseL_int8-goal/`：

| 路径 | 内容 |
|---|---|
| `prepared.json`, `*_resolved.yaml` | checkpoint、源码与配置指纹 |
| `calibrate/{a8w8,a8w4_expert}/` | 校准完成标记与 864 个尺度审计 |
| `smoke/{baseline,a8w8,a8w4_expert}/` | 单 episode 验证 |
| `eval/{baseline,a8w8,a8w4_expert}/` | 100 episodes 的 result、执行模式、覆盖验证 |
| `eval/<quant_arm>/sparsity/` | manifest、runtime、static weight、outlier CSV |
| `summary.md`, `summary.json`, `comparison.csv` | 三组总成功率、baseline 差值及组件稀疏度 |
| `task_success.csv` | 30 行逐组逐任务成功率与差值 |
| `sparsity_by_component.csv` | 组件/输入类别/权重分组，含原始计数与旁路比例 |

尺度位于 `scales/2026-11-09_phaseL_int8-goal/{a8w8,a8w4_expert}/`。raw outputs/scales 按仓库规则不入 Git；完成 GPU 运行后将 `summary.md` 的结果补入本实验 `results.md`，同时保留原始产出。

## 风险与注意

- `outlier` 是现有假量化评测路径，有浮点旁路；结果用于精度/编码稀疏度研究，不能直接推算全整数 kernel 速度。
- INT4 与 INT8 编码位宽不同，符号位贡献占比分别为 1/4 与 1/8；bit sparsity 数值不能脱离位宽比较硬件收益。
- 单 seed、每组 100 episodes 的成功率分辨率为 1 个百分点，逐任务为 10 个百分点；未设计多 seed 显著性检验。
- VLM/Expert 的 runtime 激活量由各自闭环轨迹加权，不同成功/失败轨迹长度会改变样本组成。
