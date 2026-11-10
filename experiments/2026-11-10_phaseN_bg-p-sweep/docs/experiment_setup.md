# Phase N — 原码 activation 2-bit group 的 p 扫描

## 目的

沿用 `2026-11-09_phaseL_int8-goal` 的 `a8w4_expert` 配置，在 **LIBERO Goal task 0 × 1 episode** 中统计 p=2、4、8、16、32、64、128。七个 p 在同一次量化推理中统计，避免重复加载模型和不同轨迹的干扰。

参考 Zhao et al., *A 3.51 TOPS/mm² Transformer Accelerator Exploiting Bipolar Sparsity and Approximate Gating*, DOI `10.1109/TCSI.2025.3601717`，Section III-E / Fig. 11。这里只借鉴 p 个 token 共同判断可跳过性的粒度；按用户要求改为原码幅值 BG，不启用论文的 IAG，也不复现其双极补码判定。

## 环境与版本

- 分支 `feat/hardware-timeloop-mapper`，创建时基于 `6a138f3`。
- 运行环境使用已有 `smolvla_eval`、CUDA、LeRobot、LIBERO，见根目录说明。
- CPU 单元测试使用 NumPy 和可选的 PyTorch 2.7.1 CPU；真实 rollout 仍需原项目环境。
- `execution.json` 保存源码 SHA256、实际配置 SHA256、checkpoint revision、Git commit 和依赖版本；统计结束再次验证尺度未改变。

## 模型与数据

- `lerobot/smolvla_libero`，固定 revision `31d453f7edd78c839a8bbc39744a292686daf0de`。
- 优先自动复用 `outputs/2026-11-09_phaseL_int8-goal` 中已经完成且审计一致的 **a8w4_expert** 校准。检查 checkpoint、量化精度、outlier、校准配置、尺度生成源码及全部 864 个 scale 哈希；复制到新输出目录使用。
- 若没有已完成的匹配校准，则按原实验协议校准：`HuggingFaceVLA/libero@v3.0` 抽 8 个 demonstrations，stride=4，seed=42。**不扫描完整校准数据集**；首跑的校准仍有额外成本。
- 不使用随机图像或随机 activation 代替任务实测。

## 评测协议

- Goal task 0，1 episode，seed=1000，batch=1，同步环境、单 task worker，不录视频。
- `n_action_steps=10`、`num_steps=10`，收集 episode 内全部实际调用。
- 只运行混合量化组；不启动 baseline、其他任务或完整 Goal suite。
- 288 站点：VLM 112 Linear + 32 MatMul，Expert 112 Linear + 32 MatMul。必须看到 VLM prefill 与 Expert 的十个 denoise step；缺站点/角色则失败，不生成完成标记。
- 一次 episode 的成功与否仅作运行检查，不作为可靠成功率结论。

## 实验变量与分组

| 部分 | Linear A/W/O | QK/PV A/B/O |
|---|---|---|
| VLM 文本 backbone | INT8/INT8/INT8 | INT8/INT8/INT8 |
| Expert | INT8/INT4/INT8 | INT8/INT8/INT8 |
| Vision / connector / action-state-time heads | 保持浮点 | 保持浮点 |

保留 `method=outlier, outlier_ratio=0.01`、per-site scale、per-tensor weight。配置中的旧 `quantization.*.p=4` 保持原样；本次扫描参数独立放在 `bit_group_probe.p_values`，不改变模型计算。

### BG 编码

对实际参与普通量化路径的整数编码 q，取 `mag = abs(int16(q))`，符号独立处理。四个 BG 为：

| level | 幅值位 | 提取式 |
|---|---|---|
| 0 | [1:0] | `(mag >> 0) & 3` |
| 1 | [3:2] | `(mag >> 2) & 3` |
| 2 | [5:4] | `(mag >> 4) & 3` |
| 3 | [7:6] | `(mag >> 6) & 3` |

严格 8 位原码只有 7 个幅值位，无法表示当前量化器允许的 **-128**。为保持最新实验的数值和精度配置，本统计采用“独立符号 + 8 位幅值”：一般数值的最高幅值位为 0，-128 的幅值为 128，保持精确。`minus128_elements` 单列计数；不对 -128 截断、不偷偷改变量化范围。符号位不计入 BG，也不采用 `~q` 的双极补码近似。

### p 的分组方向

每个固定 channel、固定 BG level 上，每 p 个连续 token 构成一个 pack；只有所有有效普通路径 BG 均为 00 才能跳过该 pack。

| 输入 | 实际布局 | token 维 |
|---|---|---|
| Linear activation | `[batch, tokens, channels]` | -2 |
| QK A (Q) | `[batch, heads, query_tokens, head_dim]` | -2 |
| QK B (Kᵀ) | `[batch, heads, head_dim, key_tokens]` | -1，统计时转置 |
| PV A (P) | `[batch, heads, query_tokens, key_tokens]` | -2 |
| PV B (V) | `[batch, heads, key_tokens, head_dim]` | -2 |

不跨 batch、head、算子调用或 denoise step 拼接 token；不先 flatten 为一维。mask 与 tensor 同步转置。

### 计数与分母

- `native_skip_rate = native_skipped_groups / native_groups`：主指标。分母为有效普通路径中的实际 BG 数；分子为处于可跳过 pack 内的这类 BG 数。排除 FP-protected 位置和补齐 token；尾块只按实际有效元素加权。
- `full_native_pack_zero_rate = full_native_zero_packs / full_native_packs`：严格固定 p 的对照。只计恰有 p 个真实 token、且没有任何 FP-protected lane 的 pack。分母为零时输出空值/`N/A`。
- `normal_pack_zero_rate`：包含 outlier 置零的普通计算路径 pack 零率，仅作对照；不能解释为天然稀疏率。
- `tail_pack_fraction`：不足 p 个 token 的 pack 占比。Expert 部分张量只有约 50 个 token，p=64/128 时其左输入都是尾块；不能把它解释为完整 64/128-token 检测结果。
- 各组件、各层、各调用均先累加整数分子/分母再计算比例，不平均层级百分比。合并四个 BG 时每组独立计一次。
- `bsc_left` 为论文 BSC 对应的左输入：Linear activation + QK/PV A。
- `all_inputs` 额外包含 K/V（QK/PV B），用于与前次实验“全部 activation 输入”的范围对照。K/V 指标属于转回 token-channel 布局后的数据统计，不能直接解释为当前右操作数的硬件调度收益。
- 静态权重、算子输出 O 不计入 activation；不去重重复使用的输入，表示运行期操作数工作量。FP 旁路继续执行，BG 零率不等于整体加速比。
- `by_site.csv` 中计数按 p、level 重复记录，跨 level 聚合后的 `minus128_elements` 等辅助计数也是 BG 加权量；读取元素数应选定一个 p 和 level，避免重复累加。

## 运行命令

仓库根目录，在已有环境中执行：

```bash
conda activate smolvla_eval
GPU=0 bash experiments/2026-11-10_phaseN_bg-p-sweep/scripts/run.sh
```

默认自动验证并复用已完成的 Phase L 同组尺度；否则进行 8-demo 校准。显式选择已完成的历史校准或强制重新校准：

```bash
bash experiments/2026-11-10_phaseN_bg-p-sweep/scripts/run.sh \
  --reuse-from /absolute/path/to/outputs/2026-11-09_phaseL_int8-goal
bash experiments/2026-11-10_phaseN_bg-p-sweep/scripts/run.sh --fresh-calibration \
  --output-dir outputs/2026-11-10_phaseN_bg-p-sweep/fresh-run
```

已有输出目录拒绝覆盖；新运行用 `--output-dir`。可用 `PYTHON=/path/to/python` 指定解释器。

```bash
python experiments/2026-11-10_phaseN_bg-p-sweep/scripts/run.py --preflight
bash experiments/2026-11-10_phaseN_bg-p-sweep/scripts/run.sh --check-env
python tests/test_bit_group_sparsity.py
```

`--preflight` 只检查配置；`--check-env` 检查实际 GPU/LeRobot/LIBERO 环境。缺少 PyTorch 时单元测试明确跳过真实前向测试，其余 NumPy 计数测试仍可运行。

## 输出目录映射

默认 `outputs/2026-11-10_phaseN_bg-p-sweep/`：

| 文件 | 内容 |
|---|---|
| `config.yaml`, `execution.json` | 生效配置、运行状态、版本与源码哈希 |
| `scales/`, `scale_audit.json` | 使用的 864 个尺度及其哈希 |
| `eval/eval_info.json` | 单 task / 单 episode rollout 结果 |
| `coverage.json` | 站点、角色、phase/step 覆盖与实际 tensor shape |
| `bit_groups/summary.md` | 七个 p 的 VLM/Expert 汇总表 |
| `bit_groups/summary.csv` | 组件 × 统计范围 × p × BG level，全量原始计数及比率 |
| `bit_groups/by_site.csv` | module × phase × step × role × p × level 明细 |
| `completed.json` | 所有检查通过后才写出 |

## 风险与注意

这是一次任务轨迹的快速统计，不代表全部任务分布。沿用既有量化器和 outlier 选择，观察器不修订其算法。统计通过独立 scalar oracle、mask/尾块/转置/负数边界测试，并验证 CPU 上开启/关闭统计的 Linear、QK/PV 输出逐元素完全相同；CUDA 完整 rollout 需在项目实验机运行。
