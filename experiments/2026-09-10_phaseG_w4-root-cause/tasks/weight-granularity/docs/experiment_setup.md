# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 weight-granularity（G2）
- **状态**：pending（G1 后按 Gate 启动）
- **负责人**：zyzhao
- **创建日期**：2026-09-11
- **相关前序实验**：父实验 Phase G 总纲；前置 G1 component-localization（若 G1 已定位单一组件，可只在该组件上扫粒度）

---

## 1. 实验目的

验证概率最高的根因假设：

> 当前每个物理 Linear 虽为 `per_site`，但 W4 weight matrix 内部主要使用一个 tensor scale；INT4 量化级极少，tensor-wise scale 不足以覆盖行间分布差异。

**G2 Gate**：若 $SR_{channel/group} - SR_{per\_tensor} \ge 15pp$，则确认 F3 主因是 **W4 scale granularity 而非 W4 本身不可用**——直接决定后续硬件研究采用 per-channel W4 还是 group-wise W4。
## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval（mujoco 3.3.2） |
| 仓库 revision / commit | 开跑时 `git rev-parse HEAD` 写入；本 task 需新增 `weight_quant_granularity` 代码支持，commit 必须含该实现 |
| LeRobot 路径与 commit | `lerobot_current/` @ `6adf5151`（v0.6.2） |
| GPU | NVIDIA H100 NVL（hankh100） |
| 关键依赖版本 | Python 3.12，torch 2.7.1+cu118，`MUJOCO_GL=egl` |

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（legacy_public） |
| 评测基准 | libero_goal（10 task × 10 ep；若 G1 已定位单一组件，可仅在该组件上扫描以缩减预算） |
| 校准数据 | `HuggingFaceVLA/libero` v3.0，8 ep，stride 4，seed 42；每 config 独立 scale_dir |

需在 config/代码中**新增正交字段**（与 `linear_scale_granularity` 概念不同，不能复用）：

```yaml
weight_quant_granularity: per_tensor   # per_tensor / per_output_channel / groupwise
weight_group_size: null                # groupwise 时填 128/64/32
```

Groupwise 沿 `in_features` 分组：$W\in\mathbb{R}^{N_{out}\times K}$ 的每个 output row 内 $K\rightarrow[K_0,K_1,\dots]$，每组单独 symmetric W4 scale。

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | 100 ep/config，seed=1000 |
| 采样参数 | n_action_steps=10，num_steps=10，chunk_size=50 |
| 指标 | pc_success + task-wise SR（t0/t1/t2/t4/t7/t9） |

统一固定：Linear W=INT4、Linear A/O=E4M3+PoT、Linear outlier=0.01、MatMul=F1 FP8+PoT+outlier、`linear_scale_granularity=per_site`、na=10。

唯一变量：W4 intra-weight granularity。

| 组 | config（configs/ 下文件名） | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| G2-A | `g2a_per_tensor.yaml` | `weight_quant_granularity: per_tensor` | 其余同 G1-B（F3 协议） | = F3 复现 anchor（若 G1-B 已有可直接复用其结果） |
| G2-B | `g2b_per_channel.yaml` | `per_output_channel` | 同上 | 每输出行一个 scale |
| G2-C | `g2c_group128.yaml` | `groupwise`，group_size=128 | 同上 | |
| G2-D | `g2d_group64.yaml` | `groupwise`，group_size=64 | 同上 | |
| G2-E | `g2e_group32.yaml` | `groupwise`，group_size=32 | 同上 | |

前置代码工作：`quant_linear.py` / `stat_manager.py` 需支持非标量 w scale（per-channel / per-group 的存取与量化路径），落盘格式变更需 preflight 验证。

## 6. 运行命令

<!-- 每个 config 一条可复制执行的命令，标注预期输出目录 -->

```bash
EXP=experiments/2026-09-10_phaseG_w4-root-cause
bash $EXP/tasks/weight-granularity/scripts/run_weight_granularity.sh   # 幂等，顺序跑 5 个 config
```

启动前需完成 `weight_quant_granularity` 实现 + raw 等价性冒烟（groupwise 路径在 enabled=false 时 bit-exact）。

## 7. 输出目录映射

<!-- config → outputs/2026-09-10_phaseG_w4-root-cause · 子实验 weight-granularity/ 下的实际产出目录（与实验同名，见 README §5），跑完后逐一登记，便于回溯原始数据 -->

| config | 输出目录 | 状态 |
|---|---|---|
| g2a_per_tensor.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-granularity/g2a_per_tensor/` | pending |
| g2b_per_channel.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-granularity/g2b_per_channel/` | pending |
| g2c_group128.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-granularity/g2c_group128/` | pending |
| g2d_group64.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-granularity/g2d_group64/` | pending |
| g2e_group32.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-granularity/g2e_group32/` | pending |

scale：`scales/2026-09-10_phaseG_w4-root-cause/weight-granularity/g2*/`。

## 8. 风险与注意事项

- **必须新增正交字段**，不能把 `linear_scale_granularity` 改成 per_channel 后声称完成 groupwise W4——二者概念不同（site 级 vs tensor 内粒度）；
- scale 落盘格式从标量变为向量/矩阵，`stat_manager` 聚合与 reuse 判定需同步适配并验证；
- outlier 保护的 w 元素 mask 在 groupwise scale 下的口径需保持与 tensor-wise 一可（protected 元素仍 FP）；
- 若 G1 已定位单一组件且预算紧张，可先只扫 per_channel vs per_tensor 两点（Gate 判据足够），groupwise 档位后补。
