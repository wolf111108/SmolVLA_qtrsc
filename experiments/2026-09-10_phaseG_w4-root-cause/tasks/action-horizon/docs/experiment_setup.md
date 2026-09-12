# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 action-horizon（G3）
- **状态**：pending（G1 后按 Gate 启动）
- **负责人**：zyzhao
- **创建日期**：2026-09-11
- **相关前序实验**：父实验 Phase G 总纲；Phase D/E 消融已证 na 呈倒 U 型、na=10 最优（FP 模型）——本 task 检验 W4 误差与开环执行的交互

---

## 1. 实验目的

检验 W4 数值误差是否被 open-loop action execution（chunk 内开环执行）放大：

$$\Delta_{horizon}^{W4} = SR(W4, n{=}1) - SR(W4, n{=}10) \quad vs \quad \Delta_{horizon}^{FP8} = SR(FP8, n{=}1) - SR(FP8, n{=}10)$$

若 $\Delta_{horizon}^{W4} \ge 15pp$ 且 $|\Delta_{horizon}^{FP8}| \le 6pp$，则支持「**W4 误差 × 开环执行 → 闭环误差放大**」——VLA-specific quantization finding，可单独写入论文。

> na=1 为 paper-style feedback 对照：每执行一个 action 后重新观测重预测。
## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval（mujoco 3.3.2） |
| 仓库 revision / commit | 开跑时 `git rev-parse HEAD` 写入；若与 G1 commit 完全一致，G1-A/B 可直接充当 G3 n=10 两档 |
| LeRobot 路径与 commit | `lerobot_current/` @ `6adf5151`（v0.6.2） |
| GPU | NVIDIA H100 NVL（hankh100） |
| 关键依赖版本 | Python 3.12，torch 2.7.1+cu118，`MUJOCO_GL=egl` |

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（legacy_public） |
| 评测基准 | libero_goal（10 task × 10 ep） |
| 校准数据 | `HuggingFaceVLA/libero` v3.0，8 ep，stride 4，seed 42；同一 precision 的三个 na 档可**共享同一 scale**（na 是推理参数，不影响 scale）——但需在 runner 中显式校验 scale 一致 |

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | 100 ep/config，seed=1000 |
| 固定采样参数 | num_steps=10，chunk_size=50；唯一变量 n_action_steps ∈ {1, 5, 10} |
| 指标 | pc_success + task-wise SR（t0/t1/t2/t4/t7/t9） |

> 注意 na=1/5 时 max_episode_steps 口径与 na=10 保持一致，勿额外修改环境参数。

两因素：precision（FP8 / W4）× n_action_steps（1 / 5 / 10）。其余同 G1 anchor。

| 组 | config（configs/ 下文件名） | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| G3-A1 | `fp8_n1.yaml` | FP8，na=1 | per_site / outlier 0.01 / MatMul FP8 | |
| G3-A5 | `fp8_n5.yaml` | FP8，na=5 | 同上 | |
| G3-A10 | （复用 G1-A） | FP8，na=10 | | commit/config 一致则不重跑 |
| G3-B1 | `w4_n1.yaml` | W4，na=1 | 同上 | paper-style feedback 下 W4 表现 |
| G3-B5 | `w4_n5.yaml` | W4，na=5 | 同上 | |
| G3-B10 | （复用 G1-B） | W4，na=10 | | commit/config 一致则不重跑 |

新跑预算：4 config × 100 ep = 400 episodes。

## 6. 运行命令

<!-- 每个 config 一条可复制执行的命令，标注预期输出目录 -->

```bash
EXP=experiments/2026-09-10_phaseG_w4-root-cause
bash $EXP/tasks/action-horizon/scripts/run_action_horizon.sh   # 幂等；n=10 档自动检查 G1 结果可复用性
```

## 7. 输出目录映射

<!-- config → outputs/2026-09-10_phaseG_w4-root-cause · 子实验 action-horizon/ 下的实际产出目录（与实验同名，见 README §5），跑完后逐一登记，便于回溯原始数据 -->

| config | 输出目录 | 状态 |
|---|---|---|
| fp8_n1.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/action-horizon/fp8_n1/` | pending |
| fp8_n5.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/action-horizon/fp8_n5/` | pending |
| fp8_n10 | 复用 `tasks/component-localization/g1a_fp8_all/` | conditional |
| w4_n1.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/action-horizon/w4_n1/` | pending |
| w4_n5.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/action-horizon/w4_n5/` | pending |
| w4_n10 | 复用 `tasks/component-localization/g1b_w4_all/` | conditional |

scale：FP8 三档共享 `scales/2026-09-10_phaseG_w4-root-cause/action-horizon/fp8/`，W4 三档共享 `.../w4/`（runner 校验）。

## 8. 风险与注意事项

- na=1 时每步重预测，评测耗时约为 na=10 的数倍，预算按最坏情况估计；
- FP 模型的 na 倒 U 型（na=1 峰值 74.45%）意味着 FP8/W4 在 na=1 的绝对值都偏低属正常，重点看 **Δhorizon 差分**而非绝对 SR；
- na=5 在 FP 模型上无历史数据点，属新点位；
- G1-A/B 复用的前提是 commit 与 config（除 na 外）逐键一致，runner 需显式校验后再跳过。
