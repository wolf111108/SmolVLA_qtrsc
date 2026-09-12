# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 component-localization（G1）
- **状态**：done（draft / running / done / aborted）
- **负责人**：zyzhao
- **创建日期**：2026-09-11
- **相关前序实验**：父实验 Phase G 总纲；G0 weight-error-audit（先行，结果用于交叉验证定位）

---

## 1. 实验目的

Phase G **第一组完整闭环实验**：定位 F3 退化（goal 90%→21%）主要来自 VLM 的 16 层 Linear 还是 Expert 的 16 层 Linear。

判读表：

| 结果 | 结论 |
|---|---|
| VLM-only W4 ≈ F3，Expert-only W4 接近 baseline | VLM 主导 |
| Expert-only W4 ≈ F3 | Expert 主导 |
| 两个单独 W4 都明显优于 F3 | 32 层共同量化存在累积/交互 |
| 两个单独都很差 | W4 bulk quantization 普遍不可接受（直接转 G2） |
## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval（mujoco 3.3.2） |
| 仓库 revision / commit | `7b9bbb922dcebde75a5cc114f83a3b5f652eb152` |
| LeRobot 路径与 commit | `lerobot_current/` @ `6adf5151`（v0.6.2） |
| GPU | NVIDIA H100 NVL（hankh100） |
| 关键依赖版本 | Python 3.12，torch 2.7.1+cu118，`MUJOCO_GL=egl`（显式设置） |

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（legacy_public） |
| 评测基准 | libero_goal（10 task × 10 ep） |
| 校准数据 | `HuggingFaceVLA/libero` v3.0，8 ep，stride 4，seed 42；每个 config 独立 scale_dir |

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | 100 ep/config，seed=1000，batch_size=1，async=false |
| 采样参数 | n_action_steps=10，num_steps=10，chunk_size=50（与 F1/F3 可比） |
| 指标 | pc_success + task-wise SR（重点 t0/t1/t2/t4/t7/t9） |
| rename_map | image→camera1，image2→camera2 |

唯一变量：W4 作用的组件范围（通过 `linear.include` glob 控制）。

| 组 | config（configs/ 下文件名） | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| G1-A | `g1a_fp8_all.yaml` | 全部 Linear FP8 | per_site / outlier 0.01 / MatMul FP8 / na=10 | = F1 anchor（重新跑，不复用 F1 旧数据——独立 scale_dir） |
| G1-B | `g1b_w4_all.yaml` | 全部 Linear W4 | 同上 | = F3 anchor |
| G1-C | `g1c_w4_vlm_only.yaml` | 仅 `vlm.*` Linear W4 | 同上 | `linear.include: ["vlm.*"]`，Expert 为 raw FP |
| G1-D | `g1d_w4_expert_only.yaml` | 仅 `expert.*` Linear W4 | 同上 | `linear.include: ["expert.*"]`，VLM 为 raw FP |

> G1-C/D 中未 wrap 的组件是 **raw FP 而非 FP8**——这是「从 F3 恢复某组件到 FP」的定位实验，不是严格 FP8/W4 2×2。MatMul（QK/PV）四组均保持 F1 FP8 协议。

## 6. 运行命令

<!-- 每个 config 一条可复制执行的命令，标注预期输出目录 -->

```bash
EXP=experiments/2026-09-10_phaseG_w4-root-cause
bash $EXP/tasks/component-localization/scripts/run_component_localization.sh   # 幂等，顺序跑 4 个 config
# 单个：
MUJOCO_GL=egl python main.py --config $EXP/tasks/component-localization/configs/g1c_w4_vlm_only.yaml
```

预算：4 config × 100 ep = 400 episodes。

## 7. 输出目录映射

<!-- config → outputs/2026-09-10_phaseG_w4-root-cause · 子实验 component-localization/ 下的实际产出目录（与实验同名，见 README §5），跑完后逐一登记，便于回溯原始数据 -->

| config | 输出目录 | 状态 |
|---|---|---|
| g1a_fp8_all.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/component-localization/g1a_fp8_all/` | done 09-11 18:01（SR 88.0%） |
| g1b_w4_all.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/component-localization/g1b_w4_all/` | done 09-11 21:54（SR 21.0%） |
| g1c_w4_vlm_only.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/component-localization/g1c_w4_vlm_only/` | done 09-12 00:48（SR 21.0%） |
| g1d_w4_expert_only.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/component-localization/g1d_w4_expert_only/` | done 09-12 02:56（SR 84.0%） |

wrap 数 preflight 实测：A/B 各 224 层，C/D 各 112 层（日志 `g1_run.log`），符合 §8 预期。

scale：`scales/2026-09-10_phaseG_w4-root-cause/component-localization/g1{a,b,c,d}_*/`。

## 8. 风险与注意事项

- 归因阈值：ΔSR ≥ 15pp 主要贡献，6–15pp 中等，≤6pp 不做强归因（±5.7pp 噪声）；
- 判读必须看 paired task-wise SR（t0=resistant control，t1/t2/t4/t7/t9=崩溃组），不能只看 overall；
- G1-A/B 与 F1/F3 协议相同但**独立校准**（新 scale_dir），若 G1-B 与 F3 goal 21% 差异大，先查校准复现性再下结论；
- `linear.include` 的 glob 语义需先 preflight 验证 wrap 数量（G1-C 应 wrap 112 个 vlm Linear，G1-D 应 wrap 112 个 expert Linear）；
- 评测需显式 `MUJOCO_GL=egl`。
