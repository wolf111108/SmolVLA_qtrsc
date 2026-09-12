# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 outlier-mask-diagnosis（G4）
- **状态**：conditional（**仅当 G0–G3 无法充分解释 F3 时启动**）
- **负责人**：zyzhao
- **创建日期**：2026-09-11
- **相关前序实验**：父实验 Phase G 总纲；前置 G0–G3 全部完成且归因不充分

---

## 1. 实验目的

区分 F3 退化是否来自 **dynamic outlier mask 的 calibration/runtime 漂移**，而非 W4 量化本身。三组判读：

| 结果 | 结论 |
|---|---|
| G4-B ≫ G4-A | mask 漂移重要（frozen 校准 mask 更稳定） |
| G4-C ≫ G4-B ≈ G4-A | 不是 mask 本身，而是 runtime normal-weight subset 变化后原 scalar scale 不再适配 |
| 三者接近 | 排除 dynamic mask 为主因，回到 bulk W4 量化本身 |
## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval（mujoco 3.3.2） |
| 仓库 revision / commit | 开跑时 `git rev-parse HEAD` 写入；需新增 frozen-mask / runtime-oracle 代码支持 |
| LeRobot 路径与 commit | `lerobot_current/` @ `6adf5151`（v0.6.2） |
| GPU | NVIDIA H100 NVL（hankh100） |
| 关键依赖版本 | Python 3.12，torch 2.7.1+cu118，`MUJOCO_GL=egl` |

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（legacy_public） |
| 评测基准 | libero_goal（10 task × 10 ep） |
| 校准数据 | `HuggingFaceVLA/libero` v3.0，8 ep，stride 4，seed 42；三组共享同一校准 mask，仅 runtime 行为不同 |

基础协议 = F3（全 Linear W4 + FP8 A/O + FP8 MatMul + outlier 0.01 + per_site + na=10）。

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | 100 ep/config，seed=1000 |
| 采样参数 | n_action_steps=10，num_steps=10，chunk_size=50 |
| 指标 | pc_success + task-wise SR（t0/t1/t2/t4/t7/t9） |

唯一变量：outlier mask 在 runtime 的来源/冻结方式。全部保持全 Linear W4 + F3 协议。

| 组 | config（configs/ 下文件名） | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| G4-A | `current_dynamic.yaml` | calibration mask + runtime dynamic mask + calibrated scale | = F3 现状（复现 anchor） | runtime 按当次输入重算 mask |
| G4-B | `frozen_mask.yaml` | calibration mask + **frozen 校准 mask** + calibrated scale | | runtime 不重算 mask |
| G4-C | `oracle_runtime_scale.yaml` | calibration mask + dynamic mask + **runtime oracle 重算 scale** | 仅机制诊断 | 不可部署方案 |
| G4-D（可选） | `weight_elem_only.yaml` | 仅 weight 元素级保护，关闭 activation 通道→weight 列保护 | | 需要区分保护来源时追加 |

前置代码工作：`quant_forward_with_outlier` 需支持 mask 冻结 / runtime scale 重算开关（config 字段，如 `outlier_mask_source: dynamic/frozen`）。

## 6. 运行命令

<!-- 每个 config 一条可复制执行的命令，标注预期输出目录 -->

```bash
# 仅当 G0–G3 归因不充分时执行
EXP=experiments/2026-09-10_phaseG_w4-root-cause
bash $EXP/tasks/outlier-mask-diagnosis/scripts/run_outlier_mask_diagnosis.sh
```

## 7. 输出目录映射

<!-- config → outputs/2026-09-10_phaseG_w4-root-cause · 子实验 outlier-mask-diagnosis/ 下的实际产出目录（与实验同名，见 README §5），跑完后逐一登记，便于回溯原始数据 -->

| config | 输出目录 | 状态 |
|---|---|---|
| current_dynamic.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/outlier-mask-diagnosis/current_dynamic/` | conditional |
| frozen_mask.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/outlier-mask-diagnosis/frozen_mask/` | conditional |
| oracle_runtime_scale.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/outlier-mask-diagnosis/oracle_runtime_scale/` | conditional |
| weight_elem_only.yaml（可选） | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/outlier-mask-diagnosis/weight_elem_only/` | conditional |

scale：共享 `scales/2026-09-10_phaseG_w4-root-cause/outlier-mask-diagnosis/`（同一校准 mask，仅 runtime 行为不同）。

## 8. 风险与注意事项

- **本 task 是条件触发**，启动前需在父实验 results.md 记录 G0–G3 为何不足以归因；
- G4-C（oracle）每层每步重算 absmax，开销大且仅作机制诊断，不用于任何部署结论；
- 三组共享 scale 的前提是校准完全一致，runner 需校验 scale 文件 hash；
- 若 G4-D 结果与 G4-A 接近，说明 activation 通道→weight 列保护贡献很小，可结合 G2 的 per-channel 结论合并解释。
