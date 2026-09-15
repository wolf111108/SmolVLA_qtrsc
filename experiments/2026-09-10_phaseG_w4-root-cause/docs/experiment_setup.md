# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-10_phaseG_w4-root-cause
- **状态**：running（draft / running / done / aborted）
- **负责人**：zyzhao
- **创建日期**：2026-09-11
- **相关前序实验**：`experiments/2026-09-08_phaseF_fp8pot-foursuite/`（F3 `Linear W4 + FP8 A/O + FP8 MatMul` 四 suite 平均 83.0% → 48.5%，`libero_goal` 90% → 21%，退化远超 ep10 ±5.7pp 噪声带——本实验做 root-cause attribution）

---

## 1. 实验目的

不再验证「W4 是否掉点」（Phase F 已确认），而是把 F3 的 −36pp 平均损失分解为：

$$\Delta SR_{W4} = \Delta_{component} + \Delta_{quant\text{-}grid} + \Delta_{closed\text{-}loop} + \Delta_{outlier/mask} + \epsilon$$

依次回答：

1. **哪一部分模型造成**（VLM vs Expert）？
2. **是否来自 W4 scale granularity**（当前 W4 weight 为 tensor-wise，`per_site` 只是 site 级，不是 per-channel）？
3. **是否被 action open-loop horizon 放大**（`n_action_steps`）？
4. **是否来自 dynamic outlier mask / scale mismatch**？

主诊断 suite：`libero_goal`（F3 失败具任务选择性：t1/t2/t4/t7/t9 全部 0/10，t0 仍 10/10）。
## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval |
| 仓库 revision / commit | `7b9bbb922dcebde75a5cc114f83a3b5f652eb152`（开跑时以 `git rev-parse HEAD` 重新记录于各 task 文档） |
| LeRobot 路径与 commit | `lerobot_current/` @ `6adf5151`（v0.6.2） |
| GPU | NVIDIA H100 NVL（hankh100） |
| 关键依赖版本 | Python 3.12，torch 2.7.1+cu118，mujoco 3.3.2，`MUJOCO_GL=egl`（评测时显式设置） |

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（legacy_public；VLM 16 层 + Expert 16 层） |
| 量化范围 | Linear：q/k/v/o + gate/up/down；MatMul：QK/PV；vision encoder / connector / action head 保持 FP |
| 校准数据 | `HuggingFaceVLA/libero` v3.0，8 ep，batch 8，stride 4，seed 42 |

量化 baseline 严格复制 Phase F：F1 = Linear/MatMul FP8-E4M3 + PoT + outlier 0.01；F3 = 仅 Linear 权重 INT4（连续 scale），A/O 与 MatMul 仍 FP8。`linear_scale_granularity: per_site` 全程固定。

> ⚠ 注意：`per_site` 表示不同物理 site 各自持有 scale；W4 weight matrix 内部仍是 **tensor-wise** 量化，不等于 per-channel。

## 4. 评测协议

| 项目 | 值 |
|---|---|
| suite | libero_goal（主诊断） |
| episodes × seeds | 10 task × 10 ep = 100 ep/config，seed=1000，batch_size=1，max_parallel_tasks=1，async=false |
| 采样参数 | `n_action_steps=10`（与 F1/F3 直接可比；仅 G3 修改），`num_steps=10`，chunk_size=50 |
| 指标 | pc_success + task-wise SR（重点跟踪 goal t0/t1/t2/t4/t7/t9） |
| rename_map | image→camera1，image2→camera2 |

父实验为容器，不直接运行 config；各 task 自包含 YAML/script（见 tasks/*/docs/experiment_setup.md）。执行顺序按 Gate 推进，**不一次性全部启动**：

```
G0 数值误差(weight-error-audit) → G1 组件定位(component-localization)
  → 看结果决定 G2(weight-granularity) / G3(action-horizon) 优先级 → 必要时 G4(outlier-mask-diagnosis)
  → G5 定位 VLM 内部 attention vs MLP → G6 验证结论对 Expert-FP8 背景鲁棒
```

| task | 问题 | 一句话设计 | 状态 |
|---|---|---|---|
| G0 weight-error-audit | 敏感度集中于何处？ | 离线（无 rollout）：FP8 vs W4 两套配置，全部 224 Linear 的 weight/output SQNR·NMSE·cosine·饱和率·保护率排行 | pending |
| G1 component-localization | VLM 还是 Expert？ | Goal×100ep 闭环：G1-A(F1 anchor FP8) / G1-B(F3 anchor W4) / G1-C(W4 VLM only) / G1-D(W4 Expert only) | pending |
| G2 weight-granularity | tensor-wise W4 是根因？ | 新增正交字段 `weight_quant_granularity`：per_tensor / per_output_channel / group 128/64/32 | pending（G1 后） |
| G3 action-horizon | 开环放大？ | FP8 与 W4 两 anchor × n_action_steps∈{1,5,10}；n=10 可复用 G1-A/B | pending（G1 后） |
| G4 outlier-mask-diagnosis | mask 漂移？ | 条件触发：calibration mask vs frozen mask vs runtime oracle 三对照 | conditional |
| G5 vlm-selective-precision | VLM 内 attention 还是 MLP？ | Expert raw FP 背景，VLM attention(q/k/v/o) vs MLP(gate/up/down) 分别压 W4（G5-A/B/C + 复用 G1-C） | done（90/72/39/21） |
| G6 vlm-selective-expert-fp8 | 结论对 Expert-FP8 鲁棒？ | 升级 `linear.overrides`，Expert FP8 背景重做 VLM selective precision（G6-A/B/C/D） | running（Gate 0-5 PASS，Gate 6 g6a=90%） |

归因阈值：ΔSR ≥ 15pp 为主要贡献；6–15pp 中等；≤6pp 不做强归因（ep10 100-ep 噪声 ≈±5.7pp）。比较优先看 paired task-wise SR 而非仅 Goal overall。

## 6. 运行命令

<!-- 每个 config 一条可复制执行的命令，标注预期输出目录 -->

```bash
EXP=experiments/2026-09-10_phaseG_w4-root-cause

# 第一轮：只启动 G0 + G1
bash $EXP/tasks/weight-error-audit/scripts/run_weight_error_audit.sh
bash $EXP/tasks/component-localization/scripts/run_component_localization.sh

# 后续按 Gate 决定
bash $EXP/tasks/weight-granularity/scripts/run_weight_granularity.sh
bash $EXP/tasks/action-horizon/scripts/run_action_horizon.sh
bash $EXP/tasks/outlier-mask-diagnosis/scripts/run_outlier_mask_diagnosis.sh   # conditional

# G5：VLM 内部 selective precision（Expert raw FP 背景）
bash $EXP/tasks/vlm-selective-precision/scripts/run_vlm_selective_precision.sh

# G6：Expert-FP8 背景的 VLM selective precision（linear.overrides 升级）
python $EXP/tasks/vlm-selective-expert-fp8/scripts/make_g6_configs.py
bash $EXP/tasks/vlm-selective-expert-fp8/scripts/run_smoke.sh
bash $EXP/tasks/vlm-selective-expert-fp8/scripts/run_goal.sh
```

幂等规则：存在完整 eval_info.json → skip。**每个 config 独立 scale_dir**，Phase G 不跨 config 复用 calibration cache（规避 Phase F 的 reuse 失效混杂）。

## 7. 输出目录映射

<!-- config → outputs/2026-09-10_phaseG_w4-root-cause/ 下的实际产出目录（与实验同名，见 README §5），跑完后逐一登记，便于回溯原始数据 -->

父实验无直接 config（见各子实验）。原始产出统一挂 `outputs/2026-09-10_phaseG_w4-root-cause/tasks/<task>/`，scale 挂 `scales/2026-09-10_phaseG_w4-root-cause/<task>/`。详见各 task 的 setup 文档第 7 节。

## 8. 风险与注意事项

- **不要一次跑完五个 task**：当前目标是归因，不是堆 SR 数据；G0+G1（400 ep）出来后通常已砍掉一半假设空间再决定下一步；
- **G1-C/D 是「恢复某组件到 FP」的定位实验**，非严格 FP8/W4 2×2——利用现有 `linear.include` glob 最小改动实现，若已明确定位则无需为对称性重构框架；
- **G2 新增的是 tensor 内量化粒度**（`weight_quant_granularity`），必须与 `linear_scale_granularity=per_site` 正交，不能复用旧字段；
- 所有比较优先看 paired task-wise SR（goal t0=resistant control，t1/t2/t4/t7/t9=崩溃组）；
- 评测需显式 `MUJOCO_GL=egl`（shell 环境默认 osmesa，勿沿用）。
