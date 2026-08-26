# Table-2 严格复现排查记录

> 对应手册：`SmolVLA Table 2 — LIBERO 严格复现系统排查与执行手册 v1.0.md`
> **任务设备归属：h100**（zyzhao@hankh100，H100 NVL，smolvla_eval 环境）
> 纪律：一次只改变一个变量；未 PASS 不进下一阶段；不为凑数事后挑 seed/protocol。
> 状态标记：PASS / FAIL / PARTIAL / UNRESOLVED / NOT SPECIFIED BY PAPER / **PRE-FILLED**(已有证据预填，待复核)

## 阶段状态总览

| Phase | 内容 | 状态 | 证据位置 |
|---|---|---|---|
| 0 | 环境冻结 + Manifest | **PASS**（2026-08-25 执行） | `00_manifest/` |
| 1 | LIBERO Integration Smoke | **PASS**（2026-08-25 归档 smoke 100%） | `01_integration/` |
| 2 | Observation Contract | **PASS**（2026-08-25 源码审计闭环） | `02_observation_contract/` |
| 3 | Action/Gripper Contract | **PASS**（gripper 二值，数值链自洽，无 flip hack） | `03_action_contract/` |
| 4 | Eval Protocol | **PASS**（n_action_steps=1 全程一致，无混用） | `04_eval_protocol/PROTOCOL.md` |
| 5 | Suite Ordering / RNG | ⏳ 实验 B 3.8.1 已 kill；新增 3.3.2 combined 待跑 | `05_rng_ordering/` |
| 6 | MuJoCo 版本 | **决定性发现**（Task5: 3.8.1=0/10 → 3.3.2=8/10） | `06_simulator/mj332_task5_x10` |
| 7 | Checkpoint Provenance | **PASS**（2026-08-25 五 cp 自动审计归档） | `07_checkpoint_provenance/` |
| 8 | Strict Table-2 Init | ⏳ 待执行 | `08_strict_init/` |
| 9 | Dataset 审计 | PARTIAL（tiantianx stats 已比对） | `09_dataset/` |
| 10 | Training Recipe | ⏳ 待执行（10B 已闭环） | `10_training_recipe/` |
| 10B | Global Batch/Accum 语义 | **PASS（源码已确认 step=batch loop）** | `10_training_recipe/BATCH_ACCUM.md` |
| 11 | 10-step Smoke | ⏳ 待执行 | `11_smoke_train/` |
| 11B | Trainable Param Audit | ⏳ 待执行 | `11_smoke_train/` |
| 12 | 500-step Smoke | ⏳（08-24 曾尝试 smolvla_base 路线，已废弃改 strict 路线） | `12_short_train/` |
| 13 | 5k/10k/20k Short-run | ⏳ | `12_short_train/` |
| 14 | 100k Full Training | ⏳（14 项 checklist 全过才启动） | `13_full_train/` |
| 15 | 400-episode Eval | ⏳ | `14_full_eval/` |
| 16 | 最终归因 | ⏳ | — |

---

## PHASE 0 — 环境冻结 + Manifest

- 状态：**PASS**（2026-08-25T16:18:49+08:00 执行）
- Git HEAD：`6adf51511b7625090eade8d82d9f61a1846ebe56` ✅ 与项目记录一致（branch=main）
- ⚠️ git status 发现两处未冻结：
  - ` M pyproject.toml`（本地 cmake fix 的残留修改，需在手册 0.3 精神下确认是否影响实验语义）
  - `?? k1000dai_smolvla_libero_finetune_migrated/`（preflight 迁移产物，未跟踪）
- 软件栈：
  | 项 | 值 |
  |---|---|
  | python | 3.12.13 |
  | torch | 2.7.1+cu118 |
  | cuda runtime | 11.8（驱动 CUDA 12.4） |
  | lerobot | 0.6.2 |
  | **mujoco** | **3.8.1**（见 PHASE 6，偏离手册 reference） |
  | robosuite | 1.4.0 |
  | robomimic | 0.2.0 |
  | datasets | 4.8.5 |
  | transformers | 5.5.4 |
  | accelerate | 1.14.0 |
  | huggingface_hub | 1.27.0 |
- EGL：`egl / egl / 2` ✅
- GPU：H100 NVL，550.163.01 驱动；执行时有 hhhuang 进程占用 2.5GB（smolvla_libero_mj332 环境）
- HF revisions（已含 tiantianx/k1000dai）：
  ```
  MODEL  lerobot/smolvla_base              c83c3163b8ca9b7e67c509fffd9121e66cb96205
  MODEL  lerobot/smolvla_libero            31d453f7edd78c839a8bbc39744a292686daf0de
  MODEL  HuggingFaceVLA/smolvla_libero     6721902bc4d61e50a3bfdb11dfb4cb626f05d102
  MODEL  tiantianx/smolvla_libero          98343cf58d6669cad9686c9251e4c33d18a27a76
  MODEL  k1000dai/smolvla_libero_finetune  492ac1c5f1b7808c444fae37b75a84fdeb15e70d
  DATASET HuggingFaceVLA/libero            86958911c0f959db2bbbdb107eb3e17c5f9c798e
  DATASET physical-intelligence/libero     a4336d589d589045d1c56423ffdf3b88a0e19b1f
  ```
- 全部证据落盘：`00_manifest/{git,software,egl,nvidia_smi,pip_freeze,hf_revisions}.txt`

## PHASE 1 — Integration Smoke

- 状态：**PASS（2026-08-25T16:29 归档 smoke 完成）**
- 命令：手册 1.2（hfvla official reference, spatial task0, 1 episode, seed=1000）
- 结果：**pc_success = 100.0%（1/1）**，用时 61.8s，rollout 完整、无 crash
- 手册 1.3 PASS 条件全部满足：模型加载 ✅ / env reset ✅ / 完整 rollout ✅ /
  无 EGL/CUDA crash ✅ / 机器人行为非随机（成功完成任务）✅ / success detector 正常 ✅
- 归档：`01_integration/task0/eval_info.json` + 视频
- 额外证据：hfvla 官方 reference 四 suite 全跑通（S65/O71/G72/L37），远强于单次 smoke

## PHASE 2 — Observation Contract

- 状态：**PASS（2026-08-25 源码审计闭环）**
- 完整矩阵见 `02_observation_contract/CONTRACT.md`
- 核心结论：
  - Runtime state = eef_pos(3) + axis_angle(3) + gripper_qpos(2) = **8D**（env_processor.py:75）
  - 图像 `torch.flip(dims=[2,3])` 180° flip，对齐 HuggingFaceVLA/libero convention
  - 相机：agentview→image、eye_in_hand→image2，共 2 相机，无 camera3
  - env：fps=20, hard_reset=True, init_states=True, control_mode=relative, num_steps_wait=10
  - TASK_SUITE_MAX_STEPS = {spatial:280, object:280, goal:300, libero_10:520}
  - training semantic = runtime semantic ✅，与 tiantianx normalizer 逐位一致证据互相印证
- 遗留（低风险）：_quat2axisangle 数学实现未逐行验算，但 shape/语义已确认

## PHASE 3 — Action/Gripper Contract（优先级最高）

- 状态：**PASS（2026-08-25 数值链闭环）**
- 完整结论见 `03_action_contract/ACTION_CONTRACT.md`
- 核心结论：
  - Action 链路 = model raw → policy unnormalizer(MEAN_STD 逆变换) → env postprocessor(identity) → env.step(透传)
  - **gripper 是二值信号**：`action[-1] ∈ {-1, +1}`（percentiles 只有两值，std≈0.998）
  - dataset stats 的 action.std 末分量=0.9988、mean=-0.0496 → 值域即 [-1,+1]，unnormalize 精确还原
  - **无 #1316 时代的手写 gripper flip hack**，当前数据+pipeline 下该问题不存在
  - 语义方向高置信推断：-1=开/保持（settle no-op），+1=闭合（robosuite convention）
- 遗留（非阻塞）：语义方向未做视频逐帧实锤，但数值自洽已确证

## PHASE 4 — Eval Protocol

- 状态：**PASS（2026-08-25 归档确认）**
- 完整见 `04_eval_protocol/PROTOCOL.md`
- 核心结论：
  - 五模型四 suite 全部 `n_action_steps=1, num_steps=10, seed=1000, chunk=50`，与论文协议一致
  - **无 n_action_steps=10/30/50 混用**（手册 #3264 警告项）
  - 回答 Q8：n_action_steps 严格为 1 ✅

## PHASE 5 — Suite Ordering / RNG

- 状态：⏳ 实验 B′ 在 gpupro6000a 运行中（3.3.2）
- 已确认前提：当前 `create_libero_envs` 支持逗号分隔多 suite（libero.py:464）
- 实验 A（Long 单独跑）：**已有**，A 类 Long = 59%（逐 suite 串行独立进程，3.8.1）
- 实验 B（combined 四 suite，3.8.1）：原 PID **3130854** 已 kill，`combined/` 仅 videos/ 未完成
- 实验 B′（combined 四 suite，3.3.2）：新增脚本 `scripts/run_phase5_combined_mj332.sh`
  - 输出 `05_rng_ordering/combined_mj332/`，日志 `combined_mj332_run.log`
  - **运行设备 gpupro6000a（10.113.225.63），PID 2565039**，mujoco 3.3.2 + EGL device 0
  - 40 env 已建成，已进入 rollout 阶段（2026-08-25 23:xx 确认）
- 完整对比矩阵（3.3.2 下凑齐 suite-ordering 四格）：
  - 3.8.1 combined（B）vs 3.8.1 逐 suite（A）
  - 3.3.2 combined（B′）vs 3.3.2 逐 suite（Phase 6 `mj332_A_*`）
  - 3.8.1 vs 3.3.2 的 combined 版本差异
- 判定（完成后）：ΔLong = combined_Long − 逐suite_Long；|ΔLong|≤3pp→PASS，≥5pp→RNG 深审计
- 结果：待填

## PHASE 6 — MuJoCo 版本

- 状态：**决定性发现（2026-08-25）**
- 实测：mujoco 3.8.1 → 建 `smolvla_eval_mj332` 环境（clone 后仅降级 mujoco==3.3.2）
- **对照实验（A 类 lerobot/smolvla_libero, Spatial Task5, seed=1000, 10 eps）：**

  | 环境 | mujoco | Task5 SR |
  |---|---|---:|
  | smolvla_eval | 3.8.1 | 0/10 = 0% |
  | smolvla_eval_mj332 | 3.3.2 | 8/10 = 80% |

  - 逐集：`T T T T T T T T F F`
- **结论：MuJoCo 版本是决定性变量**，比 #2114 的 35%→46% 更剧烈（0%→80%）
- 影响：
  1. A 类 Task5=0/10 的"异常点"根因 = MuJoCo 3.8.1 版本问题，非 checkpoint 缺陷
  2. 五模型 benchmark（全在 3.8.1）绝对数字可能系统性低估，异常低任务需在 3.3.2 复测
  3. 论文 87.3% 的 gap 中可能含 MuJoCo 版本贡献（公开报告多在 3.3.2 时代）
- 后续：主环境是否整体迁移到 mj332 待决策；至少需在 3.3.2 下复测五模型四 suite
- **进行中**：A 类四 suite 在 mj332 下全量评测
  - 脚本 `scripts/run_phase6_mj332_A_foursuite.sh`，输出 `06_simulator/mj332_A_<suite>/`
  - 对照基准：3.8.1 下 A 类 S81/O60/G77/L59（平均 69.2%）
  - **运行设备 gpupro6000d，PID 167786**，mujoco 3.3.2，已进入 spatial rollout
  - 预计 8-12h，与 Phase 5 combined 分机并行

## PHASE 7 — Checkpoint Provenance

- 状态：**PASS（2026-08-25 五 checkpoint 自动审计归档）**
- 完整并排表见 `07_checkpoint_provenance/PROVENANCE.md` + `configs.txt`
- 核心结论：
  - HuggingFaceVLA/smolvla_libero 三处架构漂移实测确认（Instruct 底座/32层/0.5宽）→ 非 Table-2 结构
  - A 类 lerobot 是"架构像论文、recipe 完全不像"：steps=25000、batch=32、freeze_vision_encoder=False、train_expert_only=False
  - **D 类 k1000dai 是 recipe 最贴论文的**：steps=100k、batch=64、freeze_vision_encoder=True、train_expert_only=True（比 tiantianx 多 batch=64）
  - 但 k1000dai 成绩仅 65.5% < A 类 69.2% → recipe 吻合 ≠ 成绩吻合
  - 回答手册 Q1（Table-2 原始 cp 未公开）、Q2（official 架构漂移）

## PHASE 8 — Strict Table-2 Initialization

- 状态：⏳ 待执行（direction 转折点）
- 关键决定（依手册 8.1）：**放弃 `--policy.path=lerobot/smolvla_base` 路线**（含 robotics pretraining，非 strict），
  改用 `--policy.type=smolvla --policy.load_vlm_weights=true`（仅 pretrained VLM + 随机 expert）。
- 待办：8.2 目标结构确认 / 8.3 trainable 审计 / 8.4 train_state_proj assumption 记录。
- 结果：待填

## PHASE 9 — Dataset 审计

- 状态：PARTIAL
- 已有：HuggingFaceVLA/libero stats 与 tiantianx normalizer 一致；
- 待办：physical-intelligence/libero vs HuggingFaceVLA/libero 对照；40-task assumption 标注。
- 结果：待填

## PHASE 10 / 10B — Recipe 与 Batch 语义

- 状态：10B **PASS**（源码已确认）；10 待执行
- 10B 源码结论（`10_training_recipe/BATCH_ACCUM.md`）：
  - `effective batch = micro batch × dp_world × accum_steps`（lerobot_train.py:600）
  - **`cfg.steps` 统计 micro-batch loop**（lerobot_train.py:726 `for _ in range(step, cfg.steps)`）
  - `optimizer.step()` 在 `accelerator.accumulate` 内，非 sync micro-batch 是 no-op → accum=4 时 100k steps 仅 = 25k optimizer updates
  - ⚠️ `lr_scheduler.step()` 每个 micro-batch 都调，warmup 语义也是 micro-batch 单位 → 论文 warmup=100 vs preset 1000 的差异可能源于 step 语义不同
- 10 关键差异：warmup paper=100 vs 当前 preset=1000（手册 10.2）
- 结论：正式训练若 accum>1，`steps` 须设为 `100k × accum` 才得 100k optimizer updates；若 accum=1 则无歧义

## PHASE 11–15 — 训练与评测

- 状态：⏳ 全部待执行，前置 Gate 见总览
- 08-24 的 smolvla_base 微调尝试（500 步冒烟踩坑记录见 logs/2026-08-24）改归入废弃线：
  该路线非 strict Table-2 init。踩坑经验（type/path 互斥、accelerate、repo_id）依然有效。

## PHASE 16 — 归因

- 状态：⏳
- 预判（基于现有五模型数据）：大概率落入 Case B（65-75%），
  与 #2354/#3264/#3287 及全部公开 checkpoint 的 gap 一致。

---

## 变更日志

- 2026-08-25：创建记录；预填 Phase 1/2/4/7/9 已有证据；执行 Phase 0（PASS）。
- 2026-08-25：Phase 0 回填完整 manifest；发现 MuJoCo=3.8.1 偏离手册 3.3.2 reference（Phase 6 → FAIL，主环境保持冻结，待单独建 mj332 测试环境）；确认 git HEAD 未漂移但 pyproject.toml 有本地 cmake fix 残留修改。
- 2026-08-25：Phase 2 源码审计闭环 → PASS（8D state / 180° flip / 2 相机 / env 设置全确认，CONTRACT.md 落盘）；发现 rg 未安装，改用 grep -rnE 等效。Phase 3 静态源码+dataset gripper 统计已执行，待解读。
- 2026-08-25：Phase 1 归档 smoke 完成 → PASS（hfvla spatial task0 = 100%，01_integration/task0 落盘）。
- 2026-08-25：Phase 3 源码链路审计 → PARTIAL（无 flip hack、env postprocessor=identity、MEAN_STD 链路自洽，ACTION_CONTRACT.md 落盘；dataset gripper 数值仍在下载）。Phase 10B 源码审计 → PASS（step=micro-batch loop、accum 语义、scheduler 每 batch 都 step，BATCH_ACCUM.md 落盘）。
- 2026-08-25：Phase 7 自动审计 → PASS（五 checkpoint config/train_config 归档，PROVENANCE.md 落盘；确认 HuggingFaceVLA 架构漂移、D 类 recipe 最贴论文但成绩不匹配、Q1/Q2 回答）。
- 2026-08-25：Phase 3 数值链闭环 → PASS（gpupro6000 上 dataset gripper 统计：二值 {-1,+1}、std≈0.998，结合源码 MEAN_STD 链路确证自洽、无 flip hack）。
- 2026-08-25：Phase 4 归档 → PASS（n_action_steps=1 全程一致，无混用，PROTOCOL.md 落盘）。Phase 5 实验 B 启动（PID 3130854，combined 四 suite，预计 8-12h）。
- 2026-08-25：Phase 6 决定性发现 → 建 smolvla_eval_mj332 环境（仅降级 mujoco 3.3.2），A 类 Spatial Task5 从 3.8.1 的 0/10 飙升至 3.3.2 的 8/10（+80pp）。MuJoCo 版本是决定性变量，五模型 benchmark 数字可能被系统性低估。
- 2026-08-25：Phase 6 后续启动 → A 类四 suite 在 mj332 全量评测（PID 3156295，mujoco 3.3.2 已确认），与 Phase 5 combined 并行。
- 2026-08-25：Phase 5/6 任务重排 → 原 combined（3.8.1，PID 3130854）与 mj332 A（PID 3156295）被 kill；新增 Phase 5 combined 的 3.3.2 版脚本 `run_phase5_combined_mj332.sh`（输出 `05_rng_ordering/combined_mj332/`），决定在 3.3.2 下补齐 suite-ordering 完整对比（combined vs 逐 suite 串行 × 两个 mujoco 版本）。
- 2026-08-25：任务分发到 gpupro6000 系列 → Phase 6 逐 suite 串行在 **gpupro6000d**（PID 167786），Phase 5 combined 在 **gpupro6000a**（PID 2565039，EGL device 0）。两台均确认 mujoco 3.3.2（smolvla_eval 环境即 3.3.2，无需单独 mj332 环境）。gpupro6000a 需手动补 EGL 变量（MUJOCO_GL=egl、MUJOCO_EGL_DEVICE_ID=0）并固化进 conda 环境。两任务分机并行，预计 8-12h，完成后汇总 eval_info.json 回 h100 做四格对照。
