# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。
> 本文档由原中文模板与协作者英文设计文档 `full_vision_linear_integration.md`（VLIN 部分）合并翻译而成（2026-09-21）。

- **实验名称**：2026-09-15_phaseI_vision-quantization
- **状态**：running（V0 ✅ / V1 ✅ / VLIN ✅ / V2–V5 待跑）
- **负责人**：
- **创建日期**：2026-09-15
- **相关前序实验**：`experiments/2026-09-10_phaseG_w4-root-cause/`（G6-A canonical FP8 scale 来源）、`experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/`（H3 S0 baseline）

---

## 1. 实验目的

<!-- 要回答的科学问题 / 要验证的假设，1-3 条 -->

- 量化 Vision Encoder（占整体推理 ~72% dense FLOPs）对闭环成功率的精度代价是多少？按 V0 的拆分逐段引入：V1 connector → V2 Vision MLP → V3 attention projection → V4 QK/PV → V5 patch embed（可选）。
- **VLIN（完整 Vision 72-Linear FP8 integration）**：在 canonical VLM/Expert FP8 背景上，把 Vision Encoder **全部 72 个 Linear**（q/k/v/out_proj/fc1/fc2 各 12）一并 FP8 量化、保持 Vision QK/PV 在原 SDPA 实现内、connector 保持原精度，测其净代价：覆盖 347.9 GFLOPs/sample_actions ≈ 81.2% Vision compute / 58% 整体推理。
- 验证现有框架无需改 `model_wrapper.py` 即可完整 wrap 72 个 Vision Linear，并支持分层 calibration policy（legacy reuse / Vision recalibrate）。

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval |
| 仓库 revision / commit | main@2b2a3013；VLIN 分支 `phaseI/vision-linear-full-integration`（main +9 commits，仅存在于 mirror remote） |
| LeRobot 路径与 commit | `lerobot_current/` |
| GPU | GPU 0（H100 系，与同用户 Qwen eval server 共享，EGL 渲染 `MUJOCO_EGL_DEVICE_ID=2`） |
| 关键依赖版本 | 见 `envs/smolvla_eval_pip_freeze.txt`；MUJOCO_GL=egl |

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（450M，16 层 VLM）+ `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 权重 |
| 评测基准 | LIBERO Goal（10 tasks × 10 episodes） |
| 校准数据 | V1：connector recalibrate（8ep/bs8/stride4）。VLIN：G6-A canonical VLM/Expert scale **reuse**（288 sites）；Vision 72 Linear **recalibrate**，scale 目录 `scales/2026-09-15_phaseI_vision-quantization/vlin_full_vision_linear_fp8`（独立目录，先复制 canonical 再标定，不触碰 G6 原始 scale） |

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | 10 tasks × 10 eps = 100 episodes，seed = 1000 |
| 采样参数 | n_action_steps = 10 / num_steps = 10 |
| 指标 | success rate（Wilson 95% CI）、逐 task 成功数 |
| baseline | **G6-A = 90.0%**（VLM/Expert FP8 canonical，VLIN 的直接对照）；H3 S0 FP8-all = 89.0%（V1 对照） |

## 5. 实验变量与分组

<!-- 本实验的核心变量是什么、控制了哪些变量。每个 config 对应一行。 -->

| 组 | config（configs/ 下文件名） | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| V0 | v0_workload_audit.yaml | 无量化 | — | Vision 结构 / 运行时 / FLOPs 审计（12 层、768、2 相机、428.2 G） |
| V1 | v1_connector_fp8.yaml / _goal.yaml | connector `pot_fp8_outlier`（A/W/O 全 E4M3） | VLM/Expert FP8 canonical；vision=false | routing 289；Goal×100 = 81.0%（Δ−8.0pp vs H3） |
| VLIN smoke | vlin_full_vision_linear_fp8.yaml | Vision 72 Linear 全部 `pot_fp8_outlier`（A/W/O E4M3，outlier_ratio=0.01，per-site scale group） | VLM/Expert FP8 reuse G6-A（224 Linear + 64 MatMul）；Vision QK/PV 保持 SDPA；**connector 原精度** | routing **296 Linear / 64 MatMul**（connector 关闭故为 296 非 297）；action = reuse 288 / recalibrate 72；新增 72×3 = 216 个 Vision scale 文件 |
| **VLIN 正式** | vlin_full_vision_linear_fp8_goal.yaml | 同上 | 同上 + `--skip-calibration` | Goal×100 = **80.0%（Δ−10.0pp vs G6-A）** |
| V2–V5 | （待建） | Vision MLP / attn proj / QK·PV / patch embed 逐段 | — | V4 前置 sdpa→eager 等价性 gate |

## 6. 运行命令

<!-- 每个 config 一条可复制执行的命令，标注预期输出目录 -->

```bash
# VLIN 一键 Gate 1–6（静态检查→scale 隔离复制→routing 审计→仅标定→覆盖审计→task0×1 smoke）
bash experiments/2026-09-15_phaseI_vision-quantization/scripts/run_full_vision_linear.sh

# 若本地 G6-A canonical scale 路径不同
BASE_SCALE_DIR=/你的实际路径/g6a_all_fp8_control \
bash experiments/2026-09-15_phaseI_vision-quantization/scripts/run_full_vision_linear.sh

# smoke 通过后的正式 Goal×100（不重复标定）
bash experiments/2026-09-15_phaseI_vision-quantization/scripts/run_full_vision_linear_goal.sh

# 或一次跑到底
RUN_GOAL=1 bash experiments/2026-09-15_phaseI_vision-quantization/scripts/run_full_vision_linear.sh
```

任一 Gate 失败脚本立即退出（fail-loud）。`run_full_vision_linear_goal.sh` 启动时先重查 calibration coverage 再进 eval。

## 7. 输出目录映射

<!-- config → outputs/2026-09-15_phaseI_vision-quantization/ 下的实际产出目录（与实验同名，见 README §5），跑完后逐一登记，便于回溯原始数据 -->

| config | 输出目录（outputs/ 下） | 状态 |
|---|---|---|
| v0_workload_audit.yaml | outputs/2026-09-15_phaseI_vision-quantization/v0_workload_audit/ | done |
| v1_connector_fp8.yaml | outputs/2026-09-15_phaseI_vision-quantization/v1_connector_fp8/ | done |
| v1_connector_fp8_goal.yaml | outputs/2026-09-15_phaseI_vision-quantization/v1_connector_fp8_goal/ | done |
| vlin_full_vision_linear_fp8.yaml | outputs/2026-09-15_phaseI_vision-quantization/vlin_full_vision_linear_fp8/ | done |
| vlin_full_vision_linear_fp8_goal.yaml | outputs/2026-09-15_phaseI_vision-quantization/vlin_full_vision_linear_fp8_goal/ | done（SR 80.0%） |

VLIN scale 目录：`scales/2026-09-15_phaseI_vision-quantization/vlin_full_vision_linear_fp8`（1080 文件 = 864 canonical 复制 + 216 新增）。运行日志：`vlin_full_run.log` / `vlin_goal100.log`（第 1 次尝试被 SIGTERM，已归档 `vlin_goal100.attempt1_terminated.log`）。

## 8. 风险与注意事项

<!-- 已知坑：渲染环境变量、checkpoint 特殊处理、随机性来源等 -->

- **Gate 预期输出（验收标准）**——routing gate 必须打印：

```text
QuantizedLinear total         : 296
legacy VLM/Expert Linear      : 224
Vision Linear                 : 72
Vision attention projection   : 48
Vision MLP                    : 24
Connector Linear              : 0
QuantizedMatMul total         : 64
Vision QuantizedMatMul        : 0

Calibration action plan:
reuse       = 288
recalibrate = 72

ROUTING GATE: PASS
```

  calibration coverage gate 必须打印：

```text
Vision Linear sites : 72 / 72
complete sites      : 72 / 72
Vision scale files  : 216 / 216

q_proj   : 12 / 12
k_proj   : 12 / 12
v_proj   : 12 / 12
out_proj : 12 / 12
fc1      : 12 / 12
fc2      : 12 / 12

CALIBRATION COVERAGE GATE: PASS
```

  每个落盘 scale 必须为有限值且严格为正（审计脚本逐文件打开 pickle 校验，非仅计数）。

- VLIN 分支只在 mirror remote（`git fetch origin` 拉不到，需 `git fetch mirror` 后从 `mirror/phaseI/vision-linear-full-integration` 建分支）。
- runner **无断点续跑**：单进程跑 10 task，`result.json` 仅在全部结束时落盘；中断需从头重跑。判断进度看 `videos/libero_goal_<n>/` mtime（`Built vec env` ×10 是启动时预建，不是进度）。
- 长时 rollout 期间勿编辑仓库内共享 Python 模块（避免 import 失败）；不要设 `CUDA_VISIBLE_DEVICES`（robosuite EGL 断言坑，见 experiments-convention）。
- Vision attention 实现为 `sdpa`（非 eager），本分支刻意不动 Vision QK/PV；V4 前须先做 eager-equivalence gate。
