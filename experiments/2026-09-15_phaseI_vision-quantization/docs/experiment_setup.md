# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。
> 本文档由原中文模板与协作者英文设计文档 `full_vision_linear_integration.md`（VLIN 部分）合并翻译而成（2026-09-21）。

- **实验名称**：2026-09-15_phaseI_vision-quantization
- **状态**：running（V0 ✅ / V1 ✅ / VLIN ✅ / V2 ✅ / V3 ✅ / V4–V5 待跑 / Vision sparsity-compute task ready）
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
| 仓库 revision / commit | 基线 `main@2b2a3013`；当前工作分支 `phaseI/vision-linear-full-integration`（后续修订均在该分支，不再用易过期的 `+N commits / N files` 描述） |
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
| baseline | **VLIN 严格直接对照：G6-A = 90.0%**（同一 VLM/Expert FP8 scale，Vision/connector raw）；**V1 仅参考 H3 S0 = 89.0%**，因 V1 对 legacy VLM/Expert scale 重新 calibration，不能视为严格 connector-only 单变量对照 |

## 5. 实验变量与分组

<!-- 本实验的核心变量是什么、控制了哪些变量。每个 config 对应一行。 -->

| 组 | config（configs/ 下文件名） | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| V0 | v0_workload_audit.yaml | 无量化 | — | Vision 结构 / 运行时 / FLOPs 审计（12 层、768、2 相机、428.2 G） |
| V1 | v1_connector_fp8.yaml / _goal.yaml | connector `pot_fp8_outlier`（A/W/O 全 E4M3） | vision=false；legacy VLM/Expert 也按 V1 配置重新 calibration | routing 289；Goal×100 = 81.0%；相对 H3 S0 89.0% 为 −8.0pp，但**不是严格 connector-only 单变量差值** |
| VLIN smoke | vlin_full_vision_linear_fp8.yaml | Vision 72 Linear 全部 `pot_fp8_outlier`（A/W/O E4M3，outlier_ratio=0.01，per-site scale group） | VLM/Expert FP8 reuse G6-A（224 Linear + 64 MatMul）；Vision QK/PV 保持 SDPA；**connector 原精度** | routing **296 Linear / 64 MatMul**（connector 关闭故为 296 非 297）；action = reuse 288 / recalibrate 72；新增 72×3 = 216 个 Vision scale 文件 |
| **VLIN 正式** | vlin_full_vision_linear_fp8_goal.yaml | 同上 | 同上 + `--skip-calibration` | Goal×100 = **80.0%（Δ−10.0pp vs G6-A）** |
| V2 | `v2_vision_mlp_fp8.yaml` / `_goal.yaml` | Vision MLP `fc1/fc2` 24 个 Linear FP8 | G6-A 288 个 legacy quant sites 全 reuse；Vision attn projection / QK·PV / connector raw | **done：84.0%，L_MLP=6pp**；覆盖 231.93 GFLOPs ≈ 54.2% Vision / 38.9% whole inference |
| V3 | `v3_vision_attn_proj_fp8.yaml` / `_goal.yaml` | Vision `q/k/v/out_proj` 48 个 Linear FP8 | G6-A 288 个 legacy quant sites 全 reuse；Vision MLP / QK·PV / connector raw | **done：85.0%，L_Attn=5pp**；覆盖 115.96 GFLOPs ≈ 27.1% Vision / 19.5% whole inference |
| V4 | （待建） | Vision QK/PV | V2/V3 结论确定后再进入 | **前置：sdpa → eager 等价性 gate** |
| V5 | （可选） | patch embed / 其他 Vision 非 Linear op | 前述实验完成后再决定 | 不作为当前优先级 |

### 5.1 VLIN 精度损失归因设计（下一阶段，统一放在本实验内）

VLIN 已得到严格对照结果：G6-A 90.0% → Vision 72-Linear FP8 80.0%，即 **L_all = 10pp**。下一步不新建规范外实验目录，继续在本实验 `configs/`、`scripts/`、`docs/` 内完成两组单变量消融：

| 组 | Vision MLP | Vision Attn Projection | Vision QK/PV | Connector | 期望 routing | calibration action | 目的 |
|---|---|---|---|---|---|---|---|
| G6-A baseline | raw | raw | raw/SDPA | raw | 224 Linear + 64 MatMul | 直接复用既有 G6-A | 90.0% anchor |
| V2 MLP-only | **FP8** | raw | raw/SDPA | raw | **248 Linear + 64 MatMul** | reuse 288 / recalibrate **24** | 测 `L_MLP = 90 - SR_V2` |
| V3 AttnProj-only | raw | **FP8** | raw/SDPA | raw | **272 Linear + 64 MatMul** | reuse 288 / recalibrate **48** | 测 `L_Attn = 90 - SR_V3` |
| VLIN all-Linear | **FP8** | **FP8** | raw/SDPA | raw | **296 Linear + 64 MatMul** | reuse 288 / recalibrate **72** | 已完成，`L_all = 10pp` |

固定量化设置保持与 VLIN 完全一致：`pot_fp8_outlier`、A/W/O = E4M3、`outlier_ratio=0.01`、`linear_scale_granularity=per_site`；VLM/Expert 继续复用**同一份 G6-A canonical scales**，避免再次引入 calibration background 混淆。

每组仍按同一 Gate 顺序执行：routing/count → calibration-only → coverage → Goal task0×1 smoke → Goal×100（seed=1000，n_action_steps=10，num_steps=10）。预期新增 Vision scale 文件数：V2 = 24×3 = **72**，V3 = 48×3 = **144**。

归因判据：

```text
L_MLP  = 90 - SR(V2)
L_Attn = 90 - SR(V3)
L_all  = 90 - 80 = 10
interaction = L_all - (L_MLP + L_Attn)
```

V2/V3 已完成：`L_MLP=6pp`、`L_Attn=5pp`、`L_all=10pp`，因此 `interaction=-1pp≈0`，两部分损失在当前 100-ep 分辨率下近似可加。该结果说明 Vision Linear 的 FP8 适配性总体可接受，后续在进入 V4 QK/PV 前先补齐加入 Vision 后的 sparsity / compute workload characterization。

> 运行时间 `eval_s` 只用于记录 rollout 成本，**不得直接解释为 FP8 硬件加速比**：当前框架是 fake/simulated quantization，包含量化/反量化与 outlier side path，且 SR 不同会改变 episode 是否跑满 300 steps。真实速度收益需后续用实际低精度 kernel / 硬件路径单独测量。

### 5.2 Vision sparsity + compute task（2026-09-22）

已按 `experiments/README.md §7` 创建标准子实验：

`tasks/vision-sparsity-compute/`

该 task 不创建额外设计文档，完整设计统一放在：

`tasks/vision-sparsity-compute/docs/experiment_setup.md`

目标是在 **VLIN 完整 72-Linear FP8** 上统计：

- Vision / VLM / Expert 的 runtime **native element sparsity**
- Vision / VLM / Expert 的 runtime **native bit sparsity**
- QuantizedLinear static weight element / bit sparsity
- per-`sample_actions()` MAC/FLOP 组成
- 当前量化 major-op FLOP coverage

统计明确区分量化范围与 raw 范围：Vision 72 Linear 纳入 sparsity；Vision SDPA QK/PV 与 connector 保持 raw，仅纳入 compute accounting，不进入 quantized sparsity denominator。Primary 只跑 Goal task0 × 1ep，并强制 `--skip-calibration` 复用 VLIN 已有 scales。

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
| task: vision-sparsity-compute / vsc_vlin_fp8_task0_1ep.yaml | outputs/2026-09-15_phaseI_vision-quantization/tasks/vision-sparsity-compute/vsc_vlin_fp8_task0_1ep/ | pending |

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
- **性能口径**：当前 PoT-FP8 是 fake/simulated quantization；`eval_s` 同时受失败 episode 步数、GPU 争用和量化仿真开销影响，因此只记录运行成本，不作为硬件 speedup 证据。
