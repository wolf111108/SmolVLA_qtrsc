# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 weight-granularity（G2）
- **状态**：done（G0/G1/G2-B 完成；G2-C/D/E 被 Gate 2 拦停，仅允许 offline audit）
- **负责人**：zyzhao
- **创建日期**：2026-09-11
- **最后修订**：2026-09-12
- **相关前序实验**：G0 `weight-error-audit`（done）；G1 `component-localization`（done）；父实验 `experiments/2026-09-10_phaseG_w4-root-cause/`

---

## 1. 实验目的

Phase F 已观察到：FP8 Linear（F1）四 suite Avg = 83.0%；W4 Linear（F3）四 suite Avg = 48.5%；`libero_goal` 约 90% → 21%。Phase G 的 G0/G1 已完成根因定位：

| 配置 | Goal SR |
|---|---:|
| G1-A：全 Linear FP8 | 88% |
| G1-B：全 Linear W4 | 21% |
| G1-C：仅 VLM Linear W4 | **21%** |
| G1-D：仅 Expert Linear W4 | **84%** |

因此：**F3 的主要精度退化来自 VLM 侧 W4，而不是 Action Expert W4。**

G0 同时表明：W4 weight NMSE 在全模型中较均匀；VLM/Expert 数值误差仅存在有限差异；未发现足以单独解释 SR 崩溃的极端异常 Linear site；saturation≈0；INT4 code 基本被正常利用。

因此 G2 不再回答「哪个 component 导致 W4 崩溃」，而专门回答：

> **VLM 的 W4 崩溃是否主要由当前 tensor-wise weight scale 粒度过粗造成？**

当前实现虽然使用 `linear_scale_granularity: per_site`，但 `per_site` 表示不同 physical Linear site 分别拥有 scale，并不表示一个 `[N_out, K]` weight matrix 内部采用 per-channel / group-wise scale。当前 W4 本质仍是 $W\in\mathbb{R}^{N_{out}\times K} \rightarrow s_W\in\mathbb{R}$（整个 weight tensor 共用一个 scalar scale）。

本实验验证 per-tensor → per-output-channel → group-wise W4 是否能恢复 VLM 的闭环 Success Rate。

### 核心假设（Gate 判据）

- 若 $SR_{per\text{-}channel} - SR_{per\text{-}tensor} \ge 15pp$：W4 scale granularity 是 F3 的重要根因；
- 若 $SR_{per\text{-}channel} \ge 70\%$：tensor-wise W4 是主要瓶颈，W4 本身仍具可用性；
- 若 $SR_{per\text{-}channel} \ge 80\%$：基本确认 VLM 并非本质不能用 INT4，而是 tensor-wise scaling 过粗；
- 若 $SR_{per\text{-}channel} \le 30\%$：per-output-channel 无法解释 F3，需研究 group-wise / VLM layer·operator sensitivity / activation-conditioned quantization / outlier-scale mismatch。

## 2. 环境与版本

除本实验新增的 W4 granularity 实现外，环境必须严格保持 G0/G1 不变。

| 项目 | 值 |
|---|---|
| conda 环境 | `smolvla_eval` |
| MuJoCo | `3.3.2` |
| LeRobot 路径 | `lerobot_current/` |
| LeRobot commit | `6adf5151...`（v0.6.2） |
| GPU | NVIDIA H100 NVL |
| Python / torch | 3.12 / 2.7.1+cu118 |
| rendering | `MUJOCO_GL=egl` |
| 仓库 revision | **开跑前用 `git rev-parse HEAD` 回填**（须含 tensor-aware weight scale 实现） |

本 task 会修改 quantization framework，因此正式运行 G2-B 前必须记录并校验（mujoco=3.3.2、MUJOCO_GL=egl）。正式结果必须绑定到**包含 tensor-aware weight scale 实现的 commit**。

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（legacy_public；VLM 16 层 + Expert 16 层） |
| 主诊断 suite | `libero_goal` |
| calibration dataset | `HuggingFaceVLA/libero` v3.0，8 ep，batch 8，stride 4，seed 42 |
| scale reuse | 不跨 granularity 复用，每 config 独立 scale_dir |

### 3.1 G2 的量化范围（基于 G1 的正式 Gate 决策，只量化 VLM）

```text
VLM Linear:            weight=INT4, a/o=E4M3+PoT, outlier=0.01
Action Expert Linear:  raw FP
MatMul (QK/PV):        FP8 E4M3 + PoT + outlier（G1-C 协议不变）
Vision Encoder / Connector / Action Head: raw FP
```

G2-A 与 G1-C 完全对应（VLM W4 per-tensor + Expert FP = Goal SR 21%）。G2 中唯一研究变量是 **VLM W4 tensor 内部的 weight scale granularity**。

### 3.2 新增配置字段（与 `linear_scale_granularity` 正交）

```yaml
weight_quant_granularity: per_tensor   # per_tensor / per_output_channel / groupwise
weight_group_size: null                # groupwise 时填 128/64/32，沿 in_features=K 分组
```

- per-tensor：$s_W\in\mathbb{R}$；
- per-output-channel：$s_W\in\mathbb{R}^{N_{out}}$，每 output row 用 normal 部分 absmax / $q_{max}$；
- group-wise：$s_W\in\mathbb{R}^{N_{out}\times N_{group}}$，每 row 沿 K 分组，每组单独 absmax。

## 4. 评测协议

正式闭环评测保持 G1 协议不变：libero_goal，10 task × 10 ep，seed=1000，batch=1，max_parallel_tasks=1，`n_action_steps=10`，`num_steps=10`，chunk=50；主指标 `pc_success`，辅助 task-wise SR（重点 t0/t1/t2/t4/t7/t9，其中 t1/t2/t4/t7 为 VLM W4 崩溃组）。统一 rename：image→camera1，image2→camera2。

控制变量（全部固定）：checkpoint、calibration dataset/sampling、VLM quantization scope、Expert=FP、activation/output 精度、MatMul 精度、outlier ratio、PoT、seed、MuJoCo、n_action_steps、num_steps、episode horizon。唯一正式实验变量：`weight_quant_granularity`。

## 5. 实验变量与分组

采用两级 Gate，不一次性启动五个 100-episode config：

```text
G2-P0 numerical preflight → PASS → G2-B per-channel closed-loop → 按结果决定 group-wise
```

### 5.1 G2-P0 — Numerical preflight（无闭环 rollout）

比较 VLM per-tensor W4 vs per-output-channel W4（全部 VLM QuantizedLinear）。每层输出 module_name/component/layer_idx/op_type/weight_shape/weight_scale_shape/weight_nmse/weight_sqnr_db/weight_cosine/quant_zero_ratio/saturation_ratio/unique_quant_codes/protected_ratio/output_nmse/output_sqnr_db/output_cosine。交付：`per_tensor_linear_stats.csv`、`per_channel_linear_stats.csv`、`granularity_comparison.csv`、`operator_summary.csv`、`layer_summary.csv`。

**PASS 条件**：① per-channel 无 NaN/Inf；② scale shape 与 out_features 一致；③ 未保护 weight 的 saturation 无异常；④ 绝大多数 Linear 满足 $NMSE_{channel}\le NMSE_{tensor}$；⑤ aggregate $mean(NMSE_{channel})<mean(NMSE_{tensor})$；⑥ enabled=false/raw 路径与修改前 bit-equivalent；⑦ per-tensor 新实现与旧 G1-C 数值路径等价。任一失败 → **STOP**，不得进入闭环 G2-B。

### 5.2 正式闭环分组

| 组 | config | VLM W4 granularity | Expert | 状态/说明 |
|---|---|---|---|---|
| G2-A | `g2a_per_tensor_vlm.yaml` | per_tensor | raw FP | **不重跑，复用 G1-C=21%**（config 仅作 manifest） |
| G2-B | `g2b_per_channel_vlm.yaml` | per_output_channel | raw FP | **第一优先正式 rollout** |
| G2-C | `g2c_group128_vlm.yaml` | groupwise, G=128 | raw FP | conditional |
| G2-D | `g2d_group64_vlm.yaml` | groupwise, G=64 | raw FP | conditional |
| G2-E | `g2e_group32_vlm.yaml` | groupwise, G=32 | raw FP | conditional |

统一：VLM Linear W=INT4、A/O=E4M3+PoT、outlier=0.01、MatMul=FP8+PoT+outlier、Expert raw FP、`linear_scale_granularity=per_site`、na=10、num_steps=10。

### 5.3 Gate 1：per-channel 决策（对照 $SR_{tensor}=21\%$）

| G2-B Goal SR | 判定 |
|---:|---|
| ≥80% | tensor-wise granularity 基本确认是主因；进入 group-wise 做 hardware trade-off |
| 70–79% | granularity 是主要因素；继续 group-wise |
| 50–69% | granularity 有明显贡献但不是全部根因；继续 group-wise + 后续 VLM sensitivity |
| 31–49% | 贡献有限；groupwise 先 offline audit 再决定 rollout |
| ≤30% | per-channel 基本失败；不立即烧 groupwise 全套 rollout |

同时保留原始 Gate：$\Delta SR_{channel}\ge 15pp$ 视为 granularity 有显著贡献。

### 5.4 Gate 2：group-wise 是否进入闭环（只有 G2-B 完成后决定）

- G2-B ≥ 50%：正式实现 G=128/64/32 并逐档 rollout；
- G2-B < 50%：先只做 groupwise offline numerical audit（G128/64/32），仅当某档 $NMSE_{group}\ll NMSE_{channel}$ 才进入 100-episode rollout。避免在机制未定时直接消耗 3×100 episodes。

## 6. 运行命令

```bash
cd ~/VLA_tcs2
EXP=experiments/2026-09-10_phaseG_w4-root-cause
TASK=$EXP/tasks/weight-granularity

# 6.1 G2-P0 数值 preflight（只跑 per_tensor / per_output_channel，无 rollout）
bash $TASK/scripts/run_granularity_preflight.sh

# 6.2 G2-B per-channel 正式 Goal ×100（P0 PASS 后）
bash $TASK/scripts/run_weight_granularity.sh --stage per-channel

# 6.3 Group-wise（conditional，Gate 2 允许后，顺序 group128→64→32）
bash $TASK/scripts/run_weight_granularity.sh --stage groupwise
```

脚本必须：① 检查 output 是否完整存在（完整 eval_info.json 则 skip，不完整目录不得错误标记 done）；② 使用独立 scale_dir；③ 将命令和 commit 写入 log。**不允许在 per-channel 结果未知时直接后台启动全部 group-wise rollout。**

## 7. 输出目录映射

```text
outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-granularity/
```

| config / stage | 输出目录 | 状态 |
|---|---|---|
| G2-P0 per-tensor | `.../weight-granularity/preflight/per_tensor/` | pending |
| G2-P0 per-channel | `.../weight-granularity/preflight/per_channel/` | pending |
| G2-A per-tensor anchor | **复用 G1-C 原始结果（21%）** | done / reused |
| g2b_per_channel_vlm.yaml | `.../weight-granularity/g2b_per_channel_vlm/` | pending |
| g2c_group128_vlm.yaml | `.../weight-granularity/g2c_group128_vlm/` | conditional |
| g2d_group64_vlm.yaml | `.../weight-granularity/g2d_group64_vlm/` | conditional |
| g2e_group32_vlm.yaml | `.../weight-granularity/g2e_group32_vlm/` | conditional |

scale 目录（每档独立，禁止 per_tensor scale 被 per_channel/groupwise reuse）：

```text
scales/2026-09-10_phaseG_w4-root-cause/weight-granularity/{per_tensor,per_channel,group128,group64,group32}/
```

## 8. 风险与注意事项

1. **`per_site` 与 `per_output_channel` 是两个不同维度**（site 级共享 vs tensor 内共享），二者不能合并；
2. **`w_interval` 不再一定是 float**：per_output_channel 为 `[N_out]`、groupwise 为 `[N_out, N_group]`——必须同步检查 quant_linear.py / quant_methods.py / scale_methods.py / stat_manager.py / scale save·load / reuse validation / SQNR 统计 / bias path，禁止只改 scale 计算而不改 persistence/forward；
3. **per-channel scale 广播方向必须显式控制**：$s_W\in[N_{out}]$ 必须作用于 output 最后一维，不依赖 implicit broadcasting；
4. **outlier 四路径语义保持**：$Y=Y_{Q_aQ_w}+Y_{F_aQ_w}+Y_{Q_aF_w}+Y_{F_aF_w}$，per-channel 后 $Y_{Q_aQ_w,j}=s_a s_{w,j}\sum_k q(a_k)q(w_{jk})$ 等；protected weight 仍走 FP path，不因 per-channel scale 改动而再次参与 W4 absmax；
5. **group-wise 不能在完整 F.linear 后乘 scale**：scale 位于 K reduction 内，必须「K 分组 → group partial dot → × group scale → accumulate」，否则数值语义错误且无法对应硬件 datapath；
6. **G2 只量化 VLM** 是 G1 的正式 Gate 决策（VLM-only W4=21%、Expert-only=84%），不是为省算力随意改协议；
7. **暂不修改 action horizon**：整个 G2 保持 na=10；na∈{1,5,10} 属于 G3，禁止同时改 granularity 与 horizon；
8. **实验结束后的决策**：若 G2-B 显著恢复 → 优先 groupwise W4 做 accuracy/metadata/hardware-cost trade-off；若 per-channel 与 groupwise 都无法恢复 → 不优先扫 outlier ratio，进入 VLM selective-precision（attention vs MLP、浅层 vs 深层、q/k/v/o vs gate/up/down）；若得到可用 W4 配置 → 部署候选（M1：VLM FP8 + Expert W4 + MatMul FP8；M2：VLM fine-grain W4 + Expert W4 + MatMul FP8）之后做四 suite 完整 benchmark。
