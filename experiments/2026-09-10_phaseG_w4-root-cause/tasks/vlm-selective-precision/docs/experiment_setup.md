# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 vlm-selective-precision（G5）
- **状态**：running（draft / running / done / aborted）
- **负责人**：zyzhao
- **创建日期**：2026-09-13
- **相关前序实验**：父实验 Phase G 总纲；G0（数值均匀）；G1（VLM 主导 21% vs 84%）；G2（per-channel 失败 23%，Gate 2 拦停 groupwise）

---

## 1. 实验目的

把「VLM 对 W4 敏感」进一步缩小到「VLM 的**哪一种计算**对 W4 敏感」：Attention Linear（q/k/v/o）还是 MLP Linear（gate/up/down）。

G2 已排除 scale 粒度（per-channel 仅 +2pp），继续优化 groupwise 收益低；本实验不新增框架功能（现有 per-operator w_bit/method 覆盖已支持），直接在 VLM 内部做 selective-precision 定位。

判读：

- 若 G5-B ≈ 21%、G5-C ≈ G5-A → **Attention Linear 是 W4 主敏感区**，下一步只对 q/k/v/o 做 shallow/deep layer split；
- 若 G5-C ≈ 21%、G5-B ≈ G5-A → **MLP 主导**，下一步只拆 gate/up/down；
- 若 G5-B、G5-C 都 >60% 而 G5-D=21% → **不是单一模块脆弱，而是 Attention+MLP W4 误差跨 residual block 累积**，下一步做 hidden-state propagation drift；
- 否则按降幅排序定位。

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | `smolvla_eval`（mujoco 3.3.2） |
| 仓库 revision / commit | 开跑时 `git rev-parse HEAD` 回填（当前含 G2 的 `pot_ao_outlier_channel` 实现，但本实验用 per-tensor） |
| LeRobot 路径与 commit | `lerobot_current/` @ `6adf5151`（v0.6.2） |
| GPU | NVIDIA H100 NVL（hankh100） |
| 关键依赖版本 | Python 3.12，torch 2.7.1+cu118，`MUJOCO_GL=egl` |

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（legacy_public） |
| 评测基准 | libero_goal（10 task × 10 ep） |
| 校准数据 | `HuggingFaceVLA/libero` v3.0，8 ep，batch 8，stride 4，seed 42 |
| 量化范围 | VLM Linear（q/k/v/o + gate/up/down）；Expert Linear = raw FP；MatMul QK/PV = FP8；vision/connector/action head = FP |

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | 100 ep/config，seed=1000，batch=1，max_parallel_tasks=1，async=false |
| 采样参数 | `n_action_steps=10`，`num_steps=10`，chunk_size=50 |
| 指标 | pc_success + task-wise SR（重点崩溃组 t1/t2/t4/t7） |
| rename_map | image→camera1，image2→camera2 |

唯一变量：VLM 内部 Attention vs MLP 的 Linear 精度。所有 W4 恢复 **per-tensor**（`weight_quant_granularity: per_tensor`，G2 已否决 per-channel）。Expert Linear 全程 raw FP，MatMul 全程 FP8，保证 G5 内部 control 条件一致。

| 组 | config | VLM Attention q/k/v/o | VLM MLP gate/up/down | Expert | 目的 |
|---|---|---|---|---|---|
| G5-A | `g5a_vlm_fp8_control.yaml` | FP8 | FP8 | raw FP | 精确 control（重新跑，非复用 G1-A） |
| G5-B | `g5b_vlm_attn_w4_mlp_fp8.yaml` | **W4** | FP8 | raw FP | Attention 敏感度 |
| G5-C | `g5c_vlm_attn_fp8_mlp_w4.yaml` | FP8 | **W4** | raw FP | MLP 敏感度 |
| G5-D | 复用 G1-C | W4 | W4 | raw FP | 复用 21%，不重跑 |

> 全局 `method: pot_fp8_outlier`；W4 的 operator 用 per-layer `method: pot_ao_outlier` + `w_bit: 4` 覆盖；FP8 operator 无 method 覆盖（回落全局）。A/O 全 FP8 E4M3 + PoT，outlier_ratio 0.01。

## 6. 运行命令

```bash
EXP=experiments/2026-09-10_phaseG_w4-root-cause
TASK=$EXP/tasks/vlm-selective-precision

# 批量（内置 routing smoke + 3×100ep，幂等）
bash $TASK/scripts/run_vlm_selective_precision.sh

# 单个（示例）
MUJOCO_GL=egl python main.py --config $TASK/configs/g5b_vlm_attn_w4_mlp_fp8.yaml
```

开跑前已先跑 config-routing smoke（`g5_routing_smoke.py`，三个 config 均通过：G5-A 全 FP8 / G5-B attn W4 / G5-C mlp W4，Expert 0 QuantizedLinear，MatMul 64）。

## 7. 输出目录映射

| config | 输出目录（outputs/ 下） | 状态 |
|---|---|---|
| g5a_vlm_fp8_control.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-precision/g5a_vlm_fp8_control/` | pending |
| g5b_vlm_attn_w4_mlp_fp8.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-precision/g5b_vlm_attn_w4_mlp_fp8/` | pending |
| g5c_vlm_attn_fp8_mlp_w4.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-precision/g5c_vlm_attn_fp8_mlp_w4/` | pending |
| G5-D（复用 G1-C） | `tasks/component-localization/g1c_w4_vlm_only/` | done/reused |

scale：`scales/2026-09-10_phaseG_w4-root-cause/vlm-selective-precision/g5{a,b,c}_*/`（独立）。

## 8. 风险与注意事项

- **开跑前必须 routing smoke**（G2-B 的 0% 历史教训）：已通过，确认 per-operator w_bit/method 正确路由、Expert 保持 raw；
- 本实验**无需新增框架功能**（per-operator method/w_bit 覆盖已支持）；
- W4 全部 **per-tensor**（G2 已否决 per-channel），不要设 `weight_quant_granularity: per_output_channel`；
- G5-A 需**重新跑**（不能复用 G1-A=88%：G1-A 是「所有 Linear FP8」，本实验 Expert 是 raw FP，control 条件不同）；
- groupwise（group128/64/32）闭环仍 halt（G2-B=23%<50%），本实验不做；
- 归因阈值沿用：ΔSR ≥15pp 主贡献，6–15pp 中等，≤6pp 不做强归因（±5.7pp 噪声）。
