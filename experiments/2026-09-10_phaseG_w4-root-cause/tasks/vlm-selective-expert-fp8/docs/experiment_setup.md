# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 vlm-selective-expert-fp8
- **状态**：draft（draft / running / done / aborted）
- **负责人**：
- **创建日期**：2026-09-14
- **相关前序实验**：`../vlm-selective-precision/`（G5，Expert raw FP 背景的 VLM selective precision）

---

## 1. 实验目的

验证 G5 得到的「VLM 内部 MLP > Attention 的 W4 敏感性」在 **Expert 也量化为 FP8** 的实际部署背景下是否仍成立。

核心问题：当 VLM 与 Expert 都进入量化部署域后，VLM 内部 MLP 仍是 W4 的主要 accuracy bottleneck 吗？

依赖升级：`model_wrapper.py` 新增 `linear.overrides`（component/module-aware precision routing），使同一次 run 能表达「VLM q_proj=W4 + Expert q_proj=FP8」这类混合精度。

---

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval |
| 仓库 revision / commit | `git rev-parse HEAD` 的值（升级后） |
| LeRobot 路径与 commit | `lerobot_current/` @ `6adf5151` |
| GPU | NVIDIA H100 NVL |
| 关键依赖版本 | Python 3.12, torch 2.7.1+cu118, mujoco 3.3.2, `MUJOCO_GL=egl` |

---

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（legacy_public） |
| 评测基准 | LIBERO goal |
| 校准数据 | HuggingFaceVLA/libero@v3.0，8 episodes，每 config 独立 recalibrate（禁止复用 G5 scale） |

---

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | 10 tasks × 10 eps，seed=1000 |
| 采样参数 | n_action_steps=10, num_steps=10 |
| 指标 | success rate（Goal SR） |

---

## 5. 实验变量与分组

核心变量：VLM 内部哪些 Linear 用 W4（Attention / MLP / 全部），Expert 固定 FP8 背景。

| 组 | config | VLM Attention | VLM MLP | Expert Linear | QK/PV |
|---|---|---|---|---|---|
| G6-A | g6a_all_fp8_control.yaml | FP8 | FP8 | FP8 | FP8 |
| G6-B | g6b_vlm_attn_w4_expert_fp8.yaml | **W4** | FP8 | FP8 | FP8 |
| G6-C | g6c_vlm_mlp_w4_expert_fp8.yaml | FP8 | **W4** | FP8 | FP8 |
| G6-D | g6d_vlm_all_w4_expert_fp8.yaml | **W4** | **W4** | FP8 | FP8 |

固定项：`linear_scale_granularity=per_site`、`weight_quant_granularity=per_tensor`、`outlier_ratio=0.01`、MatMul A/B/O=E4M3、method 默认 `pot_fp8_outlier`（W4 用 `pot_ao_outlier`）。

预期 routing count：

| 组 | FP8 Linear | W4 Linear | MatMul |
|---|---:|---:|---:|
| G6-A | 224 | 0 | 64 |
| G6-B | 160 | 64 | 64 |
| G6-C | 176 | 48 | 64 |
| G6-D | 112 | 112 | 64 |

---

## 6. 运行命令

```bash
# 生成 config（4 个）
python experiments/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/scripts/make_g6_configs.py

# 路由审计（Gate 2）
python experiments/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/scripts/audit_routing.py \
  --config experiments/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/configs/g6a_all_fp8_control.yaml

# calibration-only smoke（Gate 4）
bash experiments/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/scripts/run_smoke.sh

# Goal ×100 正式（Gate 6）
bash experiments/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/scripts/run_goal.sh
```

预期输出目录：`outputs/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/<config_name>/`

---

## 7. 验收 Gate

- [x] Gate 0：`py_compile` PASS
- [x] Gate 1：`resolve_linear_quant_config` 7 个单测 PASS
- [x] Gate 2：四组 routing count 全部符合预期（G6-A 224/0、B 160/64、C 176/48、D 112/112，MatMul 64）
- [ ] Gate 3：raw Linear equivalence（升级不改变 forward 数值，no-override 行为不变）
- [ ] Gate 4：calibration-only smoke 四组 PASS
- [ ] Gate 5：task0×1 smoke 四组 PASS
- [ ] Gate 6：Goal ×100 四组完成

---

## 8. 结果分析对照

原 G5（Expert raw FP）：90 / 72 / 39 / 21（control / attn-W4 / mlp-W4 / all-W4）。

G6（Expert FP8）预期对照，定义 L_attn=G6A-G6B、L_mlp=G6A-G6C、L_all=G6A-G6D。若 |L - G5 对应值| ≤ 6pp 且仍有 L_mlp >> L_attn，则结论对 Expert precision background 鲁棒。
