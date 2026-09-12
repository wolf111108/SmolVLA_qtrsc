# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 weight-error-audit（G0）
- **状态**：pending（draft / running / done / aborted）
- **负责人**：zyzhao
- **创建日期**：2026-09-11
- **相关前序实验**：父实验 `experiments/2026-09-10_phaseG_w4-root-cause/docs/experiment_setup.md`（Phase G 总纲）

---

## 1. 实验目的

**离线数值审计，不做闭环 rollout。** 判断 F3 的 W4 退化是源于少量极端敏感 Linear，还是全模型普遍低 SQNR：对全部 224 个物理 Linear 在两套量化配置（FP8 vs W4）下计算 weight 与 output 的数值误差，输出排行与聚合视图，为 G1/G2 圈定优先区域。

**G0 判据**：若最低 SQNR / 最大 output-NMSE 明显集中于某一 component / operator / layer range，后续 G1/G2 优先围绕该区域。
## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval（无需渲染，不设 MUJOCO_GL） |
| 仓库 revision / commit | 开跑时 `git rev-parse HEAD` 写入此处 |
| LeRobot 路径与 commit | `lerobot_current/` @ `6adf5151`（v0.6.2） |
| GPU | NVIDIA H100 NVL（hankh100） |
| 关键依赖版本 | Python 3.12，torch 2.7.1+cu118 |

> 注：本 task 无 rollout，无需 mujoco / EGL。

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（legacy_public） |
| 对象 | 全部 224 个物理 Linear（VLM 16 层 × 7 + Expert 16 层 × 7） |
| 前向数据 | `HuggingFaceVLA/libero` v3.0，8 ep，stride 4，seed 42（与校准同源，保证与 F3 运行时分布一致） |

两套量化配置（与 Phase F F1/F3 同协议，独立 scale_dir）：

| Group | Linear W | Linear A/O | MatMul |
|---|---|---|---|
| G0-FP8 | E4M3 | E4M3+PoT | E4M3+PoT |
| G0-W4 | INT4 | E4M3+PoT | E4M3+PoT |

## 4. 评测协议

| 项目 | 值 |
|---|---|
| rollout | 无（纯前向数值审计） |
| 每个 Linear 输出字段 | module_name / component(vlm·expert) / layer_idx / op_type(q·k·v·o·gate·up·down) / weight_shape / weight_scale / weight_sqnr_db / weight_nmse / weight_cosine / quant_zero_ratio / saturation_ratio / unique_quant_codes / protected_ratio / output_sqnr_db / output_nmse / output_cosine |
| 交付物 | `linear_error_stats.csv`、`linear_error_ranked.csv`、`component_summary.csv`、`operator_summary.csv`、`layer_summary.csv` |

排序重点：VLM vs Expert；QKV/O vs Gate/Up/Down；layer 0→15 趋势。

唯一变量：Linear 权重精度（FP8 vs INT4）。其余全部固定（per_site、outlier 0.01、MatMul FP8、同一前向数据）。

| 组 | config（configs/ 下文件名） | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| G0-FP8 | `g0_fp8.yaml` | Linear W=E4M3 | per_site / outlier 0.01 / MatMul FP8 | = F1 协议，误差基准 |
| G0-W4 | `g0_w4.yaml` | Linear W=INT4（连续 scale） | 同上 | = F3 协议，待归因对象 |

实现说明：审计脚本可复用 scale_inspection 前向 + 各层落盘的 a/w/o scale，按 `quant_awo` 模拟量化后与 FP 参考计算指标；不启动 eval。

## 6. 运行命令

<!-- 每个 config 一条可复制执行的命令，标注预期输出目录 -->

```bash
EXP=experiments/2026-09-10_phaseG_w4-root-cause
bash $EXP/tasks/weight-error-audit/scripts/run_weight_error_audit.sh
# 或两步：先校准落 scale，再审计出 CSV
python main.py --config $EXP/tasks/weight-error-audit/configs/g0_fp8.yaml --skip-evaluation
python main.py --config $EXP/tasks/weight-error-audit/configs/g0_w4.yaml --skip-evaluation
python $EXP/tasks/weight-error-audit/scripts/audit_weight_error.py
```

## 7. 输出目录映射

<!-- config → outputs/2026-09-10_phaseG_w4-root-cause · 子实验 weight-error-audit/ 下的实际产出目录（与实验同名，见 README §5），跑完后逐一登记，便于回溯原始数据 -->

| config / 脚本 | 输出目录 | 状态 |
|---|---|---|
| g0_fp8.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-error-audit/fp8/` | pending |
| g0_w4.yaml | `outputs/2026-09-10_phaseG_w4-root-cause/tasks/weight-error-audit/w4/` | pending |
| audit_weight_error.py | 同上两目录内 `linear_error_stats.csv` 等 5 个 CSV | pending |

scale：`scales/2026-09-10_phaseG_w4-root-cause/weight-error-audit/{fp8,w4}/`。

## 8. 风险与注意事项

- outlier 保护（1%）下「有效」W4 误差指标应只统计 normal 部分，protected_ratio 单独记录，避免把 FP 通道计入 SQNR 拉高假象；
- output 指标的前向输入应使用同配置上游量化后的激活（而非 FP 激活），否则会低估误差传播；若实现成本高，第一版可先用 FP 激活 + 在 results 中注明口径；
- E1 前向敏感度排序（`outputs/sensitivity/sensitivity.csv`）已证明 max_abs_diff 不能预测闭环 SR——G0 结果只用于圈定区域，归因仍以 G1 闭环为准。
