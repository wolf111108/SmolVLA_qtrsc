# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-08_phaseF_fp8pot-foursuite
- **状态**：done（draft / running / done / aborted）
- **负责人**：zyzhao
- **创建日期**：2026-09-08（补录于 2026-09-11，按 experiments/ 规范迁移归档）
- **相关前序实验**：D4 FP8 粒度 ablation（`outputs/experiments/phaseD_ablation_*`，本实验 F1 协议 = D4 per_site）；w4 粒度 ablation（`outputs/experiments/phaseD_w4_ablation_*`，F3 的 w4 协议 = 其中 per_site 档）

---

## 1. 实验目的

- 在四 suite（spatial/object/goal/10，每组 400 ep）上分离 **outlier 保护的贡献**：F1（有保护）vs F2（无保护）；
- 度量 **int4 权重（w4）的跨 suite 代价**：F1（FP8 权重）vs F3（w4 权重，a/o 仍 FP8）；
- 检验单 suite（libero_object）上「outlier 保护是关键开关」的结论在跨 suite 下是否成立。

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval（mujoco 3.3.2） |
| 仓库 revision / commit | `7d8762e`（h100 侧，含 `pot_ao_outlier` / `pot_fp8_per_tensor` 方法） |
| LeRobot 路径与 commit | `lerobot_current/` @ `6adf5151`（v0.6.2） |
| GPU | NVIDIA H100 NVL（hankh100） |
| 关键依赖版本 | Python 3.12，torch 2.7.1+cu118，MUJOCO_GL=egl |

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（legacy_public，16 层 / 0.75 宽） |
| 评测基准 | LIBERO：libero_spatial / libero_object / libero_goal / libero_10（各 10 task × 10 ep） |
| 校准数据 | `HuggingFaceVLA/libero` v3.0，8 ep，stride 4，seed 42 |

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | 每 suite 10 task × 10 ep（共 100 ep/suite），seed=1000 |
| 采样参数 | n_action_steps=10，num_steps=10（chunk_size=50） |
| 指标 | pc_success（success rate） |

## 5. 实验变量与分组

统一 `per_site` 粒度（Linear 与 MatMul 均是），唯一变量为量化方法/保护策略：

| 组 | config（configs/ 下文件名） | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| F1 | `phaseF1_fp8pot_site_outlier_libero_{suite}.yaml` | `pot_fp8_outlier` + outlier_ratio 0.01 | per_site、ep10、seed1000 | a/w/o 全 FP8(e4m3)+PoT；通道级 x/o + 通道∪元素级 w 的 outlier 保护；名义复用 D4 per_site scale（实际 reuse 未生效，见 §8） |
| F2 | `phaseF2_fp8pot_site_nooutlier_libero_{suite}.yaml` | `pot_fp8_per_tensor`（无任何保护） | 同上 | 纯 per-tensor absmax+PoT；不能用 `outlier_ratio: 0` 替代（`k=max(1,…)` 仍会保护 1 通道，故必须换方法） |
| F3 | `phaseF3_fp8pot_w4_site_outlier_libero_{suite}.yaml` | Linear `pot_ao_outlier`：w_bit=4（连续 scale），a/o FP8+PoT + outlier 0.01 | 同上；MatMul 保持 `pot_fp8_outlier`（A/B/O 全 e4m3） | w 保护 = x 通道 mask OR 元素级 mask（ratio 0.01），outlier 元素保持 FP |

量化范围：VLM text_model 16 层（q/k/v/o/gate/up/down）+ expert 16 层（self_attn + mlp）+ qk/pv matmul；vision encoder / connector / action head 不量化。

## 6. 运行命令

```bash
# 12 次运行（3 组 × 4 suite），幂等（有 eval_info.json 即跳过）
bash experiments/2026-09-08_phaseF_fp8pot-foursuite/scripts/run_phaseF_foursuite.sh

# 单组单 suite（示例）
python main.py --config experiments/2026-09-08_phaseF_fp8pot-foursuite/configs/phaseF1_fp8pot_site_outlier_libero_spatial.yaml
```

实际执行：9-8 01:34 nohup 启动（PID 232112），9-9 17:55 全部完成，总耗时约 40h。

## 7. 输出目录映射

| config | 输出目录（outputs/ 下） | 状态 |
|---|---|---|
| phaseF1_*_libero_spatial | `outputs/experiments/phaseF1_fp8pot_site_outlier_libero_spatial` | done |
| phaseF1_*_libero_object | `outputs/experiments/phaseF1_fp8pot_site_outlier_libero_object` | done |
| phaseF1_*_libero_goal | `outputs/experiments/phaseF1_fp8pot_site_outlier_libero_goal` | done |
| phaseF1_*_libero_10 | `outputs/experiments/phaseF1_fp8pot_site_outlier_libero_10` | done |
| phaseF2_*_libero_spatial | `outputs/experiments/phaseF2_fp8pot_site_nooutlier_libero_spatial` | done |
| phaseF2_*_libero_object | `outputs/experiments/phaseF2_fp8pot_site_nooutlier_libero_object` | done |
| phaseF2_*_libero_goal | `outputs/experiments/phaseF2_fp8pot_site_nooutlier_libero_goal` | done |
| phaseF2_*_libero_10 | `outputs/experiments/phaseF2_fp8pot_site_nooutlier_libero_10` | done |
| phaseF3_*_libero_spatial | `outputs/experiments/phaseF3_fp8pot_w4_site_outlier_libero_spatial` | done |
| phaseF3_*_libero_object | `outputs/experiments/phaseF3_fp8pot_w4_site_outlier_libero_object` | done |
| phaseF3_*_libero_goal | `outputs/experiments/phaseF3_fp8pot_w4_site_outlier_libero_goal` | done |
| phaseF3_*_libero_10 | `outputs/experiments/phaseF3_fp8pot_w4_site_outlier_libero_10` | done |

scale 目录：F2 → `scales/phaseF2_fp8pot_site_nooutlier`，F3 → `scales/phaseF3_fp8pot_w4_site_outlier`；F1 名义复用 `scales/phaseD_ablation_per_site`（实际 reuse 未生效，见 §8）。运行日志：`outputs/phaseF_foursuite.log`。

## 8. 风险与注意事项

- **F1 scale reuse 未生效**：预期复用 D4 per_site（`reuse: 288`），实际三次均 `{'reuse': 0, 'recalibrate': 288}`——cache key 机制待排查；因校准数据/方法相同（F1 object 91.0% 与 D4 per_site 完全一致），结果仍有效；
- F2 不能用 `outlier_ratio: 0` 实现无保护对照（会仍保护 1 通道），故新增 `pot_fp8_per_tensor` 方法；
- ep10 下 100 ep 二项噪声 95% CI ≈ ±5.7pp，单 suite 差异小于此量级时应谨慎归因；
- 与 w4 global 并行运行（各占 ~2.7G 显存），算力分摊约六七成速。
