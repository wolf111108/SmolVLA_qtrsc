# 实验设置（Experiment Setup）

- **实验名称**：2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal
- **状态**：done（2026-09-25 在 commit `4cdccc4` 上按本文件协议执行完毕，结果见 results.md / logs.md）
- **创建日期**：2026-09-25
- **相关前序实验**：2026-09-25_phaseI_vision-joint-smoke（VJ1）；2026-09-23_phaseI_vision-attention-smoke

## 1. 实验目的

将 Vision 与 VLM 作为同一精度组，使用 WINT8 + FP8 A/O；Expert 使用 WINT4 + FP8 A/O。三部分的 QK/PV 均使用 FP8。评估 LIBERO Goal 成功率并收集 native 元素/bit 稀疏度，与同协议 eager raw 基线对照。

该实验只衡量本配置的闭环精度与 workload 特征，不推导实际硬件速度或能耗收益。

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| 分支 / 基础提交 | phaseI/vision-linear-full-integration / ee10143 |
| 环境 | 沿用 smolvla_eval、现有 GPU / MuJoCo / EGL |
| 前序实际 torch / transformers | 2.7.1+cu118 / 5.5.4 |
| 运行时记录 | commit.txt、worktree.patch、packages.txt、source_configs/ |
| checkpoint 固定 | prepare 使用 snapshot_download 解析完整 snapshot，两组及预检共用同一本地目录 |

不安装或升级依赖。已包含 bcba565 校准统计器生命周期修复；历史 BF16 eager/SDPA 差异不视为量化误差。新 runner 不依赖 main.py 的模式选择，明确设置 raw / scale_inspection / quant_forward。

## 3. 模型与数据

policy = lerobot/smolvla_libero（legacy public）。prepare 将 checkpoint 连同 processor/normalizer 固定到同一 snapshot 路径，保存 prepared.json 与 *_resolved.yaml；避免只给权重 revision 而 processor 仍加载 main。

| 部分 | Linear W | Linear A/O | QK/PV A/B/O | 方法 |
|---|---|---|---|---|
| Vision | INT8 | E4M3 | E4M3 | Linear pot_ao_outlier；MatMul pot_fp8_outlier |
| VLM | INT8 | E4M3 | E4M3 | 同上 |
| Expert | INT4 | E4M3 | E4M3 | 同上 |
| Connector / patch embedding / action projection等未包装算子 | raw | raw | — | 不在本次覆盖内 |

**PoT 的确切范围：Linear 的 A/O 与 MatMul 的 A/B/O scale 为 PoT；整数权重 scale 保持连续 calibrated scale，并不强制为 PoT。**因此不能把本实验称为全部 scale 均 PoT。

统一 outlier_ratio=0.01，weight_quant_granularity=per_tensor，Linear/MatMul scale 均 per_site；Vision与VLM只统一精度和汇总分组，不共享scale。

校准：HuggingFaceVLA/libero v3.0，8 episodes，batch_size=1，frame_stride=4，seed=42。新精度必须独立 fresh calibration，不沿用前序 FP8 weight scale。batch_size=1 降低显式 Vision attention 的显存需求。

## 4. 评测协议

| 项目 | 固定值 |
|---|---|
| Suite / tasks | libero_goal，显式 task_ids=0..9 |
| 每 task episodes | 10（每组共100） |
| seed | 1000 |
| eval batch / max_parallel_tasks | 1 / 1 |
| n_action_steps / num_steps | 10 / 10，沿用当前 Phase I 协议 |
| chunk_size | 沿用同一checkpoint，不改 |
| 相机 | 沿用 camera1/camera2 rename |
| 视频 / unit sparsity | 关闭 |

这是项目当前 na10 协议，不宣称严格复现论文 na1 协议。只和本实验同协议 baseline 比较，不能直接减去历史 object 成功率。

报告整体 SR、逐 task 成功数与 delta_pp = quant SR − baseline SR。100 episodes/单 seed 仍有采样不确定性，不据此声称普遍无损。

## 5. 实验变量与分组

| 组 | config | 执行模式 | 用途 |
|---|---|---|---|
| B0 | baseline.yaml | 全部wrapper raw | 同eager backend基线；不校准、不收集量化位统计 |
| M0 | quant.yaml | mixed quant_forward | 正式Goal100 + 稀疏统计 |
| M0-smoke | 从quant解析配置生成 | 同M0；task0×1 | 校准后工程预检，单独输出，不计入正式100ep |

两组均build相同296 Linear + 88 MatMul wrapper并校验精度路由。B0的manifest配置虽然包含INT/FP字段，但执行模式强制raw，不执行任何量化，也不读取scale。baseline执行方式必须用本实验run_arm.py，不能将baseline.yaml直接交给main.py（main.py量化开启时默认quant_forward）。

量化覆盖：Vision 72 Linear + 24 MatMul；VLM 112 + 32；Expert 112 + 32。总384 sites；1152 scale（888 Linear+264 MatMul）。所有module_id、精度和方法逐项检查；校准后逐模块确认_stat_manager已恢复。

校准保存scale_audit.json，记录数值和SHA256。smoke/正式quant再次读取并要求与校准审计完全一致。仅A/O和MatMul scales要求PoT，W scale检查有限且正。

Runtime预期3736条（Vision216、VLM320、Expert3200），按module_id/phase/flow_step/role精确验收；静态权重296条。Vision/VLM限定prefill、flow_step=-1；Expert限定denoise、flow_step=0..9。

Runtime使用native counters，FP8采用S|MMM v1，验证4 bits/element。权重使用现有INT sign-aware口径，INT8和INT4分组报告，不能与FP8 S|MMM误当成相同格式指标。主分组Vision+VLM、Expert；保留Vision/VLM明细，每组再分runtime Linear/MatMul与静态weight。所有比率先求和计数，再相除；静态weight不随episode重复计权，不与runtime合并。

## 6. 运行命令

```bash
conda activate smolvla_eval
git pull --ff-only
bash experiments/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/scripts/run_all.sh
```

执行顺序：保存版本 → 固定checkpoint → fresh calibration → M0-smoke与覆盖检查 → B0 Goal100 → M0 Goal100与覆盖检查 → 汇总。

本轮只创建实验，未下载checkpoint、安装依赖或执行测试/rollout。首次使用snapshot_download时本地需要能访问Hub或已有完整缓存。

重复执行run_all.sh会拒绝覆盖已有outputs/scales。某阶段失败时归档其阶段目录，检查并解决原因后，可使用原prepared配置和校准scale单独继续：

```bash
python experiments/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/scripts/run_arm.py smoke
python experiments/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/scripts/run_arm.py baseline
python experiments/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/scripts/run_arm.py quant
python experiments/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/scripts/summarize.py
```

不要跳过校准覆盖审计。smoke episode失败会记录为失败，但若计算、统计和结果产物完整，仍可继续正式评测；单次任务失败不等于工程故障。自动runner中任一运行异常或覆盖不全均停止。

## 7. 输出目录映射

根目录：outputs/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/

| 阶段 | 输出 | 状态 |
|---|---|---|
| prepare | prepared.json、baseline_resolved.yaml、quant_resolved.yaml | done（snapshot `31d453f7…`） |
| 校准 | calibrate/scale_audit.json、completed.json | done（PASS，384 sites / 1152 scales） |
| 预检 | smoke/result.json、coverage_summary.json、sparsity/ | done（task0×1ep SR=100%，coverage PASS） |
| B0 | baseline/result.json、execution.json、config.yaml | done（100ep SR=87%，`mode=raw`） |
| M0 | quant/result.json、execution.json、scale_audit.json、coverage_summary.json、sparsity/ | done（100ep SR=83%，`mode=quant_forward`，coverage PASS） |
| 汇总 | summary.json、task_success.csv | done（`delta_sr_pp=-4.0`） |

scale目录：scales/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/quant/。

请提交summary、逐task结果、manifest、runtime/weight/outlier CSV、scale_audit、prepared/config/version记录及run.log以便独立复核；无需提交模型或数据集。分享证据时可将本地snapshot路径替换为占位路径，保留snapshot revision。

## 8. 风险与注意事项

- 100 episodes × 2组，以及全模型fake quant和逐张量bit统计，耗时明显高于先前1ep；不承诺一小时内结束。
- Outlier sidepath保持浮点，不能把此配置解释为全部运算纯INT/FP8硬件实现。
- raw基线与量化组均用相同显式attention路径，以隔离backend差异；仍不证明eager与原SDPA闭环等价。
- Connector保持raw；“Vision+VLM整体”指已覆盖算子的统一精度/汇总，不代表每个算子都量化。
- 先完成本配置再决定更多消融，不把本次W格式改变归因于某一个组件。
