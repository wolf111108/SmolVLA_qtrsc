# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-22_quickscan_smmm-bit-sparsity
- **状态**：running
- **负责人**：
- **创建日期**：2026-09-22
- **相关前序实验**：`experiments/2026-09-15_sparsity-ratio-quickscan/`（q3_fp8_po2 的 scale 与协议来源）

---

## 1. 实验目的

- 在新的 **S|MMM bit 统计口径**（sign+mantissa，无隐藏前导 1，见 `stat_manager._extract_sm_from_raw` 2026-09-22 修改）下重新测定 FP8 PoT 的 runtime/weight bit sparsity。
- 与 2026-09-15 quickscan 的旧口径（1.MMM，~42%）对照，量化口径变化的影响（预期**上升**约 +12.5pp：旧 hidden bit 对 normal 数恒为 1（0% sparse），新 sign 位在正负均衡时约 50% 为 0，首位 zero-rate 差 50pp/4bit = +12.5pp，即约 42% → ~54.5%；实际还受 ±0/subnormal/native 修正影响）。
- 仅 workload characterization（task0×1ep），不做 accuracy claim。

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval |
| 仓库 revision / commit | 分支 phaseI/vision-linear-full-integration（含 SMMM 口径修改） |
| LeRobot 路径与 commit | `lerobot_current/` @ |
| GPU | |
| 关键依赖版本 | torch / mujoco 等（如有非默认版本） |

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | （路径 + checkpoint 分类：legacy_public / official / paper_like / 自训） |
| 评测基准 | （LIBERO spatial / goal / object / 10，或 Meta-World 任务集） |
| 校准数据 | （episodes 数、来源，若复用注明 scale 文件路径） |

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | （如 10 tasks × 10 eps，seed=1000） |
| 采样参数 | （n_action_steps / chunk_size / num_steps 等） |
| 指标 | （success rate 等） |

## 5. 实验变量与分组

| 组 | config | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| S4 | s4_fp8_smmm.yaml | FP8 PoT（A/W/O e4m3，outlier_ratio=0.01），**SMMM 口径** | VLM+Expert 288 sites reuse q3 canonical scale；task0×1ep；seed=1000 | 与 09-15 quickscan q3 完全同配置，仅统计口径不同 |

## 6. 运行命令

```bash
bash experiments/2026-09-22_quickscan_smmm-bit-sparsity/scripts/run_smmm_quickscan.sh
```
（Gate 0 pytest → Gate 1 scale 复制+sha256 校验 → Gate 2 eval-only task0×1ep → Gate 3 汇总+行数 gate）

## 7. 输出目录映射

<!-- config → outputs/2026-09-22_quickscan_smmm-bit-sparsity/ 下的实际产出目录（与实验同名，见 README §5），跑完后逐一登记，便于回溯原始数据 -->

| config | 输出目录（outputs/ 下） | 状态 |
|---|---|---|
| s4_fp8_smmm.yaml | outputs/2026-09-22_quickscan_smmm-bit-sparsity/s4_fp8_smmm/ | done |

## 8. 风险与注意事项

- **新旧口径数字不可直接混用**：旧 CSV/图（~42%）是 1.MMM 口径；本实验产出 S|MMM 口径。比较时必须注明。
- 符号位语义：负数符号位=1（非零 bit）；±0 分别为 0000/1000；normal 数无隐藏位（1.0 → 0000）。
- scale 三重防护：复制到独立 scale_dir + sha256 逐文件校验 + runner 强制 `--skip-calibration`。
- 1 episode 采样：绝对值有噪声，重点是口径对照而非绝对精度。
