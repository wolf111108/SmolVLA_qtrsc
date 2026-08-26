# 2026-08-25 hfvla 四 suite 结果 + 五 checkpoint 对比定论

> **任务设备归属：h100**（zyzhao@hankh100，NVIDIA H100 NVL，主评测机）
> 所有评测/审计/训练任务默认在本机执行；若使用 gpupro6000e（lfwang@10.113.225.67）会在对应条目单独标注。

## 今日任务

- [x] `HuggingFaceVLA/smolvla_libero` 四 suite 评测跑完（08-24 16:19 启动，今晨结束）
- [x] 生成五模型对比（`outputs/hfvla_vs_baselines.md` 自动产出）
- [x] 量化 baseline 最终确认：A 类 `lerobot/smolvla_libero` (69.2%)
- [x] MuJoCo 版本定论：3.8.1 → 3.3.2，Spatial Task5 从 0/10 飙到 8/10（决定性变量）
- [x] Phase 5 combined 加做 3.3.2 版（新建 `run_phase5_combined_mj332.sh`）
- [x] 任务分发：Phase 6 → gpupro6000d，Phase 5 combined → gpupro6000a
- [x] h100 启动 strict Table-2 10-step smoke（issue3287 preflight，PID 3288379）
- [ ] 更新 `.gitignore` 后提交 git（logs/*.md 纳入版本控制）
- [ ] 自微调冒烟跑完后回填结果，再决定是否推 500 steps / 全量 100k

## 运行进程

### hfvla 四 suite benchmark（已结束）【设备：h100】

| 进程 | PID | 命令 | 输出位置 | 状态 |
|---|---|---|---|---|
| hfvla LIBERO benchmark | 2277367/2277371 | `nohup bash scripts/run_hfvla_libero_suites.sh > outputs/hfvla_libero_run.log 2>&1 &` | `outputs/hfvla_libero_libero_{spatial,object,goal,10}/` | ✅ 四 suite 全部完成 |

启动于 08-24 16:19，总耗时约 14h（32 层模型单步推理比 16 层慢，~1.6-1.9 it/s）。

## 结果

### hfvla（官方 current reference, 32 层/0.5 宽/0.6B）四 suite

| Suite | hfvla | 说明 |
|---|---:|---|
| Spatial | 65% (65/100) | Task0=60% 起步即偏低 |
| Object | 71% | 相对其较强的一项 |
| Goal | 72% | task0/1=100% 但 task3=0%，方差极大 |
| Long | **37%** | 全部模型中最低 |
| **平均** | **61.2%** | 五模型中排第 3 |

### 五模型总表（最终版）

| Suite | 论文 | **hfvla** | lerobot (A) | k1000dai (D) | tiantianx (C) |
|---|---:|---:|---:|---:|---:|
| Spatial | 90% | 65% | **81%** | 64% | 76% |
| Object | 96% | 71% | 60% | **82%** | 59% |
| Goal | 92% | 72% | **77%** | 70% | 63% |
| Long | 71% | 37% | **59%** | 46% | 40% |
| **平均** | **87.3%** | 61.2% | **69.2%** | 65.5% | 59.5% |

完整 per-task 矩阵：`outputs/hfvla_vs_baselines.md`

## 结论与下一步

### 已定论

1. **官方 32 层大架构反而排第 3**（61.2% < A 类 69.2%）——参数更多（0.6B vs 0.45B）不等于更好；
   Long suite 甚至全场最低（37%），仅 Object/Goal 相对有优势；
2. **Long 是所有公开 checkpoint 的共同短板**（最好 59% vs 论文 71%，gap 最大），提示论文在长程任务上有未公开处理；
3. **五模型无一接近论文 87.3%**（最佳 69.2%，差 18pp）；各 suite 最佳分属不同模型，无全能选手——
   "论文数字依赖未公开细节、无法通过公开 checkpoint 复现"至此有 4 模型 × 4 suite 的完整证据；
4. **量化 baseline 最终确认 A 类 `lerobot/smolvla_libero`（69.2%）**：绝对最高 + 逐 suite 领先。
   hfvla 的加入不改变反而强化该选择。

### 计划

1. [ ] git 提交：hfvla 结果 + logs/*.md + 新脚本（.gitignore 已修，logs/*.md 现在会入库）；
2. [ ] **自微调冒烟**（主线）【设备：h100 起步，全量可转 gpupro6000e】：
   `lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=HuggingFaceVLA/libero`
   500 steps；已知坑见 08-24 日志（type/path 互斥、accelerate、repo_id 已解决），下一个预期坑：
   dataset key 对齐（image/image2/8D state vs policy camera1/2/3/6D state）；
3. [ ] 冒烟通过后全量 100k（gpupro6000e 并行，注意该机 shell 是 tcsh + EGL device 需重新枚举，见 MIGRATION_GUIDE）；
4. [ ] 量化框架不被阻塞【设备：h100】：用 A 类 checkpoint 开发 QuantizedLinear + calibration。

## 备查

- hfvla 速度：~1.6-1.9 it/s（16 层模型 ~2-2.4 it/s），32 层推理更重；
- 失败 episode 跑满 280 步产生大量 `running_success_rate=0.0%` 日志行——看进度用
  `grep -a "Stepping through eval batches" <log> | tail -3`（每个 task 完成更新一行），不要被片段 0% 误导；
- 五 checkpoint 架构与成绩全景：`outputs/checkpoint_baselines_summary.md`（08-24 版，hfvla 数据今日补齐）。

---

## 08-25 晚间进展（MuJoCo 定论 + 任务分发 + strict 训练启动）

### 1. MuJoCo 版本是决定性变量

对照实验（A 类 `lerobot/smolvla_libero`, Spatial Task5, seed=1000, 10 eps）：

| 环境 | mujoco | Task5 SR |
|---|---|---:|
| smolvla_eval | 3.8.1 | 0/10 = 0% |
| smolvla_eval_mj332 | 3.3.2 | 8/10 = 80% |

结论：五模型 benchmark（全在 3.8.1）绝对数字可能被系统性低估，异常低任务需在 3.3.2 复测。

### 2. Phase 5 combined 加做 3.3.2 版

- 新建 `scripts/run_phase5_combined_mj332.sh`（输出 `05_rng_ordering/combined_mj332/`）
- 目标：凑齐 suite-ordering 四格对照（combined vs 逐 suite × 3.8.1 vs 3.3.2）

### 3. 任务分发到 gpupro6000 系列（两机并行）

| 设备 | 任务 | PID | 环境 | 状态 |
|---|---|---|---|---|
| gpupro6000d | Phase 6 逐 suite 串行（四 suite） | 167786 | smolvla_eval (3.3.2) | ✅ spatial rollout 中 |
| gpupro6000a | Phase 5 combined | 2565039 | smolvla_eval (3.3.2) + EGL device 0 | ✅ 已进入 rollout |

- gpupro6000 系列的 `smolvla_eval` 本身就是 mujoco 3.3.2，无需单独 mj332 环境
- gpupro6000a 需手动补 EGL 变量（`MUJOCO_GL=egl`、`MUJOCO_EGL_DEVICE_ID=0`）并固化进 conda
- 预计各 8-12h，完成后把 `eval_info.json` 汇总回 h100 做四格对照

### 4. h100 启动 strict Table-2 10-step smoke（issue3287 preflight）

- 输出：`outputs/smolvla_issue3287_preflight_run1`，日志 `outputs/smolvla_issue3287_preflight_run1.log`
- PID：**3288379**（CUDA_VISIBLE_DEVICES=0）
- 关键配置（strict init 路线：仅 pretrained VLM + 随机 expert）：
  ```
  --policy.type=smolvla --policy.load_vlm_weights=true
  --policy.expert_width_multiplier=0.75 --policy.num_vlm_layers=16
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true
  --policy.chunk_size=50 --policy.n_action_steps=10 --policy.num_steps=10
  --policy.optimizer_lr=1e-4
  --policy.scheduler_warmup_steps=100 --policy.scheduler_decay_steps=100000 --policy.scheduler_decay_lr=2.5e-6
  --dataset.repo_id=HuggingFaceVLA/libero --dataset.revision=v3.0
  --batch_size=64 --steps=10 --num_workers=6
  --num_processes=1 --num_machines=1 --mixed_precision=bf16
  ```
- ⚠️ 待验证坑：`accelerate launch --mixed_precision=bf16` 之前被证明不传进 lerobot（需 `--accelerator.mixed_precision=bf16`）。
  若 smoke 的 loss/dtype 异常，优先查这点。
