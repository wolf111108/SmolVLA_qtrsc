# 2026-08-22 tiantianx LIBERO benchmark 启动 + A 类 baseline 汇总

## 今日任务

- [x] 汇总 `lerobot/smolvla_libero`（A 类 legacy）四 suite 结果并与论文对比
- [x] 确认 tiantianx normalizer 审计结论（8D state，与 HuggingFaceVLA/libero stats 一致）
- [x] 新增 `run_tiantianx_libero_suites.sh` + `compare_tiantianx_vs_lerobot.py`
- [x] 后台运行 tiantianx 四 suite benchmark（2026-08-22 启动，次日跑完）
- [x] 结果分析（见下）

## 运行进程

### tiantianx 四 suite benchmark（已结束）

| 进程 | PID | 命令 | 输出位置 | 状态 |
|---|---|---|---|---|
| tiantianx LIBERO benchmark | 951793 | `nohup bash scripts/run_tiantianx_libero_suites.sh > outputs/tiantianx_run.log 2>&1 &` | `outputs/tiantianx_smolvla_libero_<suite>/` | ✅ 四 suite 全部完成 |

启动命令（在 `smolvla_eval` 环境执行）：

```bash
conda activate smolvla_eval
cd ~/VLA_tcs2
nohup bash scripts/run_tiantianx_libero_suites.sh > outputs/tiantianx_run.log 2>&1 &
```

进程环境验证通过（`/proc/951798/environ`）：

```text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
MUJOCO_EGL_DEVICE_ID=2
CONDA_DEFAULT_ENV=smolvla_eval
```

查看进度：

```bash
tail -f outputs/tiantianx_run.log                        # 总日志
ls outputs/tiantianx_smolvla_libero_*/eval_info.json     # 已完成的 suite
```

预期耗时：Spatial/Object 各 ~1.5-2h，Goal ~2h，Long ~3-4h，共约 8-10h。
脚本幂等：中断后重跑会跳过已有 `eval_info.json` 的 suite。

## 结果

### A 类 baseline 汇总（lerobot/smolvla_libero）

`scripts/summarize_lerobot_vs_paper.py` 产出，报告：`outputs/lerobot_smolvla_libero_vs_paper.md`

| Suite | 论文 | A 类本地 | 差值 |
|---|---:|---:|---:|
| Spatial | 90% | 81% (81/100) | -9pp |
| Object | 96% | 60% (60/100) | -36pp |
| Goal | 92% | 77% (77/100) | -15pp |
| Long | 71% | 59% (59/100) | -12pp |
| **平均** | **87.3%** | **69.2%** | **-18.0pp** |

Spatial task5 = 0/10（A 类已知异常点）；Object 差距最大且均匀偏低。

### tiantianx 审计结论（来自 outputs/ 已有文件）

- `outputs/audit_tiantianx_normalizer.txt`：normalizer `observation.state` 实际为 **8D**，config 中 6D 是 stale metadata；
- `outputs/compare_libero_stats.txt`：tiantianx stats 与 `HuggingFaceVLA/libero` 数据集 mean/std/min/max **完全一致**（|diff| ≤ 4.4e-08）→ 该 checkpoint 确实训练自官方 LIBERO 数据，作为 paper-like baseline 可信度高。

### tiantianx (C 类) 四 suite 最终结果

`scripts/run_tiantianx_libero_suites.sh` 自动跑完后由 `compare_tiantianx_vs_lerobot.py` 汇总，
报告：`outputs/tiantianx_vs_lerobot_libero.md`

| Suite | 论文 | tiantianx (C) | lerobot (A) | C vs A | C vs 论文 |
|---|---:|---:|---:|---:|---:|
| Spatial | 90% | 76% (76/100) | 81% (81/100) | -5pp | -14pp |
| Object | 96% | 59% (59/100) | 60% (60/100) | -1pp | -37pp |
| Goal | 92% | 63% (63/100) | 77% (77/100) | -14pp | -29pp |
| Long | 71% | 40% (40/100) | 59% (59/100) | -19pp | -31pp |
| **平均** | **87.3%** | **59.5%** | **69.2%** | **-9.8pp** | **-27.8pp** |

## 结论与下一步

### 结论

1. **A 类 checkpoint 四 suite 固化**：69.2% vs 论文 87.3%（-18pp）。
2. **C 类 tiantianx 反而整体更弱**：平均 59.5%，比 A 类还低 9.8pp，距论文 -27.8pp。
   尽管 recipe 更贴论文（100k steps / expert-only / frozen VLM / 官方数据 stats），
   在当前 eval 协议下并未兑现论文性能。
3. **两个 checkpoint 的失败模式互补而非同构**：
   - Spatial task5：A=0/10 → C=3/10（部分恢复，但仍远低于 B 类 9/10）；
   - Spatial task8/9、Long task4/7/8：C 明显更差（task8/9 共 -90pp，Long task4=0% vs 60%）；
   - Long suite C 类首次出现 task4/task8 = 0% 的完全失败点。
4. **没有任何公开 checkpoint 复现论文 Table 2**：A=69.2%、C=59.5%，都距论文很远。
   论文未公开原始 checkpoint 与精确 eval 细节，差距无法进一步归因。
5. Object 是两个 checkpoint 共同的短板（A=60%, C=59%, 论文 96%），
   疑似存在系统性因素（输入约定 / eval 细节 / 数据版本），非单 task 崩溃。

### 下一步

1. ~~等 tiantianx 四 suite 跑完~~（已完成）；
2. 可选：人工抽查 C 类失败视频（尤其 Long task4/8=0%、Spatial task8/9），
   判断是否 grasp/wait 行为类失败，与 A 类 task5 的推倒 bowl 模式对比；
3. **固化 quantization baseline 的选择**：建议以 **A 类（69.2%，四 suite 完整、无 0% task）**
   作为 PTQ 的 FP/BF16 参照，C 类可作为第二参照（注意其 0% task 会放大相对退化）；
4. 恢复量化框架开发（ModelWrapper / QuantizedLinear / StatManager，见 README §15）；
5. 若还要追论文数字，剩余路径只有自行 fine-tune（成本高）或联系论文作者，优先级低。

## 备查

- 本日新增脚本：
  - `scripts/run_tiantianx_libero_suites.sh`（幂等，跑 4 suite，结束自动对比）
  - `scripts/compare_tiantianx_vs_lerobot.py`（C vs A vs 论文三向对比，写 `outputs/tiantianx_vs_lerobot_libero.md`）
- conda activate 时出现 `WARNING: overwriting environment variables ... MUJOCO_GL` 属预期（env 内配置覆盖系统级），实际进程环境已验证正确；
- 判断后台进程 EGL 配置不要用当前 shell 的 printenv（可能在 base），要用 `/proc/<PID>/environ`。

## 追记（2026-08-23 补充）

### 后续发现：新的候选 checkpoint（HF Hub 搜索）

8-22 结果显示 A/C 两类公开 checkpoint 都无法接近论文 87.3%，遂于 8-23 用 HF Hub API
全面搜索 `smolvla`+`libero`，核查各候选的 config 架构与 train_config recipe，发现两个
高价值新候选（README 第 8 节之外的）：

1. **`HuggingFaceVLA/smolvla_libero_ckpts@100000/pretrained_model`**（官方训练轨迹）
   - 文件树含 20k/40k/60k/80k/100k 五档完整 checkpoint（model + optimizer state）；
   - 100k 档 config：16 层 / 0.75 宽 / `n_action_steps=1` / image+wrist_image / 8D state；
   - train_config：dataset = `physical-intelligence/libero`，100k steps，batch64；
   - repo 已标记 DEPRECIATED（官方改推 32/0.5 的 B 类），但**最可能是论文 Table 2 对应的
     官方训练本身**——之前 README 只记录过社区转载版 `jadechoghari/smolvla-libero-ckpts`
     （parser 报错不可用），官方 org 自己这份此前未被发现。
2. **`k1000dai/smolvla_libero_finetune`**（社区，下载量 134）
   - 16 层 / 0.75 宽，image+wrist_image / 8D state，from `lerobot/smolvla_base`；
   - train_config：**steps=100000, batch_size=64** —— 精确匹配论文 recipe
     （连 global batch 64 都对上，tiantianx 只有 batch32 需假设 2 GPU 才能解释）。

次级候选（暂不跑）：`godnpeter/smolvla_libero_scratch_bs64_us300k_0913`、
`k1000dai/smolvla_libero_spatial_scratch`、`mimimimi2002/smolvla_libero_object_100k`、
`AustineJohnBreaker/smolvla_stratch_libero_{spatial,10,goal}`（分 suite 训练版）。
非 SmolVLA 参照（不进量化主线）：`moojink/openvla-7b-oft-*`、`lerobot/pi05_libero_base`。

### 为新候选搭建的流水线（8-23）

- `scripts/preflight_policy.py`：解析 `hub:` / `hfsub:<repo>:<subfolder>` 两种 spec，
  自动下载子目录并从 config 推导 rename_map（两候选相机 key 为 `image`/`wrist_image`，
  rename 自动生成，无需手写）；
- `scripts/run_new_candidates_libero.sh`：per 模型 preflight → smoke(task0×1，失败自动跳过
  不阻塞另一模型) → 四 suite ×100 episodes（幂等）→ 汇总；
- `scripts/compare_new_candidates_vs_baselines.py`：四向对比（新候选×2 vs C vs A vs 论文），
  写 `outputs/new_candidates_vs_baselines.md`，并自动建 `logs/2026-08-23_new_candidates_libero_benchmark.md`。

### 新 benchmark 启动记录（8-23）

```bash
nohup bash scripts/run_new_candidates_libero.sh > outputs/new_candidates_run.log 2>&1 &
# PID 1277524
```

启动验证：preflight 正常拉取 hfvla 100k 档（cache 已达 1.5GB），预计两模型共 ~20-26h。
结果与选型结论见 `logs/2026-08-23_new_candidates_libero_benchmark.md`（跑完后回填）。
