# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**（区别于 `results.md` 的结果记录）。
> 每次启动/完成一次运行、发现或修复异常时追加一条；章节为固定结构，不可删除；无内容写「无」。
> 运行日志按时间**正序**追加，最新状态反映在「当前状态」与头部字段。

- **实验名称**：2026-09-15_phaseI_vision-quantization
- **状态**：running（V0 ✅ / V1 ✅ / VLIN ✅ / **V2 ✅（84.0%）** / **V3 ✅（85.0%）** / V4–V5 待跑；§5.1 归因已完成）
- **最后更新**：2026-09-22

---

## 1. 当前状态

Gate L0（legacy regression）、V0（Vision workload audit）、V1 与 VLIN 已完成。VLIN 正式 Goal×100 = **80.0%（80/100）**，严格直接对照 G6-A = 90.0%，Δ = **−10.0pp**。V1 Goal×100 = 81.0%，相对 H3 S0 89.0% 为 −8.0pp，但因 V1 对 legacy VLM/Expert scales 也重新 calibration，现将其记为**参考差值而非严格 connector-only 单变量结论**。下一阶段按 `experiment_setup.md §5.1` 拆分 Vision MLP-only 与 attention-projection-only。

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| Gate L0 legacy regression | ✅ done | 2026-09-15 | 224/64/0/0，向后兼容 |
| V0 workload audit | ✅ done | 2026-09-15 | 12 层 / 768 / 2 camera / 428 GFLOPs |
| V1-R raw equivalence | ✅ done | 2026-09-19 | max/mean/max_rel error 全 `0.000e+00`（bit-exact） |
| V1 Connector FP8 | ✅ done | 2026-09-19 | routing 289、calibration 生成、smoke 100%、Goal×100 = 81.0%；−8.0pp vs H3 仅作参考，**非严格 connector-only 单变量** |
| V1 方法学审计 | ✅ documented | 2026-09-21 | raw wrapper bit-exact 保留；确认 legacy scales 独立 recalibrate，因此 8pp 降级为参考证据 |
| VLIN Vision 72-Linear FP8（Gate 1–6） | ✅ done | 2026-09-21 | 分支 `phaseI/vision-linear-full-integration`；296/64 routing、reuse288/recal72、72/72 + 216/216 scale、smoke 100% |
| VLIN Goal×100 | ✅ done | 2026-09-21 | **SR = 80.0%（80/100）**，Δ −10.0pp vs G6-A；第 1 次尝试 12:50 被 SIGTERM（task2 中途，非 OOM 非人为），重启后完整跑完 |
| V2 Vision MLP FP8（Gate 1–6） | ✅ done | 2026-09-21 | 248/64 routing、reuse288/recal24、24/24 + 72 scale 文件、smoke 100%（runner `run_vision_linear_variant.sh`） |
| V2 Goal×100 | ✅ done | 2026-09-21 | **SR = 84.0%（84/100，L_MLP = 6pp）**，逐 task [10,10,10,6,9,10,4,9,10,6]，eval_s = 8013 |
| V3 AttnProj FP8（全流程） | ✅ done | 2026-09-22 | Gate1–6 全过（272/64、48/48、144 文件、smoke 100%）；**Goal×100 = 85.0%（85/100，L_Attn = 5pp）**，逐 task [10,9,10,8,9,9,4,9,10,7]，eval_s = 18241 |
| V2 Vision MLP-only | ✅ done | 2026-09-21 | Goal×100 = **84.0%**，L_MLP=6pp |
| V3 Vision AttnProj-only | ✅ done | 2026-09-22 | Goal×100 = **85.0%**，L_Attn=5pp；interaction with V2 = −1pp ≈ 0 |
| Vision sparsity-compute | ✅ done | 2026-09-22 | VSC-0 sparsity valid / compute invalid；VSC-1 修复后 **594.41 G / 86.5% coverage**；Vision/VLM/Expert elem sparsity 4.08/1.92/3.02% |
| V4/V5 | ⏳ pending | — | sparsity/compute 完成后再决定；V4 前置：sdpa → eager 等价性 |

## 2. 运行日志

| 日期 | 阶段 | 命令 / 配置 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-09-15 | Gate L0 | `gate_l0_legacy_regression.py --config g6a_all_fp8_control.yaml` | ✅ | — | 224 Linear/64 MatMul/0 vision/0 connector |
| 2026-09-15 | V0 | `audit_vision_structure.py --config v0_workload_audit.yaml` | ✅ | `outputs/.../v0_workload_audit` | 1ep，SR=100%，cameras/sample=2.0 |
| 2026-09-15 | V0 绘图 | `scripts/plot_phaseI_flops_pies.py` | ✅ | `docs/figures/phaseI_flops_pie_*.png` | FLOPs 占比饼图（V0 实测 + 手册 §0 整体） |
| 2026-09-19 | V1-R | `v1_connector_raw_equiv.py --config v1_connector_fp8.yaml` | ✅ | — | connector output bit-exact；289 quant modules |
| 2026-09-19 | V1 calibration | `main.py --config v1_connector_fp8.yaml --skip-evaluation` | ✅ | `scales/.../v1_connector_fp8` | connector scale 首次生成 |
| 2026-09-19 | V1 smoke | `main.py --config v1_connector_fp8.yaml --skip-calibration` | ✅ | `outputs/.../v1_connector_fp8` | task0 × 1ep，SR=100% |
| 2026-09-19 | V1 Goal×100 | `main.py --config v1_connector_fp8_goal.yaml --skip-calibration` | ✅ | `outputs/.../v1_connector_fp8_goal` | **SR=81.0%（81/100）**，eval_s=40735；CI `[72.2, 87.5]` |
| 2026-09-21 | VLIN 分支切换 | `git fetch mirror && git checkout -b phaseI/vision-linear-full-integration mirror/phaseI/vision-linear-full-integration` | ✅ | — | 分支**仅存在于 mirror remote**（origin 无）；基于 main@2b2a3013，+9 commits / 8 files（2 config + 2 audit 脚本 + 2 runner + 1 doc + 测试扩充） |
| 2026-09-21 | VLIN Gate 1–6 | `bash experiments/.../scripts/run_full_vision_linear.sh`（BASE_SCALE_DIR 默认 G6-A canonical） | ✅ | `outputs/.../vlin_full_run.log` | Gate1 pytest ✅ / Gate2 scale 复制到独立目录 ✅ / Gate3 routing **296 Linear + 64 MatMul**，reuse288+recal72，Vision op 各 12/12 / Gate4 calibration 45/45 forwards / Gate5 **72/72 sites、216/216 scale 文件**（总 1080 = 864 + 216 新增）/ Gate6 smoke task0×1 **SR=100%**（eval_s≈94.2）；总耗时 ≈25 min |
| 2026-09-21 | VLIN 结果回填 | 编辑 `docs/results.md` + `docs/logs.md` | ✅ | — | 新增 §3.6 / 总结果表 3 行 / 结论第 7 条；Goal×100 待审计后启动 |
| 2026-09-21 | VLIN Goal×100（第 1 次） | `run_full_vision_linear_goal.sh`（nohup） | ❌ terminated | `vlin_goal100.attempt1_terminated.log` | 12:50 在 task2 rollout 80/300 步处被外部 SIGTERM 杀掉（非 OOM：内存 20/251G、dmesg 无 kill 记录；非人为）；已完成 task0/1 未落盘（result.json 仅在全部结束时写，无断点续跑） |
| 2026-09-21 | VLIN Goal×100（第 2 次） | 同上，旧日志归档后重启 | ✅ | `outputs/.../vlin_full_vision_linear_fp8_goal` | preflight coverage PASS、复用 calibration；**SR = 80.0%（80/100）**，逐 task [10,10,8,7,9,9,4,9,9,5]；eval_s = 10189（≈2.83h）；result.json 已落盘 |
| 2026-09-21 | V2 全流程 | `RUN_GOAL=1 run_vision_linear_variant.sh v2_vision_mlp_fp8.yaml` | ⚠️ 部分 | `v2_run.log` | Gate1–6 全过（248/64、24/24、72 文件、smoke 100%）；Gate7 首次触发 goal config YAML 缩进错误（`n_episodes` 多缩 4 格），已修复（commit `2551e68`） |
| 2026-09-21 | V2 Goal×100 | `main.py --config v2_vision_mlp_fp8_goal.yaml --skip-calibration` | ✅ | `outputs/.../v2_vision_mlp_fp8_goal` | 17:22–19:39（≈2.2h）；**SR = 84.0%（84/100）**，逐 task [10,10,10,6,9,10,4,9,10,6]；eval_s = 8013 |
| 2026-09-21 | V3 全流程启动 | `RUN_GOAL=1 run_vision_linear_variant.sh v3_vision_attn_proj_fp8.yaml` | ✅ | `v3_run.log` | PID 864521；Gate1–6 全过；Goal×100 20:0x–00:5x（≈5.1h）；**SR = 85.0%（85/100）**，逐 task [10,9,10,8,9,9,4,9,10，7]；eval_s = 18241；result.json 已落盘 |
| 2026-09-21 | 结果审计修订 | 对照 G6-A / V1 / VLIN 配置与文档口径 | ✅ | — | VLIN 严格 baseline 固定为 G6-A 90%；逐 task Δ=[0,0,-1,-1,-1,0,-3,0,-1,-3]；V1 8pp 降级为非严格参考；删除 fake-quant runtime speedup 推断；V2/V3 设计合并进 experiment_setup.md §5.1 |

| 2026-09-22 | vision-sparsity-compute task setup | 创建标准 task config / runner / summarizer / 3 个固定 docs | ✅ | `outputs/.../tasks/vision-sparsity-compute/` | 复用 VLIN scales；强制 `--skip-calibration`；统计 Vision/VLM/Expert native sparsity + compute |

| 2026-09-22 | VSC-0 compute audit | 复核 `export_workload_csv` MatMul shape accounting | ⚠ fix merged | task `vision-sparsity-compute` | VSC-0 native sparsity 保留有效；688.50 G / 88.3% coverage 作废；已新增 exact physical-MAC accounting + T12 + VSC-1 rerun |

| 2026-09-22 | VSC-1 workload-fix | `vsc1_vlin_fp8_task0_1ep_workloadfix.yaml` | ✅ | `outputs/.../tasks/vision-sparsity-compute/vsc1_vlin_fp8_task0_1ep_workloadfix` | pytest 12 passed；summarizer GATE PASS；**594.41 G / 86.5%**；VLM 57.60 G / Expert 108.59 G |

## 3. 后台任务

| PID | 阶段 | 启动时间 | 状态 | 备注 |
|---|---|---|---|---|
| — | V1 Goal×100 | 2026-09-19 | ✅ finished | `v1_goal_run.log`；已与 Phase H H3、G6 串行避开 GPU 争用 |
| 481815 | VLIN Gate1–6 | 2026-09-21 | ✅ finished | nohup 日志 `outputs/2026-09-15_phaseI_vision-quantization/vlin_full_run.log`；启动时 export `MUJOCO_GL=egl / MUJOCO_EGL_DEVICE_ID=2` |
| 521631 | VLIN Goal×100 #1 | 2026-09-21 | ❌ killed | 12:50 被外部 SIGTERM（原因未明，非 OOM 非人为）；日志已归档 `vlin_goal100.attempt1_terminated.log` |
| 590130 | VLIN Goal×100 #2 | 2026-09-21 | ✅ finished | 13:30 重启，≈2.83h 完成；**SR = 80.0%**；`vlin_goal100.log` + `result.json` |

## 4. 异常与处理

- V0 FLOPs 首版把 `vision_forward_calls=38`（整个 rollout 总数）误当 camera 数，导致 FLOPs 被 ×38。修正为 hook `_get_action_chunk` 计数 sample_actions 次数，得 `cameras_per_sample_actions=2.0`（19 sample_actions × 2 camera = 38 calls）。FLOPs 按「per sample_actions」语义重算。
- 关键发现：vision attention 实现为 `sdpa`（非 eager），V4 前须先做 eager-equivalence gate（手册 §12.2）。
- **V1 方法学口径已修正**：connector raw wrapper bit-exact 只证明 wrapper 本身无数值偏差；V1 对 legacy VLM/Expert scales 也重新 calibration，因此 H3 S0 89% → V1 81% 的 −8pp 不再表述为严格 connector-only 净损失。逐 task 相似性继续保留为 visual-prefix 敏感性的待验证线索。
- **VLIN 分支不在 origin**：`phaseI/vision-linear-full-integration` 只 push 到了 mirror remote（`wolf111108/SmolVLA_qtrsc`），`git fetch origin` 拉不到；需 `git fetch mirror` 后从 `mirror/phaseI/vision-linear-full-integration` 建本地分支。
- VLIN 运行无异常：Gate 1–6 一次通过，无手动干预；HF Hub 未设 token 的 rate-limit warning 不影响结果。
- **VLIN Goal×100 第 1 次尝试被外部 SIGTERM 杀掉**（12:50，task2 rollout 中，非 OOM 非人为非脚本失败）。runner 单进程跑 10 task、`result.json` 仅在全部结束时落盘 ⇒ 无断点续跑，重启需从头。若再发生，需排查系统级会话/清理机制。注意 `Built vec env` 10 条是启动时一次性预建，**不是进度标记**（判断进度看 `videos/libero_goal_<n>/` 目录 mtime 或 rollout 进度条）。
- VLIN Goal×100 #2 与同用户两个 Qwen eval server 共享 GPU 0（19.5G/95G）未出问题。
- **V2 Gate7 首次失败原因：goal config 生成脚本把 `n_episodes: 10` 写成 6 格缩进（应在 evaluation 下 2 格）**，YAML 解析报 `mapping values are not allowed here`。修复后 V2/V3 goal config 均通过 `yaml.safe_load` 验证。教训：从模板派生 config 后必须过 YAML lint。
- **性能口径**：VLIN `eval_s=10189`、G6-A `eval_s=7430` 不用于推断 FP8 加速。当前为 fake/simulated quantization，且 SR 变化会改变 rollout 步数；真实性能需后续在低精度 kernel / 硬件路径单独评测。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-15 | 创建文档 | |
| 2026-09-15 | 回填 Gate L0 + V0 进度 | |
| 2026-09-15 | 新增 V0 FLOPs 饼图记录 | |
| 2026-09-19 | 回填 V1 全流程（V1-R 等价性 bit-exact + smoke 100% + **Goal×100 = 81.0%**）；状态改为 `running（V0 ✅ / V1 ✅ / V2–V5 待跑）`；记录 8pp 损失与「先审计再进 V2」 | |
| 2026-09-21 | 回填 VLIN Gate1–6（分支切换 + 一键脚本全过：296/64、72/72、216/216、smoke 100%）；状态更新为 `running（V0 ✅ / V1 ✅ / VLIN Gate1–6 ✅ / V2–V5 待跑）`；记录「VLIN 分支仅存于 mirror remote」 | |
| 2026-09-21 | 结果审计修订：统一 VLIN baseline=G6-A 90%；V1 改为非严格参考；补 G6-A→VLIN 逐 task Δ；删除 fake-quant speedup 推断；下一阶段 V2/V3 设计统一归入 `experiment_setup.md §5.1`，未新增规范外实验文件 | |
| 2026-09-22 | V2/V3 已完成并归因闭合；新增 Vision sparsity-compute 标准 task（仅 configs/scripts/固定 docs），用于加入 Vision 后的 native element/bit sparsity 与 MAC/FLOP 统计 | |
