# 实验日志（Logs）

- **实验名称**：2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal
- **状态**：done（prepare → summarize 全阶段跑通，无中断、无重跑）
- **最后更新**：2026-09-25

## 1. 当前状态

`run_all.sh` 六个阶段全部执行完毕并写出 `summary.json` / `task_success.csv`（stdout 末行 `PASS: summary.json and task_success.csv written`）。运行时 commit = `4cdccc4`（`commit.txt` 记录一致，`worktree.patch` 为 0 字节，说明工作区干净，结果可直接对应该 commit）。

## 2. 运行日志

| 日期 / 时间 | 阶段 | 状态 | 备注 |
|---|---|---|---|
| 2026-09-25 14:35 | 版本记录 + 实验创建 | done | `git rev-parse HEAD` → `commit.txt`（4cdccc4）；`git diff` → `worktree.patch`（0 字节）；`pip freeze` → `packages.txt`（152 行）；`configs/` → `source_configs/` |
| 2026-09-25 14:35 | prepare | done | `snapshot_download(lerobot/smolvla_libero)` → snapshot `31d453f7edd78c839a8bbc39744a292686daf0de`；写入 `prepared.json`、`baseline_resolved.yaml`、`quant_resolved.yaml`（两组的 `model.path` 均指向该 snapshot，`revision=null`） |
| 2026-09-25 ≈14:35–16:29 | calibrate | done | `HuggingFaceVLA/libero v3.0`，8 episodes（全局 1693 中抽样 51/228/285/457/501/563/1309/1518）→ 356 帧（`frame_stride=4`、`batch_size=1`、双相机）；`reuse=0`、`recalibrate=384`；累计 632256 个样本；写出 1152 个 scale 文件（`Saved …` 1152 条）与 `calibrate/scale_audit.json`、`completed.json`（`PASS, sites=384, scales=1152`） |
| 2026-09-25 16:29–16:31 | smoke（M0-smoke） | done | task0×1ep，SR=100%，eval 116.2 s；`smoke/coverage_summary.json` = PASS（384 / 1152 / 3736 / 296） |
| 2026-09-25 16:31–17:50 | baseline（B0） | done | `mode=raw`，libero_goal task 0–9 × 10ep = 100 ep，SR=**87%**，eval 4756.8 s |
| 2026-09-25 17:51–20:53 | quant（M0） | done | `mode=quant_forward` 且 `sparsity.enabled=true`（`chunk_size=4194304`，unit sparsity 关闭）；scale 审计与校准逐项一致；100 ep，SR=**83%**，eval 10910.7 s；导出 5 个 sparsity CSV + `coverage_summary.json` = PASS |
| 2026-09-25 20:53 | summarize | done | `summary.json`（`delta_sr_pp=-4.0`）、`task_success.csv` |

墙钟总计约 6 h 18 min（14:35 → 20:53）：校准约 1 h 54 min、B0 约 1 h 20 min、M0 约 3 h 02 min。时间边界取自各阶段产出文件 mtime 与 `eval_s`（log 中未单独打印阶段耗时）。

## 3. 后台任务

无。`run_all.sh` 前台执行，stdout/stderr 经 `tee` 写入 `outputs/2026-09-25_phaseK_vlm-vision-w8-expert-w4-goal/run.log`（7.0 MB，绝大部分是 rollout tqdm 刷屏，过滤后的行已随回填提交为 `docs/run_log_filtered.txt`）。

## 4. 异常与处理

| 现象 | 影响 | 处理 |
|---|---|---|
| `torchcodec` 载入失败：FFmpeg 8/7/6/5 分别缺 `libavutil.so.60/59/58/57`，最终 `libtorchcodec_core4.so: undefined symbol: torch_dtype_float4_e2m1fn_x2` → `Could not load libtorchcodec` | 仅影响视频渲染；本实验 `max_episodes_rendered=0`，视频关闭 | 记录并忽略；未安装或升级任何依赖 |
| HF Hub 未认证警告（`You are sending unauthenticated requests to the HF Hub`） | 无 | 忽略（snapshot 已固定到本地缓存目录） |
| `torch_dtype is deprecated! Use dtype instead!` | 无 | 忽略 |
| `lerobot_eval.py:372` `DeprecationWarning: … 'np.bool' scalars …` | 无（成功标志转 tensor 的旧写法） | 忽略 |
| `robosuite WARNING: No private macro file found`（3 条） | 无 | 忽略 |
| `worktree.patch` 为 0 字节 | — | 说明运行期间工作区无改动，结果可对应 commit `4cdccc4`，无需归档补丁 |

无中断、无阶段归档、无失败重跑。smoke 预检 episode 一次成功，正式 B0/M0 均完整覆盖 10 tasks × 10 episodes。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-25 | 创建日志 | |
| 2026-09-25 | 回填 prepare→summarize 全阶段运行记录、阶段墙钟耗时与环境异常处理；附过滤版 run log | |
