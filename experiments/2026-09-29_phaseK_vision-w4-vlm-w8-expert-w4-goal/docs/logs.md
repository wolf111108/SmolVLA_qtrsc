# 实验日志（Logs）

- **实验名称**：2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal
- **状态**：draft
- **最后更新**：2026-09-29

## 1. 当前状态

| 阶段 | 状态 | 备注 |
|---|---|---|
| prepare | done | snapshot 复用本地缓存 `31d453f7…`，写出 `prepared.json` / `quant_resolved.yaml` |
| calibrate | running | 独立 fresh scales，目标 1152 个 scale 文件；scale 目录仍为 0 |
| smoke | pending | Goal task0 ×1ep |
| quant | pending | Goal 10×10ep，SR only |
| summarize | pending | summary.json 与 task_success.csv |

## 2. 运行日志

| 日期 / 时间 | 阶段 | 状态 | 备注 |
|---|---|---|---|
| 2026-09-29 | 创建 | draft | 仅新增实验，等待本地运行；不重跑 baseline |
| 2026-09-29 19:04 | prepare | done | `outputs/…/prepared.json`、`commit.txt`（`56c64cb`）、`worktree.patch`、`packages.txt`、`source_configs/` 均已写出 |
| 2026-09-29 19:04 | calibrate | running | `python run_arm.py calibrate` 已启动，QK/PV+Linear 位点校验通过；19:40 时已写入 ~151 GB（见 §4-1），估算前向 356 步已完成 ~80–90%，scale 目录仍为 0 |

## 3. 后台任务

| PID | 阶段 | 启动时间 | 状态 | 命令 | 日志 |
|---|---|---|---|---|---|
| 832558 | prepare→summarize | 2026-09-29 19:04 | 🔄 running | `GPU=0 nohup bash -c 'conda activate smolvla_eval && exec bash experiments/2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal/scripts/run_all.sh'` | 内部 `outputs/…/run.log`（tee）；外层包装日志 `/tmp/phaseK_vw4vlmw8ew4_goal_outer.log` |

主机 `gpupro6000d`，GPU 0（RTX PRO 6000 Blackwell）空闲，`GPU=0` 与 `smolvla_eval` 的 `MUJOCO_EGL_DEVICE_ID=0` 一致。

## 4. 异常与处理

### 4-1 校准期 tensor dump 失控写入（已定位，非本轮引入）

| 项 | 内容 |
|---|---|
| 现象 | 校准阶段 GPU util 仅 ~4%，`/proc/<pid>/io` 的 `write_bytes` 以 ~70–100 MB/s 增长，19:40 已达 **151 GB**；`nvidia-smi` 显存正常，CPU 约 98% 空闲 |
| 定位 | 高频采样 `/proc/<pid>/fd` 抓到 fd 46 指向 `/home/lfwang/VLA_tcs2/outputs/tensor_dump/output_*_step0.pt`（`openat` 带 `O_WRONLY\|O_CREAT\|O_TRUNC`） |
| 根因 | `src/vla_tcs2/quant_linear.py::_maybe_dump_tensors()` 由全局 `_dump_step` 控制，本意只 dump 前 `VLA_TENSOR_DUMP_STEPS`（默认 3）步；但 **`src/vla_tcs2/calibration.py` 从未调用 `set_dump_step()`**，`_dump_step` 恒为 0 → 每个校准 step 都重写 296 个 `activation_*.pt` + 296 个 `output_*.pt`（实测 **443 MB/step**，文件名固定 `step0`，互相覆盖） |
| 影响 | 校准退化为 I/O bound（9-25 校准 1h54m 主要即此开销）；**磁盘占用有界**（`outputs/tensor_dump/` 稳定 ~1.2 GB，296 weight + 296 activation + 296 output，全部 `step0`），不会撞 300 GB NFS 配额；不进入 `quant_forward`，正式评测不受影响 |
| 处理 | **不处理**。`src/**` 属于 `run_arm.py` 的 source fingerprint 校验范围，运行中修改会在下一 stage 断言失败终止整轮；且 9-25 参照组走的是同一代码路径，保持一致性。仅在事后提 issue/记录 |

`/proc/<pid>/io` 的 `write_bytes` 与 `cancelled_write_bytes` 会同时被 `grep write_bytes` 命中，取值需用 `awk '/^write_bytes/{print $2}'`。

## 5. 修订记录

| 日期 | 修改内容 |
|---|---|
| 2026-09-29 | 新建 W4/W8/W4 AFP8 PoT Goal SR 实验 |
