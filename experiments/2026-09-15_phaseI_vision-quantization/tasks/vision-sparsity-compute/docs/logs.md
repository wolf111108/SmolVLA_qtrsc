# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**。运行日志按时间正序追加。

- **实验名称**：2026-09-15_phaseI_vision-quantization / task: vision-sparsity-compute
- **状态**：done
- **最后更新**：2026-09-22

---

## 1. 当前状态

VSC-0 已完成，Gate 全 PASS。

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| config / runner / summarizer | ✅ ready | 2026-09-22 | 复用 VLIN scales；1ep workload characterization |
| VSC-0 run | ✅ done | 2026-09-22 | Gate 全 PASS；三份汇总 CSV 已生成 |

## 2. 运行日志

| 日期 | 阶段 | 命令 / 配置 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-09-22 | experiment setup | 创建 task 标准结构 | ✅ | — | 未创建规范外设计文件 |
| 2026-09-22 | VSC-0 | `bash .../tasks/vision-sparsity-compute/scripts/run_vision_sparsity_compute.sh`（nohup，PID 2081010） | ✅ | `outputs/.../tasks/vision-sparsity-compute/vsc_vlin_fp8_task0_1ep` | preflight coverage PASS（复用 VLIN scales，`calibration_policy: reuse` + `--skip-calibration`）；rollout task0×1ep；summarizer Gate 全 PASS（manifest 360 / Vision 72 / VLM・Expert 112+32 / weight rows 296 / flow_step 0–9 / Vision compute 347.89 G）；产出 `sparsity_compute_summary.csv` / `sparsity_by_operator.csv` / `compute_summary.csv`；运行日志 `vsc_run.log` |

## 3. 后台任务

| PID | 阶段 | 启动时间 | 状态 | 备注 |
|---|---|---|---|---|
| 2081010 | VSC-0 | 2026-09-22 | ✅ finished | nohup 日志 `.../tasks/vision-sparsity-compute/vsc_run.log`；启动时 export `MUJOCO_GL=egl / MUJOCO_EGL_DEVICE_ID=2` |

## 4. 异常与处理

无（一次通过，无手动干预）。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-22 | 创建 task；继承 Phase H / quickscan 的 native sparsity 与 skip-calibration 约束 | |
| 2026-09-22 | 回填 VSC-0 运行记录与完成状态；状态改 done | |
