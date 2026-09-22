# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**。运行日志按时间正序追加。

- **实验名称**：2026-09-15_phaseI_vision-quantization / task: vision-sparsity-compute
- **状态**：running（VSC-0 sparsity ✅ / compute ❌；VSC-1 pending）
- **最后更新**：2026-09-22

---

## 1. 当前状态

VSC-0 rollout 已完成；**sparsity Gate 有效，但 compute Gate 经复核判定存在 instrumentation bug**。VSC-1 workload-fix rerun 待运行。

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| config / runner / summarizer | ✅ ready | 2026-09-22 | 复用 VLIN scales；1ep workload characterization |
| VSC-0 run | ⚠ partial-valid | 2026-09-22 | sparsity CSV 有效；compute summary 因 MatMul MAC shape bug 作废 |
| workload exporter fix | ✅ merged | 2026-09-22 | MatMul MAC 改为 `O.numel() × A.shape[-1]`；新增 `MAC_semantics=matmul_physical_exact_v2` + T12 regression |
| VSC-1 rerun | ⏳ pending | — | 同一 VLIN scales / task0 / seed1000，只重跑 workload characterization |

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

### 4.1 VSC-0 compute instrumentation bug（P0）

首轮 summarizer 虽 Gate PASS，但复核 core exporter 后发现 MatMul MAC shape accounting 错误：

```text
collect_quant_tensor(A) -> module_last_dims = A shape-derived
collect_quant_tensor(B) -> overwrite
collect_quant_tensor(O) -> overwrite again
export_workload_csv()   -> reads the final O-derived dims as K/N
```

因此 VLM/Expert QK/PV 的 MACs 不可信，连带 `688.50 GFLOPs / 88.3%` coverage 作废。该问题**不影响 module_sparsity / weight_sparsity 的 numerator/denominator counters**。

处理：

1. core 新增按 `(module_id, phase, flow_step, attention_kind)` 的 MatMul workload accumulator；
2. A role 记录 inner K，O role 用 `O.numel() × K` 累计 physical MACs；
3. workload CSV 增加 `MAC_semantics=matmul_physical_exact_v2`；
4. 新增 T12 synthetic batched-matmul regression；
5. VSC summarizer 拒绝 legacy MatMul rows；
6. 新建 VSC-1 同条件 rerun，避免覆盖 VSC-0 原始记录。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-22 | 创建 task；继承 Phase H / quickscan 的 native sparsity 与 skip-calibration 约束 | |
| 2026-09-22 | 回填 VSC-0 首轮运行记录 | |
| 2026-09-22 | **P0 compute audit**：sparsity 结果保留有效；688.50 G / 88.3% 作废。修复 MatMul physical-MAC accounting，新增 T12 + VSC-1 rerun | |
