# 实验日志（Logs）

- **实验名称**：2026-09-22_phaseI_vision-smmm-sparsity
- **状态**：done（VS1 task0×1ep）
- **最后更新**：2026-09-22

---

## 1. 当前状态

完整 Vision+VLM+Expert S\|MMM 1ep 实验已跑通，5 个 Gate 全 PASS。

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| config | ✅ ready | 2026-09-22 | full VLIN / S\|MMM / task0×1ep |
| summarizer | ✅ ready | 2026-09-22 | Vision 144 / VLM 320 / Expert 3200 row gate |
| runner | ✅ ready | 2026-09-22 | 1080 scale + sha256 + skip-calibration |
| VS1 run | ✅ done | 2026-09-22 | 5 Gate PASS；eval_s=454.5；结构 Gate 全中 |

## 2. 运行日志

| 日期 | 阶段 | 命令 / 配置 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-09-22 | setup | 创建标准实验结构 | ✅ | — | 未创建规范外文档 |
| 2026-09-22 | Gate 0 | `py_compile summarizer` + `pytest tests/test_sparsity_accounting.py` | ✅ | — | 12 passed in 4.60s |
| 2026-09-22 | Gate 1 | scale 隔离拷贝 + sha256 | ✅ | `scales/2026-09-22_phaseI_vision-smmm-sparsity/vs1_full_vlin_smmm_1ep` | source/target 1080 `.p`（864+72×3）bit-identical |
| 2026-09-22 | Gate 2 | VLIN routing + Vision scale coverage preflight | ✅ | — | Vision 72/72 sites、216/216 scale files、CALIBRATION COVERAGE GATE PASS |
| 2026-09-22 | Gate 3 | `main.py --config vs1 --skip-calibration` | ✅ | `outputs/2026-09-22_phaseI_vision-smmm-sparsity/vs1_full_vlin_smmm_1ep` | task0×1ep，SR=100%，eval_s=454.52 |
| 2026-09-22 | Gate 4 | primary CSV presence | ✅ | `.../sparsity/` | module 3665 / weight_static 297 / manifest 361 行 |
| 2026-09-22 | Gate 5 | S\|MMM summary | ✅ | `.../vision_smmm_sparsity_summary.csv` | SUMMARY GATE PASS |

关键 stdout：

```text
Replaced 224 Linear modules.
Replaced 72 vision/connector Linear modules.
Injected SmolVLA QuantizedMatMul (64 physical qk/pv objects)
Switched 360 quantized modules to mode=quant_forward
Quantized modules: 360

[sparsity] static weight collection finished: 296 QuantizedLinear layers
[sparsity] ... (workload rows: 3664)

=== FULL VLIN S|MMM SPARSITY ===
vision        runtime elem= 4.075% S|MMM= 48.643% | weight elem= 0.000556% S|MMM= 54.411%
vlm           runtime elem= 1.917% S|MMM= 53.975% | weight elem= 0.000552% S|MMM= 53.366%
expert        runtime elem= 3.024% S|MMM= 54.148% | weight elem= 0.000332% S|MMM= 55.075%
all_quantized runtime elem= 3.442% S|MMM= 51.288% | weight elem= 0.000489% S|MMM= 54.120%
metric : S|MMM v1 (sign+mantissa; exponent/hidden-1 excluded)
SUMMARY GATE: PASS
```

## 3. 后台任务

无。

## 4. 异常与处理

- `Warning: You are sending unauthenticated requests to the HF Hub`：无 `HF_TOKEN`，不影响运行。
- `conda run` 打印 `WARNING: overwriting environment variables set in the machine ... {'MUJOCO_GL'}`：conda 端 `MUJOCO_GL=egl` 覆盖 shell，为本仓库 EGL 约定，符合预期。
- 无失败、无 SIGTERM、无 OOM。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-22 | 创建实验 | |
| 2026-09-22 | VS1 跑通，5 Gate PASS，回填结果 | |