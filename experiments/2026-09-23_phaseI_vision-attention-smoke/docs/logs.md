# 实验日志（Logs）

- **实验名称**：2026-09-23_phaseI_vision-attention-smoke
- **状态**：done（run #6 全链路 PASS，check_outputs PASS）
- **最后更新**：2026-09-25

## 1. 当前状态

实验完成：run #6 全链路 PASS（preflight PASS_WITH_BACKEND_DRIFT → 校准 72 scale → Goal task0×1ep SR=100% → 稀疏统计 72 行 → check_outputs PASS）。两个框架 bug 已修复并随提交 `846f798` 上传；详细迭代见 results.md §3.3。

## 2. 运行日志

| 日期 | 阶段 | 命令 | 状态 | 备注 |
|---|---|---|---|---|
| 2026-09-23 | 小型 CPU 单元测试 | python -m pytest tests/test_vision_attention.py -q | 6 passed | 用户要求停止运行验证前完成 |
| 2026-09-23 | checkpoint / rollout | scripts/run_smoke.sh | **failed（preflight Gate）** | run #1：pytest 6 passed → 模型加载 OK（Replaced 0 Linear 符合预期，QK/PV 走 MatMul 注入）→ preflight 注入 24 MatMul 站点断言通过 → **raw 等价性断言失败**：Mismatched 2/786432，max abs diff 0.00677 > atol 0.005（bf16 容差），max rel diff 2.64 > rtol 0.05 @ (0,709,551)；另有 HF_TOKEN 未设置告警。日志：`outputs/2026-09-23_phaseI_vision-attention-smoke/run.log` |
| 2026-09-24 | 三方对照诊断 | `python experiments/2026-09-23_phaseI_vision-attention-smoke/scripts/preflight.py configs/vision_qkpv_fp8.yaml`（commit `e3d5b1e` 新版） | **FAIL（backend Gate）** | run #2：adapter_vs_eager ✅ 24/24 用例零超差（bf16，atol 1e-6）；eager_vs_sdpa / adapter_vs_sdpa ❌ 仅 layer0（unmasked 2/786432 max abs 0.015625、masked 1/786432 max abs 0.007812），计数与索引完全一致；适配器与原生 eager bit 级一致。证据：`outputs/.../preflight.json`（副本 `docs/preflight_run2.json`） |
| 2026-09-24 | 完整 smoke（commit `dc8f633`） | `bash scripts/run_smoke.sh`（归档 run#1/#2 产出后） | **failed（main.py 防护）** | run #3：preflight **PASS_WITH_BACKEND_DRIFT** ✅、校准 72 scale ✅、rollout **SR=100%** ✅，但 main.py 稀疏导出前报 `no QuantizedLinear weight sparsity`。产出归档 `outputs/*_run3_partial` |
| 2026-09-24 | 修复 Bug 1 后重跑 | `bash scripts/run_smoke.sh`（归档 run#3 后） | **failed（check_outputs）** | run #4：管线完成（SR=100%）但 `module_sparsity.csv` 0 行 → check_outputs 断言失败。发现 Bug 2：`calibrate()` 新建 stat_manager 重绑定致 runtime 统计丢失。产出归档 `outputs/*_run4_no_runtime_stats` |
| 2026-09-24 | 修复 Bug 2-① 后重跑 | `bash scripts/run_smoke.sh`（归档 run#4 后） | **failed（check_outputs）** | run #5：sparsity 配置已同步（日志可见 `sparsity collection preserved`）但模块仍绑在临时 SM，导出侧原 SM 仍空。补修复②：校准后恢复 SM 绑定。产出归档 `outputs/*_run5_no_runtime_stats` |
| 2026-09-24 | 修复 Bug 2-② 后重跑 | `bash scripts/run_smoke.sh`（归档 run#5 后） | **PASS（全链路）** | run #6：preflight PASS_WITH_BACKEND_DRIFT → 校准 72 scale（7.2s）→ rollout **SR=100%**（78.2s）→ 稀疏导出 workload/runtime 各 72 行 → **check_outputs PASS**（element 7.24% / S\|MMM bit 57.68%）。正式产出在 `outputs/2026-09-23_phaseI_vision-attention-smoke/`；证据副本 `docs/{preflight,coverage_summary,result}_run6.json` |

## 3. 后台任务

无。

## 4. 异常与处理

| 日期 | 异常 | 处理 |
|---|---|---|
| 2026-09-23 | run #1 preflight：`torch.testing.assert_close` 失败——raw SDPA 原始 forward 与注入 quant_qk/quant_pv 后的 forward 不等价（bf16 下 max abs 0.00677 > 0.005，仅 2/786432 个元素超差） | run #2 三方对照证实：适配器与原生 eager bit 级一致，差异全部来自原生 SDPA/eager backend（仅 layer0 少量元素）；审阅方在 `dc8f633` 将 backend 差异降级为单独报告 |
| 2026-09-24 | run #3：main.py 报 `sparsity.enabled=true but no QuantizedLinear weight sparsity was collected` | **Bug 1**（框架，首次暴露）：MatMul-only workload 无 QuantizedLinear，防护误伤。修复：仅当存在 QuantizedLinear 却未收集到、或无任何量化模块时报错（`846f798`） |
| 2026-09-24 | run #4/#5：`module_sparsity.csv` 导出 0 行，runtime 统计全部丢失 | **Bug 2**（框架，首次暴露）：`calibrate()` 新建 stat_manager 重绑定模块，新 SM `sparsity_enabled=False` 且校准后绑定未恢复，eval 统计与导出侧 SM 分离。修复①同步 sparsity 配置、②校准后恢复绑定（`846f798`）。历史实验全走 `--skip-calibration` 故未触发 |
| 2026-09-23→24 | 环境告警（不阻断）：HF_TOKEN 未设置（Hub 限流）、libtorchcodec 加载失败（自动回退 pyav）、robosuite 私有宏缺失 | 未处理，不影响结果；建议后续设置 HF_TOKEN |

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-23 | 创建日志；记录本地待运行状态 | |
| 2026-09-23 | 更新诊断脚本与文档；遵照用户要求未运行测试或实验 | |
| 2026-09-25 | 补充 run #3–#6 运行记录、异常表（两个框架 bug 的发现与修复）、状态更新为 done | |

### run #3 准备

验收策略已修改：适配器/覆盖/有限性仍为硬门槛，有限 backend 差异单独报告。没有执行测试、校准或 rollout；历史 run #2 JSON 保持原样。


### 2026-09-25 审阅后框架修复

逐模块保存原统计器并用 finally 恢复，覆盖提前返回和异常；移除临时 manager 的 sparsity 配置复制；main.py 类型判断改为 isinstance。仅修改代码，未运行测试或实验，修复后回归待本地执行。run #6 仍为修复前的历史已完成结果。
