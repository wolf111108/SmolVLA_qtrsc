# 实验日志（Logs）

- **实验名称**：2026-09-23_phaseI_vision-attention-smoke
- **状态**：running（run #2 已审阅，run #3 待本地运行）
- **最后更新**：2026-09-23

## 1. 当前状态

run #1 在 raw gate 失败；三方对照与失败报告已修改，新版等待用户本地运行。校准/rollout 仍未执行。

## 2. 运行日志

| 日期 | 阶段 | 命令 | 状态 | 备注 |
|---|---|---|---|---|
| 2026-09-23 | 小型 CPU 单元测试 | python -m pytest tests/test_vision_attention.py -q | 6 passed | 用户要求停止运行验证前完成 |
| 2026-09-23 | checkpoint / rollout | scripts/run_smoke.sh | **failed（preflight Gate）** | run #1：pytest 6 passed → 模型加载 OK（Replaced 0 Linear 符合预期，QK/PV 走 MatMul 注入）→ preflight 注入 24 MatMul 站点断言通过 → **raw 等价性断言失败**：Mismatched 2/786432，max abs diff 0.00677 > atol 0.005（bf16 容差），max rel diff 2.64 > rtol 0.05 @ (0,709,551)；另有 HF_TOKEN 未设置告警。日志：`outputs/2026-09-23_phaseI_vision-attention-smoke/run.log` |
| 2026-09-24 | 三方对照诊断 | `python experiments/2026-09-23_phaseI_vision-attention-smoke/scripts/preflight.py configs/vision_qkpv_fp8.yaml`（commit `e3d5b1e` 新版） | **FAIL（backend Gate）** | run #2：adapter_vs_eager ✅ 24/24 用例零超差（bf16，atol 1e-6）；eager_vs_sdpa / adapter_vs_sdpa ❌ 仅 layer0（unmasked 2/786432 max abs 0.015625、masked 1/786432 max abs 0.007812），计数与索引完全一致；适配器与原生 eager bit 级一致。证据：`outputs/.../preflight.json`（副本 `docs/preflight_run2.json`） |

## 3. 后台任务

无。

## 4. 异常与处理

| 日期 | 异常 | 处理 |
|---|---|---|
| 2026-09-23 | run #1 preflight：`torch.testing.assert_close` 失败——raw SDPA 原始 forward 与注入 quant_qk/quant_pv 后的 forward 不等价（bf16 下 max abs 0.00677 > 0.005，仅 2/786432 个元素超差） | 待诊断。注意本应为 raw 模式（`_inject_..._matmul(..., 'raw', ...)`），超差可能来自：① raw 等价路径数值本身与 SDPA 实现有微小 bf16 差异；② 注入路径在 raw 模式下未完全走原始实现。修复后需归档旧 outputs/scales 再重跑（run_smoke.sh 会拒绝覆盖） |

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-23 | 创建日志；记录本地待运行状态 | |

| 2026-09-23 | 更新诊断脚本与文档；遵照用户要求未运行测试或实验 | |

### run #3 准备

验收策略已修改：适配器/覆盖/有限性仍为硬门槛，有限 backend 差异单独报告。没有执行测试、校准或 rollout；历史 run #2 JSON 保持原样。
