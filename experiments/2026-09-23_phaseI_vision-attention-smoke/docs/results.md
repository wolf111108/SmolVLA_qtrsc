# 结果记录（Results）

- **实验名称**：2026-09-23_phaseI_vision-attention-smoke
- **状态**：running（run #1 preflight Gate 失败，待修复后重跑）
- **最后更新**：2026-09-23

## 1. 摘要

run #1（2026-09-23，commit `ece908a`）在 preflight raw 等价性 Gate 失败。回填报告为 BF16 下 2/786432 个元素超差，max abs 0.00677、max rel 2.64。初步怀疑 SDPA/eager 的数值路径差异；尚未完成同一 checkpoint 的原生 eager 对照，不能认定适配器逻辑已完全验证。校准与 rollout 未执行。原始 run.log 未上传，本记录中的运行数值来自回填。

## 2. 总结果表

| 组 | 结果 | 输出 |
|---|---|---|
| 小型 attention 单元测试 | 6 passed（CPU torch 2.14.0+cpu / transformers 4.52.4） | 非 benchmark |
| pytest（run_smoke.sh 内） | 6 passed（smolvla_eval / GPU 环境） | `outputs/2026-09-23_phaseI_vision-attention-smoke/run.log` |
| 模型加载 + 注入 | ✅ Replaced 0 Linear（预期，仅 QK/PV）；24 vision MatMul 站点断言通过 | 同上 |
| checkpoint raw gate | **failed**（bf16 断言超差，见 §3.1） | 无 preflight.json |
| VM1 Goal task0×1 | 未执行（脚本终止于 preflight） | — |

## 3. 分组结果与分析

### 3.1 run #1 preflight 失败详情

- 失败点：`preflight.py:48` `torch.testing.assert_close`（bf16 容差 atol=5e-3 / rtol=5e-2）；
- 超差幅度：Mismatched **2 / 786432**（0.0%），max abs **0.00677** @ (0, 320, 276)，max rel **2.640625** @ (0, 709, 551)（rtol 判据在接近零元素上天然敏感）；
- 对照两边：`attn._original_vision_forward`（checkpoint 默认 SDPA）vs 注入后 `vision_attention_forward`（eager 路径，raw 模式下 quant_qk/quant_pv 退化为普通 matmul）；
- 佐证：CPU 单测 `test_raw_equivalence` 在 **float32 + atol 1e-6** 下 eager/sdpa 两 backend 全过；这支持 backend 差异假设，但不能排除真实 checkpoint、依赖版本或 mask 路径问题。

小型单元测试包含 eager/SDPA、masked/unmasked、scale 重载及量化统计、异常时上下文恢复。未运行完整 LeRobot 模型。

## 4. 结论

run #1 未通过 raw 等价性 Gate，校准/rollout 前停止。24 个模块创建不等于 24 个算子完整运行覆盖，最终调用计数检查尚未执行。待三方对照确认失败原因。

## 5. 问题与后续

- 已修改 preflight：同一 checkpoint/input/mask 下比较原生 eager、新适配器 raw、原生 SDPA；严格 adapter/eager 检查与 backend 检查分别记录。
- 保持原 SDPA 对照容差不变；即使 adapter/eager 通过，backend 超差仍会阻止 calibration/rollout。
- 每个 case 都记录误差、超差数和运行异常，遍历所有层后统一判定。preflight.json 在失败时也保留；新版待本地运行。
- 判据为 `abs(actual-expected) <= atol + rtol*abs(expected)`，不能仅凭 max_abs > atol 判定失败原因。
- 重跑前需归档 `outputs/2026-09-23_phaseI_vision-attention-smoke/` 与 `scales/2026-09-23_phaseI_vision-attention-smoke/`（run_smoke.sh 拒绝覆盖）；
- 另：HF_TOKEN 未设置导致 Hub 限流告警，建议设置后重跑。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-23 | 创建实验与待运行记录 | |
| 2026-09-23 | 回填 run #1：preflight Gate 失败（bf16 SDPA/eager kernel 差异，2/786432 超差），诊断与修复选项见 §3.1 / §5 | |

| 2026-09-23 | 审阅修正：归因降为假设；增加三方对照与失败报告，未重跑 | |
