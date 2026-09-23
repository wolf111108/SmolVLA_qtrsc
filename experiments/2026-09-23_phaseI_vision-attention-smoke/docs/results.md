# 结果记录（Results）

- **实验名称**：2026-09-23_phaseI_vision-attention-smoke
- **状态**：running（run #2 适配器验证通过；run #3 校准/rollout 待运行）
- **最后更新**：2026-09-23

## 1. 摘要

run #2 原始三方报告已上传并审阅：24 个用例的 adapter_vs_eager 最大/平均绝对误差均为0，24个site各调用2次；原生 eager 与 SDPA 有有限数值差异，超出容差的元素仅在 layer0（无mask 2个、有mask 1个）。按当时严格规则状态为 FAIL，校准与 rollout 尚未执行。现在仅为工程 smoke 调整验收规则，run #3 待本地执行。

## 2. 总结果表

| 组 | 结果 | 输出 |
|---|---|---|
| 小型 attention 单元测试 | 6 passed（CPU torch 2.14.0+cpu / transformers 4.52.4） | 非 benchmark |
| pytest（run_smoke.sh 内） | 6 passed（smolvla_eval / GPU 环境） | `outputs/2026-09-23_phaseI_vision-attention-smoke/run.log` |
| 模型加载 + 注入 | ✅ Replaced 0 Linear（预期，仅 QK/PV）；24 vision MatMul 站点断言通过 | 同上 |
| checkpoint raw gate | run #2 三方对照：adapter_vs_eager ✅（24/24 用例零超差）；eager_vs_sdpa ❌（仅 layer0，2+1 个元素）；adapter_vs_sdpa ❌（与 eager_vs_sdpa 完全一致） | `docs/preflight_run2.json` |
| VM1 Goal task0×1 | 未执行（脚本终止于 preflight） | — |

## 3. 分组结果与分析

### 3.1 run #1 preflight 失败详情

- 失败点：`preflight.py:48` `torch.testing.assert_close`（bf16 容差 atol=5e-3 / rtol=5e-2）；
- 超差幅度：Mismatched **2 / 786432**（0.0%），max abs **0.00677** @ (0, 320, 276)，max rel **2.640625** @ (0, 709, 551)（rtol 判据在接近零元素上天然敏感）；
- 对照两边：`attn._original_vision_forward`（checkpoint 默认 SDPA）vs 注入后 `vision_attention_forward`（eager 路径，raw 模式下 quant_qk/quant_pv 退化为普通 matmul）；
- 佐证：CPU 单测 `test_raw_equivalence` 在 **float32 + atol 1e-6** 下 eager/sdpa 两 backend 全过；这支持 backend 差异假设，但不能排除真实 checkpoint、依赖版本或 mask 路径问题。

### 3.2 run #2 三方对照诊断（新版 preflight，commit `e3d5b1e`）

| Gate | 结果 | 详情 |
|---|---|---|
| complete / coverage | ✅ / ✅ | 24 用例全跑完，24 站点各被调用 2 次 |
| **adapter_vs_eager** | ✅ | 全部 24 用例 0 mismatched（bf16，atol 1e-6 / rtol 1e-5）——适配器与原生 eager bit 级一致 |
| eager_vs_sdpa | ❌ | 仅 layer0：unmasked 2/786432（max abs 0.015625 @ [0,149,725]）、masked 1/786432（max abs 0.007812 @ [0,7,725]）；layer1–11 全部零超差 |
| adapter_vs_sdpa | ❌ | mismatched 计数与 max_error_index 与 eager_vs_sdpa 逐项相同 |

结论：已测输入下，适配器与原生 eager 输出数值完全一致；SDPA/eager 差异在多层存在，仅 layer0 有元素超出容差。具体 kernel 内部原因未定位，不能断言两者累加精度不同。run #2 按当时规则阻断校准/rollout。

## 4. 结论

本次12层独立随机输入、masked/unmasked共24用例支持适配器 raw 实现正确，不等价于真实图像完整前向或闭环等价。允许推进工程 smoke；FP8校准、量化rollout和稀疏统计尚待验证。

## 5. 问题与后续

- 已将完整覆盖、无异常、数值有限性、adapter/eager 严格等价设为硬门槛；本 smoke 配置将有限 backend 差异设为诊断，保留容差与误差记录。
- 归档旧 outputs/scales 后重跑 run_smoke.sh；新版报告为 schema_version=2，旧 run #2 证据不修改。
- 完成后回填校准、episode 与稀疏统计产物。正式量化掉点实验需 eager raw 基线。
- 当前修改未运行测试或实验。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-23 | 创建实验与待运行记录 | |
| 2026-09-23 | 回填 run #1：preflight Gate 失败（bf16 SDPA/eager kernel 差异，2/786432 超差），诊断与修复选项见 §3.1 / §5 | |
| 2026-09-24 | 回填 run #2 三方对照：adapter_vs_eager 全过（bit 级一致），差异全部来自原生 eager/sdpa backend（layer0，2+1 元素）；证据 `preflight_run2.json` | |

| 2026-09-23 | 审阅修正：归因降为假设；增加三方对照与失败报告，未重跑 | |

| 2026-09-23 | 审阅修正归因与范围；新增 smoke 门槛策略，run #3 待运行 | |
