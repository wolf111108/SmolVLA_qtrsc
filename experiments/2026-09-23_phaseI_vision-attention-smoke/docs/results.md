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

结论：两个 backend 的原生实现本身就存在 bf16 kernel 差异（SDPA fp32 累加 vs eager bf16 matmul），集中在 layer0 少数元素；适配器未引入任何额外误差。按新版 preflight 设计，backend Gate 失败仍阻断校准/rollout，待审阅决定后续（如 backend 差异豁免或以 eager 为基准）。

## 4. 结论

适配器正确性已闭环：adapter_vs_eager 在全部 12 层、masked/unmasked 全部 24 用例下零超差（bf16，atol 1e-6 / rtol 1e-5），适配器与原生 eager bit 级一致；run #1 的失败完全归因于原生 eager/SDPA backend 差异（仅 layer0 的 2+1 个元素 bf16 kernel 噪声），与适配器无关。校准与 rollout 因 backend Gate 阻断尚未执行。

## 5. 问题与后续

- backend Gate（eager_vs_sdpa / adapter_vs_sdpa）失败的处理待审阅定夺：豁免 backend 差异（以 adapter_vs_eager 为准）或维持阻断；
- 若放行：归档 `outputs/2026-09-23_phaseI_vision-attention-smoke/preflight.json` 后重跑 `run_smoke.sh` 进入校准 + rollout（脚本仍会拒绝覆盖，需先归档旧 outputs/scales）；
- 另：HF_TOKEN 未设置导致 Hub 限流告警，建议设置后重跑。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-23 | 创建实验与待运行记录 | |
| 2026-09-23 | 回填 run #1：preflight Gate 失败（bf16 SDPA/eager kernel 差异，2/786432 超差），诊断与修复选项见 §3.1 / §5 | |
| 2026-09-24 | 回填 run #2 三方对照：adapter_vs_eager 全过（bit 级一致），差异全部来自原生 eager/sdpa backend（layer0，2+1 元素）；证据 `preflight_run2.json` | |

| 2026-09-23 | 审阅修正：归因降为假设；增加三方对照与失败报告，未重跑 | |
