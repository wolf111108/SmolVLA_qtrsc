# 结果记录（Results）

- **实验名称**：2026-09-23_phaseI_vision-attention-smoke
- **状态**：running（run #1 preflight Gate 失败，待修复后重跑）
- **最后更新**：2026-09-23

## 1. 摘要

run #1（2026-09-23，commit `ece908a`）在 **preflight raw 等价性 Gate 失败**：注入后 forward 与 checkpoint 原始 SDPA forward 在 bf16 下仅 **2/786432 个元素超差**（max abs 0.00677 > atol 0.005；max rel 2.64 出现在接近零的元素上）。诊断结论：适配器逻辑本身无错，是 **raw SDPA 与 raw eager 在 bf16 下的 kernel 累加路径差异**（SDPA 内部 fp32 累加 vs eager bf16 matmul）越过容差；CPU 单测同断言在 float32 + atol 1e-6 下全过佐证了这一点。校准与 rollout 未执行。

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
- 佐证：CPU 单测 `test_raw_equivalence` 在 **float32 + atol 1e-6** 下 eager/sdpa 两 backend 全过——差异来自 bf16 kernel 数值噪声而非适配器逻辑。

小型单元测试包含 eager/SDPA、masked/unmasked、scale 重载及量化统计、异常时上下文恢复。未运行完整 LeRobot 模型。

## 4. 结论

run #1 未通过 raw 等价性 Gate，实验链路在校准/rollout 之前终止；适配器逻辑本身经 CPU 单测与注入站点断言验证无误，当前卡点是 bf16 下 SDPA/eager 的 kernel 级噪声与容差设定不匹配。

## 5. 问题与后续

- 待决策修复方向：① 放宽 bf16 容差（atol 5e-3 → 1e-2~2e-2）；② 对照基准改为 eager 实现，保持严容差、真正隔离适配器差异（语义更干净）；③ 适配器内部升 fp32（不推荐，与「匹配 eager 计算」设计声明冲突）；
- 重跑前需归档 `outputs/2026-09-23_phaseI_vision-attention-smoke/` 与 `scales/2026-09-23_phaseI_vision-attention-smoke/`（run_smoke.sh 拒绝覆盖）；
- 另：HF_TOKEN 未设置导致 Hub 限流告警，建议设置后重跑。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-23 | 创建实验与待运行记录 | |
| 2026-09-23 | 回填 run #1：preflight Gate 失败（bf16 SDPA/eager kernel 差异，2/786432 超差），诊断与修复选项见 §3.1 / §5 | |
