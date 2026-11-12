# Phase O — RTL BG / padded weight 统计结果

## 摘要

状态：**implemented / CPU verified / pending GPU run**。新增统计、单任务入口、报表及回归检查完成；没有填写真实 SmolVLA 稀疏度结果。

## 总结果表

| pa | pw | VLM A BG2 | VLM W BG2 | VLM W BG4 | Expert A BG2 | Expert W BG2 | Expert W BG4 |
|---:|---:|---|---|---|---|---|---|
| 2 | 1024 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 512 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 256 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 16 | 128 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 32 | 64 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 64 | 32 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

运行完成后从 `bit_groups/summary.md` 填入；每个 BG level 的结果和 full/tail 分母见 CSV。

## 分组结果与分析

CPU 测试在真实生产 Linear、QK、PV quant_forward 入口上检查：开启和关闭 observer 的输出逐元素完全相同，捕获的编码与实际 normal path 相同，静态权重只计一次。随机小张量仅用于单元测试，不代表模型任务实测。

全部 INT8/INT4 值在 BG2/BG4 下精确重构通过；负数 BG0 始终非零，INT4 符号扩展高组恒为零。由定义可推得 INT4 的补齐整体 BG 跳过率至少 50%，扩展高组单独为 100%；这是数学性质，不能替代实测表。

## 结论

本轮验证：新增 14 项测试、Phase N 10 项测试、Phase L 10 项测试全部通过；三个 phase 配置 preflight 通过。CUDA/SmolVLA/LIBERO rollout、芯片 RTL 仿真与硬件周期/PPA 不在本轮执行范围。

## 问题与后续

在已有 GPU 评测环境运行新实验脚本，保留完整 CSV/尺度/coverage；检查大 pa/pw 的 tail_pack_fraction，避免将尾包当作完整包。静态权重统计与带动态浮点保护的 runtime weight 不可混淆。

## 修订记录

- 2026-11-12：新增 Phase O，保留 Phase N 原码口径；完成 CPU 回归，实测值待填。
