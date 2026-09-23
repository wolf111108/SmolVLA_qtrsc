# 结果记录（Results）

- **实验名称**：2026-09-23_phaseI_vision-attention-smoke
- **状态**：done（run #6 全链路 PASS，check_outputs PASS）
- **最后更新**：2026-09-24

## 1. 摘要

Vision QK/PV FP8 PoT 工程 smoke 于 run #6（2026-09-24）全链路跑通：pytest 6 passed → preflight **PASS_WITH_BACKEND_DRIFT**（硬门槛全过：complete/coverage/finite/adapter_vs_eager）→ 校准（24 站点 × A/B/O 共 72 个 PoT scale 落盘）→ Goal task0×1ep **SR=100%**（avg/max reward 1.0，eval 78.2s）→ 稀疏统计导出（runtime 72 行 + workload 72 行）→ **check_outputs PASS**。Vision QK/PV 运行时 S\|MMM bit sparsity = **57.68%**，element sparsity = 7.24%。过程中发现并修复两个首次暴露的框架 bug（见 §3.3）。

## 2. 总结果表

| 组 | 结果 | 输出 |
|---|---|---|
| 小型 attention 单元测试 | 6 passed（CPU torch 2.14.0+cpu / transformers 4.52.4） | 非 benchmark |
| pytest（run_smoke.sh 内） | 6 passed（smolvla_eval / GPU 环境） | `outputs/2026-09-23_phaseI_vision-attention-smoke/run.log` |
| 模型加载 + 注入 | ✅ Replaced 0 Linear（预期，仅 QK/PV）；24 vision MatMul 站点断言通过 | 同上 |
| checkpoint raw gate | run #2 三方对照：adapter_vs_eager ✅（24/24 用例零超差）；eager_vs_sdpa ❌（仅 layer0，2+1 个元素）；adapter_vs_sdpa ❌（与 eager_vs_sdpa 完全一致） | `docs/preflight_run2.json` |
| VM1 Goal task0×1 | **run #6 done：SR=100%**（avg/max reward 1.0，eval 78.2s） | `outputs/.../vision_qkpv_fp8/result.json` |
| 稀疏统计 | **72 行 runtime（A/B/O × 24 站点）**；element 7.24% / S\|MMM bit **57.68%** | `outputs/.../vision_qkpv_fp8/sparsity/` |
| check_outputs | **PASS**（24 sites、72 scale、72 runtime rows、S\|MMM v1） | `outputs/.../vision_qkpv_fp8/coverage_summary.json` |

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

### 3.3 run #3–#6 迭代与框架 bug 修复

| run | 结果 | 发现 / 修复 |
|---|---|---|
| #3（`dc8f633`，backend 差异降级为报告） | preflight PASS_WITH_BACKEND_DRIFT、校准、rollout（SR=100%）全过，但 main.py 稀疏导出前防护报错 | **Bug 1**：MatMul-only workload 无 QuantizedLinear，`collect_model_weight_sparsity` 返回 0 触发防护。修复：仅当存在 QuantizedLinear 却未收集到、或无任何量化模块时报错 |
| #4 | 管线完成但 `module_sparsity.csv` 0 行 → check_outputs 断言失败 | **Bug 2**：`calibrate()` 新建 stat_manager 重绑定模块，新 SM `sparsity_enabled=False`，静默丢失 runtime 统计。修复①：重绑定时同步原 SM 的 sparsity 配置（chunk_size/unit 参数） |
| #5 | 同 #4：配置已同步，但模块仍绑在临时 SM，eval 统计进了临时 SM，导出侧原 SM 仍空 | 修复②：`calibrate()` 末尾（scale 落盘后）恢复模块绑定到原 SM，eval 统计累积到 `wrapper.stat_manager` |
| #6 | **全链路 PASS**：workload rows=72、runtime 72 行、check_outputs PASS（element 7.24% / bit 57.68%，S\|MMM v1） | — |

两个 bug 均为首次走「真校准 + 稀疏收集」路径才暴露（历史实验全部 `--skip-calibration` 复用 scale，不进 `calibrate()`）。

## 4. 结论

本次12层独立随机输入、masked/unmasked共24用例支持适配器 raw 实现正确，不等价于真实图像完整前向或闭环等价。run #6 完成 FP8 校准、量化 rollout（SR=100%，1ep 不作精度结论）与稀疏统计（runtime S\|MMM bit 57.68%），工程 smoke 目标全部达成；适配器接入链路（注入/校准/量化前向/统计收集）可复用。正式量化掉点实验需 eager raw 基线。

## 5. 问题与后续

- 已将完整覆盖、无异常、数值有限性、adapter/eager 严格等价设为硬门槛；本 smoke 配置将有限 backend 差异设为诊断，保留容差与误差记录。
- run #3–#5 归档于 `outputs/*_run3_partial`、`*_run4_no_runtime_stats`、`*_run5_no_runtime_stats`；run #6 正式产出在 `outputs/2026-09-23_phaseI_vision-attention-smoke/`。
- 框架修复（main.py 防护、calibrate() SM 绑定）已本地完成并随本次回填提交；建议审阅确认后合入。
- 1 episode 采样仅作工程验证；正式量化掉点实验需 eager raw 基线与多 episode 方差估计。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-23 | 创建实验与待运行记录 | |
| 2026-09-23 | 回填 run #1：preflight Gate 失败（bf16 SDPA/eager kernel 差异，2/786432 超差），诊断与修复选项见 §3.1 / §5 | |
| 2026-09-24 | 回填 run #2 三方对照：adapter_vs_eager 全过（bit 级一致），差异全部来自原生 eager/sdpa backend（layer0，2+1 元素）；证据 `preflight_run2.json` |
| 2026-09-24 | 回填 run #3–#6：两个框架 bug 修复（main.py MatMul-only 防护、calibrate() SM 绑定）；run #6 全链路 PASS（SR=100%、runtime 72 行、check_outputs PASS） | |

| 2026-09-23 | 审阅修正：归因降为假设；增加三方对照与失败报告，未重跑 | |

| 2026-09-23 | 审阅修正归因与范围；新增 smoke 门槛策略，run #3 待运行 | |
