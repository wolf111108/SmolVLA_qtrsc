# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**（区别于 `results.md` 的结果记录）。
> 每次启动/完成一次运行、发现或修复异常时追加一条；章节为固定结构，不可删除；无内容写「无」。
> 运行日志按时间**正序**追加，最新状态反映在「当前状态」与头部字段。

- **实验名称**：2026-09-22_quickscan_smmm-bit-sparsity
- **状态**：done
- **最后更新**：2026-09-22

---

## 1. 当前状态

S4 已完成，SUMMARY GATE: PASS。

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| config/runner/summarizer | ✅ ready | 2026-09-22 | FP8 PoT task0×1ep；scale 三重防护 |
| 审阅修正（P0/P1） | ✅ done | 2026-09-22 | commit `ea9aa6f`：补 sparsity.enabled、统一 fixed-chunk 编码 |
| S4v1（带 P0 问题） | ❌ aborted | 2026-09-22 | 未生成 module_sparsity.csv（佐证 P0）；已终止并删除产物 |
| S4v2（修复版） | ✅ done | 2026-09-22 | Gate 全 PASS；pooled S\|MMM bit 54.11% |

## 2. 运行日志

| 日期 | 阶段 | 命令 / 配置 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-09-22 | S4v1 | `run_smmm_quickscan.sh` | ❌ aborted | （已删除） | P0：config 缺 `sparsity.enabled`，无 CSV 产出 |
| 2026-09-22 | S4v2 | `run_smmm_quickscan.sh`（nohup，PID 2607070） | ✅ | `outputs/.../s4_fp8_smmm` | Gate0 pytest 12 passed；Gate1 scale 复制 864 + sha256 bit-identical；Gate2 rollout task0×1ep；Gate3 汇总 **SUMMARY GATE: PASS**（行数 320 VLM + 3200 Expert = 3520，与 09-15 协议一致）；pooled runtime S\|MMM bit **54.11%** |

## 3. 后台任务

| PID | 阶段 | 启动时间 | 状态 | 备注 |
|---|---|---|---|---|
| 2607070 | S4v2 | 2026-09-22 | ✅ finished | nohup 日志 `outputs/2026-09-22_quickscan_smmm-bit-sparsity/s4_run.log` |

## 4. 异常与处理

- **S4v1 因 P0（config 缺失 `sparsity.enabled`）无法采集**：审阅发现后终止，未生成 module_sparsity.csv（验证了审阅判断）。已在 commit `ea9aa6f` 补齐完整 `sparsity:` 段。
- **P1（unit path 仍为旧 1.MMM）**：已把 `_fp_tensor_to_mantissa_fixed_chunk` 重写为 S\|MMM，E4M3/FP16 走 cast-读位快路径（与 raw 路径密集采样 1518/1518 = 100% 一致），bf16 走数值重构（`signbit` 保留 −0.0、subnormal 单独恢复）。
- **summarizer 行数 gate 阈值首次写错**（猜 416/3104）：实际 320 VLM + 3200 Expert = 3520，与 09-15 协议一致（`112×2 + 32×3` / `320×10`），已修正为 320/3200。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-22 | 创建文档 | |
| 2026-09-22 | 回填 S4 结果：S\|MMM pooled runtime bit = **54.11%**（+12.10pp vs 旧 1.MMM 42.01%）；验证审阅的 +12.5pp 预测；Gate 全 PASS | |
