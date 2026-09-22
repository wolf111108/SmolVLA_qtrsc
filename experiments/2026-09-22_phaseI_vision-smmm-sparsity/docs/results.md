# 结果记录（Results）

- **实验名称**：2026-09-22_phaseI_vision-smmm-sparsity
- **状态**：ready
- **最后更新**：2026-09-22

---

## 1. 摘要

待运行。

## 2. 总结果表

| Component | Runtime element sparsity | Runtime S|MMM bit sparsity | Weight element sparsity | Weight S|MMM bit sparsity | 状态 |
|---|---:|---:|---:|---:|---|
| Vision | — | — | — | — | pending |
| VLM | — | — | — | — | pending |
| Expert | — | — | — | — | pending |
| all_quantized pooled | — | — | — | — | pending |

## 3. 分组结果与分析

### 3.1 Vision / VLM / Expert

待运行。

### 3.2 与历史 1MMM 的关系

历史 VSC 的 1MMM 只能作为旧口径参考。正式比较必须明确 old=1MMM、new=S|MMM，不能把两者视为同一 bit metric。

## 4. 结论

待运行。

## 5. 问题与后续

- 1ep 完成后先检查 Vision/VLM/Expert 是否仍落在相近的 S|MMM 区间。
- 若需要稳定性验证，在本实验目录内新增 3ep/10ep config 变体，不新建新的顶层实验。

## 6. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-22 | 创建实验；暂时只配置 task0×1ep | |