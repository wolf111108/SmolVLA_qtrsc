# PHASE 4 — Evaluation Protocol 审计

> 设备：h100 · 2026-08-25
> 性质：协议确认（无需实验），证据来自已完成的五 checkpoint 四 suite benchmark

## 4.1 论文 Table-2 协议 vs 我们实际使用

| 项 | 论文要求 | 我们实际 | 一致 |
|---|---|---|---|
| chunk_size | 50 | 50（policy config 默认） | ✅ |
| n_action_steps | 1（每步重新观察/预测） | **1** | ✅ |
| num_steps | 10（flow 推理步数） | 10 | ✅ |
| seed | 未明确（#2114 建议 1000） | 1000 | ✅ |
| 每 task episodes | 10 | 10 | ✅ |
| 每 suite tasks | 10 | 10 | ✅ |

## 4.2 关键：我们没有混用 n_action_steps=10

手册明确警告 #3264 的 `n_action_steps=10` 结果（63/93/81/56）**不能作为严格 Table-2 结果**。

我们的五模型四 suite 结果全部在 `n_action_steps=1` 下取得：

| Checkpoint | 协议 | 平均 SR |
|---|---|---|
| lerobot/smolvla_libero (A) | n_action_steps=1 | 69.2% |
| k1000dai (D) | n_action_steps=1 | 65.5% |
| HuggingFaceVLA/smolvla_libero | n_action_steps=1 | 61.2% |
| tiantianx (C) | n_action_steps=1 | 59.5% |

无一混用 10/30/50。

## 4.3 标准 eval 模板（我们所有评测都遵循）

```bash
lerobot-eval \
    --policy.path=<CHECKPOINT> \
    --policy.n_action_steps=1 \
    --policy.num_steps=10 \
    --env.type=libero \
    --env.task=<SUITE> \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --env.max_parallel_tasks=1 \
    --seed=1000 \
    --output_dir=<OUTPUT>
```

与手册 4.3 完全一致。

## 判定

**PASS: EVAL_PROTOCOL**（协议与论文一致，无 n_action_steps 混用）

## 回答手册 Q8

- Q8 n_action_steps 是否严格为 1？→ **是**，五模型四 suite 全部 `n_action_steps=1`
