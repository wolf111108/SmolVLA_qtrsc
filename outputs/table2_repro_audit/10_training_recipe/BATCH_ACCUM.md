# PHASE 10B — Global Batch / Gradient Accumulation / 100k 语义审计

> 设备：h100 · lerobot_current @ 6adf515 · 2026-08-25

## 源码确认结论

### 1. effective batch 定义（lerobot_train.py:598-600）

```python
samples_per_step = cfg.batch_size * parallel_dims.dp_world_size
effective_batch_size = samples_per_step * cfg.accelerator.gradient_accumulation.steps
```

即：

```text
effective batch = micro batch × dp_world_size × grad_accum_steps
```

与手册 10B 的预期一致。

### 2. 训练主循环（lerobot_train.py:726）

```python
for _ in range(step, cfg.steps):
    ...
    update_policy(...)   # 内含 accelerator.accumulate(policy) + optimizer.step()
    step += 1
```

**`cfg.steps` 统计的是 micro-batch loop 次数**（每次循环 consume 一个 micro-batch）。

### 3. optimizer.step 在 accumulate 内（lerobot_train.py:193-232）

```python
with accelerator.accumulate(policy):   # accum==1 时是 no-op
    ...
    accelerator.backward(loss)
    if accelerator.sync_gradients and grad_clip_norm > 0:   # 只在 sync step clip
        ...
    optimizer.step()    # 非 sync micro-batch 上是 no-op
    optimizer.zero_grad()
    lr_scheduler.step()  # 每个 batch 都 step
```

关键：`optimizer.step()` 在 `accelerator.accumulate` 上下文内，只有 `sync_gradients`（即最后一个 micro-batch）时才真正更新权重。注释明确写 "Optimizer step (a no-op on non-final micro-batches under gradient accumulation)"。

### 4. LR scheduler 语义（重要发现）

`lr_scheduler.step()` 在**每个 micro-batch** 都调用，不是每个 optimizer update。这与手册 10.2 的 warmup 语义相关：
- 若 `warmup=100`，指 **100 个 micro-batch**，不是 100 个 optimizer update；
- 若 accum=4，则 warmup 覆盖的 optimizer update 只有 25 次。

这解释了为什么当前 preset 的 `scheduler_warmup_steps=1000` 与论文 `warmup=100` 差异巨大——它们可能在**不同的 step 语义**下定义。

## 100k 语义结论

| 配置 | optimizer updates | 说明 |
|---|---|---|
| batch64, accum=1, world=1 | `steps` = updates | 最干净 |
| batch16, accum=4, world=1 | `steps/4` = updates | 100k steps → 仅 25k updates |

**正式训练若用 accum>1，必须把 `cfg.steps` 设为 `100k × accum` 才能得到 100k optimizer updates。**

## 手册 14.1 checklist 对应项

- `[x] Effective global batch = 64` — 源码公式确认
- `[x] 100k step semantic verified` — 确认 `steps` = micro-batch loop，accum 下需折算
- `[ ] Scheduler choice explicitly documented` — 见 PHASE 10（warmup 语义未定）

## 待办

- PHASE 10（recipe 对照表）需要结合本结论：warmup 的 step 语义（micro-batch vs update）是填表关键；
- 若最终 accum=1（单卡 batch64 能放下），则 step 语义无歧义，100k = 100k updates。
