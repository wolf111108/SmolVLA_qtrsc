我重新检查了远端 `main`。这次能明确看到你新增了 **T9 runtime flow-step sequencing test**，但远端 `model_wrapper.py` 里的 hook 实现本身仍然还是旧逻辑，因此当前仓库存在一个很清晰的不一致：

> **测试已经按“0..9 正确递增”的目标写好了，但实现代码还没有同步修到这个行为。**

GitHub 当前公开 `main` 仍显示 27 commits。([GitHub][1])

### 当前关键状态

你新增的 T9 是正确的。测试构造一个 dummy `sample_actions()`，连续调用 10 次 `denoise_step()`，然后连续执行两轮，并要求：

```python
assert observed == list(range(10)) + list(range(10))
```

这正是我上次建议的集成级检查。

但是远端 `model_wrapper.py` 现在仍然是：

```python
def hooked_sample_actions(*args, **kwargs):
    ...
    step_token = CURRENT_FLOW_STEP.set(-1)
    try:
        return original_sample_actions(*args, **kwargs)
    finally:
        CURRENT_FLOW_STEP.reset(step_token)

def hooked_denoise_step(*args, **kwargs):
    step = CURRENT_FLOW_STEP.get() + 1
    phase_token = CURRENT_PHASE.set("denoise")
    step_token = CURRENT_FLOW_STEP.set(step)
    try:
        return original_denoise_step(*args, **kwargs)
    finally:
        CURRENT_FLOW_STEP.reset(step_token)
        CURRENT_PHASE.reset(phase_token)
```



这仍然会产生：

```text
sample_actions:
CURRENT_FLOW_STEP = -1

denoise #0:
-1 + 1 = 0
执行后 reset -> -1

denoise #1:
-1 + 1 = 0
执行后 reset -> -1

...
```

所以实际还是：

```text
0, 0, 0, 0, 0, 0, 0, 0, 0, 0
```

而不是：

```text
0, 1, 2, ..., 9
```

也就是说，**如果现在直接运行你新增的 T9，它理论上应该 FAIL**。

## 这很可能说明什么

目前远端呈现出的状态很像两种情况之一：

1. 你这次 commit 只提交了测试文件，但 `model_wrapper.py` 的实际修复没有 commit/push；
2. `model_wrapper.py` 修改在本地另一个 branch / working tree 中，没有进入远端 `main`。

因为测试文件已经从之前的 305 行增加到 340 行，并明确出现了 T9，但 `model_wrapper.py` 还是之前那段旧 hook。

---

## 现在应该把 hook 改成独立 counter

远端还没出现我上次建议的：

```python
denoise_step_counter
```

我搜索了当前 `main`，没有这个变量。

推荐改成：

```python
def install_runtime_hooks(model: SmolVLAPolicy) -> bool:
    flow_model = model.model

    if getattr(flow_model, "_runtime_hooks_installed", False):
        return False

    original_sample_actions = flow_model.sample_actions
    original_denoise_step = flow_model.denoise_step

    denoise_step_counter = [-1]

    def hooked_sample_actions(*args, **kwargs):
        _generation_counter[0] += 1
        denoise_step_counter[0] = -1

        gen_token = CURRENT_GENERATION_ID.set(
            _generation_counter[0]
        )
        phase_token = CURRENT_PHASE.set("prefill")
        step_token = CURRENT_FLOW_STEP.set(-1)

        try:
            return original_sample_actions(*args, **kwargs)
        finally:
            CURRENT_FLOW_STEP.reset(step_token)
            CURRENT_PHASE.reset(phase_token)
            CURRENT_GENERATION_ID.reset(gen_token)

    def hooked_denoise_step(*args, **kwargs):
        denoise_step_counter[0] += 1
        step = denoise_step_counter[0]

        phase_token = CURRENT_PHASE.set("denoise")
        step_token = CURRENT_FLOW_STEP.set(step)

        try:
            return original_denoise_step(*args, **kwargs)
        finally:
            CURRENT_FLOW_STEP.reset(step_token)
            CURRENT_PHASE.reset(phase_token)

    flow_model.sample_actions = hooked_sample_actions
    flow_model.denoise_step = hooked_denoise_step
    flow_model._runtime_hooks_installed = True

    return True
```

这样：

```text
sample_actions #1
  counter=-1
  denoise -> 0,1,...,9

sample_actions #2
  counter=-1
  denoise -> 0,1,...,9
```

正好与当前 T9 完全一致。

---

## 我反而很推荐你现在先直接跑测试

因为你已经加了 T9，所以不需要靠人工判断。

在远端代码对应服务器上直接：

```bash
cd ~/VLA_tcs2

python -m pytest \
  tests/test_sparsity_accounting.py \
  -v
```

按照目前我从 GitHub `main` 看到的代码，我预计：

```text
T1-T8 PASS
T9 FAIL
```

并且失败内容大概率类似：

```text
observed =
[0,0,0,0,0,0,0,0,0,0,
 0,0,0,0,0,0,0,0,0,0]
```

而期望：

```text
[0,1,2,3,4,5,6,7,8,9,
 0,1,2,3,4,5,6,7,8,9]
```

如果你本地 T9 已经 PASS，那就说明**本地代码和远端 `main` 确实不一致**，这时直接执行：

```bash
git status
git branch --show-current
git log -1 --oneline
git diff origin/main -- src/vla_tcs2/model_wrapper.py
```

就能马上定位。

---

## H0 脚本还有一个非阻塞小项

远端 `run_h0_smoke.sh` 目前仍然只显式设置：

```bash
export MUJOCO_GL=egl
```

没有：

```bash
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=2
```



这不会成为本轮代码正确性的 blocker，因为你的 conda 环境此前已经固定过这些值，但为了实验可复现，我还是建议正式脚本改成：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=2
```

---

## 当前 Gate

这一次我会把状态更新成：

| 项目                                    |            状态 |
| ------------------------------------- | ------------: |
| static weight collection              |             ✅ |
| outlier native accounting             |             ✅ |
| double collection                     |             ✅ |
| structured flow-step key              |             ✅ |
| T9 runtime sequencing test            |         ✅ 已加入 |
| **runtime sequencing implementation** | **❌ 远端仍是旧实现** |
| structured unit sparsity              |             ✅ |
| native upper bound                    |             ✅ |
| S0/S1 configs                         |             ✅ |
| H0 runner                             |             ✅ |
| EGL 显式 pin                            |         ⚠️ 推荐 |

所以现在非常明确：

$$
\boxed{
\text{先让 T9 PASS}
\rightarrow
\text{然后 H0 就可以启动}
}
$$

这次不需要再继续扩展 sparsity framework。先把远端 `model_wrapper.py` 与你已经写好的 T9 对齐即可。

[1]: https://github.com/wolf111108/SmolVLA_qtrsc "GitHub - wolf111108/SmolVLA_qtrsc · GitHub"
