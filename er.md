我重新检查了远端 `main`，这次上一次指出的 **double collection P0 已经正确修掉**，`native sparsity` 的 upper bound 也修正了。不过继续顺着 runtime hook 往下审计时发现了一个新的、而且会直接破坏 Phase H flow-step 实验的 **P0 blocker**：

> **当前 `flow_step` hook 实际上会把每一个 denoise step 都标成 `0`，不会得到 0,1,...,9。**

所以目前结论是：

$$
\boxed{\text{H0 READY = NO}}
$$

但现在只剩这一处真正影响实验语义的 P0，修完后基本就可以启动 H0。

---

## 这次已经确认修好的部分

上次的核心问题已经正确处理。

`collect_quant_activation()` 现在已经变成真正的 **audit-only**：只增加 legacy audit counter，然后立即 `return`；structured sparsity 只由 `_collect_linear_runtime()` / `_collect_matmul_runtime()` → `collect_quant_tensor()` 收集，因此 activation/A/B 不会再被重复累计。你还补了 T8 专门验证这一点。

现在的 structured key 也已经是：

```python
(
    module_id,
    phase,
    flow_step,
    tensor_role,
    attention_kind,
)
```

因此从 `stat_manager` 这一侧看，flow-step 分桶已经完全支持。

`module_sparsity.csv` 的 `ideal_sparse_upper_bound` 也已经不再偷偷使用 reported sparsity，而是明确基于：

```python
sparse_bit_rate_native
```

计算：

$$
U_\text{native}
=
\frac{1}{1-S_{\text{bit,native}}}
$$

这一点已经正确。

另外以下也都 PASS：

* `export.dir: null` fallback 正确；
* `chunk_size` 真正接入；
* `bit_group_size/dim_group_size` 真正接入；
* rollout 后调用 static weight collector；
* `weight_sparsity_static.csv`；
* outlier partition；
* structured unit sparsity；
* static weight 空结果保护；
* S0/S1 继续复用 Phase G scale；
* H0 用 `--skip-calibration`。

S0/S1 配置本身也没漂移：S0 仍是 FP8-all，S1 仍是 VLM raw + Expert W4/AO FP8，二者都固定 `n_action_steps=10, num_steps=10, Goal, seed=1000`。

---

# 当前真正的 P0：flow step 计数逻辑有 bug

问题在 `install_runtime_hooks()`。

现在外层进入 `sample_actions` 时：

```python
step_token = CURRENT_FLOW_STEP.set(-1)
```

然后每次进入 `denoise_step`：

```python
step = CURRENT_FLOW_STEP.get() + 1
phase_token = CURRENT_PHASE.set("denoise")
step_token = CURRENT_FLOW_STEP.set(step)

try:
    return original_denoise_step(...)
finally:
    CURRENT_FLOW_STEP.reset(step_token)
    CURRENT_PHASE.reset(phase_token)
```

代码当前就是这个结构。

表面看像：

```text
-1 -> 0 -> 1 -> 2 -> ... -> 9
```

但实际不是。

第一次 `denoise_step`：

```text
CURRENT_FLOW_STEP = -1
step = -1 + 1 = 0

set(0)
执行 denoise step 0
reset(step_token)
```

而这里的：

```python
CURRENT_FLOW_STEP.reset(step_token)
```

会把 ContextVar 恢复成进入该函数之前的：

```text
-1
```

于是第二次 `denoise_step` 再进来：

```text
CURRENT_FLOW_STEP = -1
step = 0
```

第三次仍然：

```text
0
```

最终实际得到的是：

```text
denoise:
0
0
0
0
0
0
0
0
0
0
```

而不是：

```text
0
1
2
3
4
5
6
7
8
9
```

这会导致你虽然已经实现了完美的五维 structured key，但所有 denoise 数据仍然落进：

```text
flow_step = 0
```

一个桶里。

所以 Figure H2：

```text
flow step 0 → 9 sparsity evolution
```

现在跑出来是不成立的。

---

# 推荐修法

不要利用 `CURRENT_FLOW_STEP` 自己充当累计器。

它应该只承担：

> “当前 forward 正在第几个 flow step”

而 flow step 的**计数状态**应该单独保存。

最小修改可以直接仿照现在已有的 `_generation_counter`。

在：

```python
install_runtime_hooks()
```

里建立一个 model-local counter：

```python
denoise_step_counter = [-1]
```

然后：

```python
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
```

denoise：

```python
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
```

于是每次新的 `sample_actions`：

```text
counter = -1
```

然后十次 denoise：

```text
0
1
2
3
4
5
6
7
8
9
```

下一次 observation 重新执行 `sample_actions`：

```text
counter reset -> -1
```

再重新：

```text
0..9
```

这正是你 Phase H 所需要的语义。

---

## 为什么不推荐“denoise 后不要 reset CURRENT_FLOW_STEP”

也可以写成：

```python
step = CURRENT_FLOW_STEP.get() + 1
CURRENT_FLOW_STEP.set(step)
```

然后不 reset。

它也可能让：

```text
0 → 1 → ... → 9
```

跑起来。

但这样 `CURRENT_FLOW_STEP` 同时承担：

1. 当前 context；
2. persistent loop state。

职责混在一起，以后 async/concurrent inference 很容易出问题。

现在已经有：

```text
CURRENT_PHASE
CURRENT_FLOW_STEP
CURRENT_GENERATION_ID
```

这几个 runtime context，最合理的设计就是：

```text
counter = internal bookkeeping
ContextVar = current observation tag
```

分开。

---

# 强烈建议加 T9

你现在 T4 只验证了：

> “如果手工给 collector 一个 flow_step=0 和 flow_step=1，stat manager 能否分开。”

这个测试已经 PASS，但它测不到 **runtime hook 是否真的生成 0..9**。

所以建议增加：

```text
T9: runtime hook flow-step sequencing
```

最低要求：

```python
observed = []
```

dummy `denoise_step` 中记录：

```python
observed.append(
    get_runtime_context()["flow_step"]
)
```

让 dummy `sample_actions()` 连续调用十次 `self.denoise_step()`。

最后：

```python
assert observed == list(range(10))
```

更完整一点再调用第二次 `sample_actions()`：

```python
assert observed == (
    list(range(10))
    + list(range(10))
)
```

这样以后不会再次把这个 bug 引回来。

这是比 T4 更重要的集成级测试。

---

# outlier accounting 这次没有发现新的 blocker

我也重新顺了一遍。

Linear activation：

```python
protected_channels = x_channel_mask.sum()
repeat = x.numel() // x.shape[-1]
protected_elements = protected_channels * repeat
```

这个计算符合 channel mask 广播语义。Weight runtime mask 直接对 activation-channel mask 与 weight-own mask 的 union 求和，也是正确的。

MatMul A 的 channel mask 也按 token/head 等前维重复；B mask 已扩展到实际 B shape 后直接 `sum()`，逻辑自洽。

`outlier_partition` 与主 structured sparsity 使用的 key 也一致：

```text
module_id
phase
flow_step
tensor_role
attention_kind
```

所以 native subtraction 能正确 join。

---

# 剩余三个非阻塞项

这些不需要挡住 H0。

### 1. `quant_activation_calls` audit counter 还是会偏大

现在 legacy：

```python
collect_quant_activation()
```

会 +1。

而新的：

```python
collect_quant_tensor()
```

开头也会：

```python
self.quant_activation_calls[module_id] += 1
```

。

因此这个 **legacy audit dict 本身仍可能 double/multi count**。

不过这不会影响：

```text
module_sparsity.csv
outlier_sidepath.csv
unit_sparsity.csv
```

里的 structured counters。

所以属于 P2 cleanup。

建议以后把：

```python
collect_quant_tensor()
```

里的：

```python
self.quant_activation_calls[module_id] += 1
```

删掉，或者另建：

```text
structured_tensor_calls
```

不要两个系统共用一个 audit counter。

---

### 2. `workload.csv` 仍然不要拿来算最终硬件 speedup

我之前的判断仍然不变。

Phase H 正式结果应主要使用：

```text
module_sparsity.csv
weight_sparsity_static.csv
outlier_sidepath.csv
unit_sparsity.csv
quantization_manifest.csv
```

`workload.csv` 目前可以保留用于开发，但不要把其中的 BOP proxy 做最终硬件结论。

Phase I 再把：

$$
S_A,\;S_W/S_B,\;S_O
$$

按 physical operator join 成一行。

---

### 3. H0 脚本最好把 EGL 三个变量都显式固定

当前脚本只有：

```bash
export MUJOCO_GL=egl
```

。

结合你之前已经固定过的环境，建议正式实验脚本明确写：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=2
```

即使 conda 环境本身已经有，也建议脚本中显式记录，增强 reproducibility。

---

# 最终 Gate

当前我会更新成：

| 项目                                 |         状态 |
| ---------------------------------- | ---------: |
| static weight collection           |          ✅ |
| outlier native accounting          |          ✅ |
| double structured collection       |       ✅ 已修 |
| structured key                     |          ✅ |
| unit structured                    |          ✅ |
| native upper bound                 |          ✅ |
| S0/S1 config                       |          ✅ |
| H0 orchestration                   |          ✅ |
| **runtime flow-step 0..9 tagging** |   **❌ P0** |
| audit counter cleanup              |     ⚠️ 非阻塞 |
| workload hardware semantics        | ⚠️ Phase I |
| EGL explicit pin                   |      ⚠️ 推荐 |

所以现在实际上非常接近：

$$
\boxed{
\text{只修 flow-step runtime counter}
\rightarrow
\text{跑 tests}
\rightarrow
\text{启动 H0}
}
$$

修完后建议先不要再改其它框架代码，直接执行：

```bash
python -m pytest tests/test_sparsity_accounting.py -v
```

确认新增 T9 也 PASS，然后：

```bash
bash \
experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/run_h0_smoke.sh
```

H0 跑完以后，下一轮就不需要再继续静态 code review 了，应该直接审计 **实际生成的 `module_sparsity.csv / outlier_sidepath.csv / unit_sparsity.csv / weight_sparsity_static.csv`**。那一步才能最终确认真实 rollout 中 `prefill=-1`、`denoise=0..9`、A/B/O、native counters 是否全部闭环。
