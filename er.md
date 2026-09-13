我重新拉了远端 `main` 的 raw 文件检查。这一版相比上次已经修好了绝大多数问题，但又发现了一个**新的 P0 级重复统计问题**。所以当前状态是：

> **Phase H 框架已经接近可运行，但我仍不建议马上启动正式 H0。先修掉“activation/A/B 被统计两次”这一处，否则 `native sparsity` 会算错。**

## 1. 上次指出的问题，基本都已经修好

这次这些项可以判定 PASS：

| 项目                          | 状态 | 说明                                                 |
| --------------------------- | -- | -------------------------------------------------- |
| `export.dir: null` fallback | ✅  | 已用 `or os.path.join(...)` 修复                       |
| `chunk_size` 接入             | ✅  | `sm.enable_sparsity(... chunk_size=...)`           |
| unit YAML key               | ✅  | 已兼容 `bit_group_size/dim_group_size` 和旧 `rows/cols` |
| static weight collection    | ✅  | rollout 后调用 `collect_model_weight_sparsity(model)` |
| static weight 空结果保护         | ✅  | `n_weight_layers <= 0` 直接报错                        |
| flow-step 拆分                | ✅  | key 已包含 `phase/flow_step/role/attention_kind`      |
| structured unit sparsity    | ✅  | 已建立同粒度 `per_role_unit_sparsity`                    |
| outlier partition           | ✅  | Linear A/W/O、MatMul A/B/O 都开始记录 protected elements |
| reported/native CSV         | ✅  | `module_sparsity.csv` 已同时输出两套指标                    |
| weight CSV 命名               | ✅  | 改为 `weight_sparsity_static.csv`                    |
| task generator deepcopy     | ✅  | 已换成 `copy.deepcopy(base)`                          |
| S0/S1 配置                    | ✅  | 仍严格复用 G1-A/G1-D                                    |
| quantization manifest       | ✅  | 已增加部署/精度配置 manifest                                |

`main.py` 现在确实会在 rollout 后收集 static weight，并导出 `quantization_manifest.csv / module_sparsity.csv / workload.csv / weight_sparsity_static.csv / outlier_sidepath.csv / unit_sparsity.csv`；`dir: null` 也已经正确 fallback。

Flow step 这一块也是真的修了，而不只是传了参数。现在 structured key 已经是：

```python
(
    layer_name,
    phase,
    int(flow_step),
    tensor_role,
    attention_kind or "unknown",
)
```

所以 `denoise step 0..9` 终于不会混在一起。

Unit sparsity 也已经用同样的五维 key 做独立累计，可以真正区分 A/B/O、不同 flow step 和 self/cross attention。

Outlier accounting 的主体也已经实现：Linear 会记录 activation、runtime weight mask、output；MatMul 会记录 A/B/O 的 protected elements。

---

# 2. 现在剩下的主要 P0：同一个 activation/A/B 被统计了两遍

这是这轮检查里最重要的新发现。

当前 `collect_quant_activation()` 已经不再只是 audit counter，它会继续把 tensor 转发给：

```python
self.collect_quant_tensor(...)
```

Linear 会统计：

```python
("activation", x_code)
```

MatMul 会统计：

```python
("A", A_sim)
("B", B_sim)
```

。

但 `quant_methods.py` 在调用完 `collect_quant_activation()` 后，**又立即调用了一次新的 structured collector**。

例如 Linear：

```python
stat_collector.collect_quant_activation(
    ...
    x_sim,
    ...
)

_collect_linear_runtime(
    layer,
    stat_collector,
    input_code=x_sim,
    input_spec=layer.a_spec,
)
```



非 outlier 的普通 per-tensor Linear 也是同样结构：

```python
collect_quant_activation(...)
_collect_linear_runtime(...)
```



MatMul 同样如此：

```python
collect_quant_activation(...)
_collect_matmul_runtime(
    A_code=A_sim,
    B_code=B_sim,
)
```



也就是说，一次真实 forward：

```text
Linear activation
```

会进入 structured sparsity accumulator 两次；

一次 MatMul：

```text
A
B
```

也各进去两次。

---

# 3. 为什么这个 bug 很严重

如果没有 outlier，比例有时看起来还“正常”。

假设真实 activation：

```text
100 elements
20 zeros
```

被收集两遍以后：

```text
200 elements
40 zeros
```

ratio 还是：

$$
40/200=20\%
$$

所以如果你只看 `zero_rate_reported`，很容易以为没问题。

但是 Phase H 最重要的是 **native sparsity correction**。

Outlier partition 每次 forward 只记录一次。

例如真实：

```text
total       = 100
reported 0  = 30
protected   = 10
```

正确 native 应该是：

$$
N_{\text{native}}=100-10=90
$$

$$
N_{\text{zero,native}}=30-10=20
$$

所以：

$$
Z_{\text{native}}
=
20/90
=
22.22\%
$$

但 structured sparsity 被统计两遍后：

```text
total       = 200
reported 0  = 60
protected   = 10   ← partition 没重复
```

代码会得到：

$$
N_{\text{native}}=190
$$

$$
N_{\text{zero,native}}=50
$$

$$
Z_{\text{native}}
=
50/190
=
26.32\%
$$

**直接错了。**

而你当前 export 正是：

```python
total_elements_native = total_elements - protected_elements
zero_elements_native = zero_elements - protected_elements
```

。

因此这是 Phase H 的 **P0 correctness blocker**。

此外还会造成：

```text
calls ×2
total_elements ×2
total_bits ×2
unit total_units ×2
```

所以即使 reported ratio 没变，raw counters、unit workload 和后续 hardware workload 都会错。

---

# 4. 推荐怎么修：保留新的 collector，废掉 legacy structured forwarding

我建议不要删：

```python
_collect_linear_runtime()
_collect_matmul_runtime()
```

因为这两个新接口更清晰，而且已经统一承担：

```text
module_id
tensor_role
phase
flow_step
attention_kind
metadata
```

应该保留。

反过来，让旧的：

```python
collect_quant_activation()
```

只承担它原来最有价值的用途：

> legacy/audit call counting。

也就是改成类似：

```python
def collect_quant_activation(
    self,
    layer_name,
    layer_idx,
    *args,
    **kwargs,
):
    key = f"{layer_name}_{layer_idx}"

    self.quant_activation_calls[key] = (
        self.quant_activation_calls.get(key, 0) + 1
    )

    # Legacy audit only.
    # Structured sparsity is collected explicitly through
    # collect_quant_tensor() by _collect_linear_runtime /
    # _collect_matmul_runtime.
    return
```

也就是说，把现在 `stat_manager.py` 里大约 373–411 行这段 compatibility forwarding 删除：

```python
n_extra = len(args)
...
self.collect_quant_tensor(...)
```

。

### 为什么这种方案比删 `_collect_*_runtime()` 好

因为 output 已经只通过：

```python
_collect_linear_runtime(output_code=...)
_collect_matmul_runtime(O_code=...)
```

来收集。

所以统一成：

```text
collect_quant_activation
    ↓
只 audit 调用次数

_collect_linear_runtime
    ↓
activation / output structured sparsity

_collect_matmul_runtime
    ↓
A / B / O structured sparsity
```

数据路径最干净。

---

# 5. 建议顺手加一个专门防止这个问题的测试

你现在测试已经不错：

* T1 reported==native；
* T2 outlier correction；
* T3 numerator/denominator 加权；
* T4 flow-step separation；
* T5 A/B/O unit separation；
* T6 collector 不修改输入；
* T7 manifest/workload export。

但是它们都是直接调用：

```python
collect_quant_tensor()
```

所以抓不到“legacy collector + new collector 同时调用”的问题。

建议增加 **T8**：

```python
def test_t8_no_double_collection_between_legacy_and_structured():
    m = QuantStatManager(...)
    m.enable_sparsity()

    spec = QuantSpec(kind="int", bits=4)

    x = torch.tensor([1., 0., 1., 1.])

    # legacy audit call
    m.collect_quant_activation(
        "expert.layers.0.mlp.up_proj",
        0,
        x,
        x,
        spec,
        4,
        1,
        4,
        4,
    )

    # actual structured collection
    m.collect_quant_tensor(
        module_id="expert.layers.0.mlp.up_proj",
        tensor_role="activation",
        tensor_code=x,
        spec=spec,
        runtime_context=_ctx(),
    )

    rows = ...
    row = ...

    assert int(row["calls"]) == 1
    assert int(row["total_elements_reported"]) == 4
```

修改后这个测试才能通过。

更进一步，outlier 可以验证：

```text
structured calls == partition calls
```

对于同一个：

```text
module_id / phase / flow_step / role / attention_kind
```

正常应该是：

$$
N_{\text{structured calls}}
=
N_{\text{partition calls}}
$$

至少对启用 outlier 的 A/B/O 路径成立。

这个 invariant 很有价值。

---

# 6. 另一个需要修的小问题：`ideal_sparse_upper_bound` 现在仍然基于 reported sparsity

你已经改了字段名称，这是好的。

但是 `_accumulate_sparsity_entry()` 现在计算：

```python
upper_bound = 1 / (1 - entry["sparse_bit_rate"])
entry["ideal_sparse_upper_bound"] = upper_bound
```

这里的：

```text
sparse_bit_rate
```

是 **reported sparsity**。

而 export 时：

```python
upper_bound = entry.get(
    "ideal_sparse_upper_bound",
    1 / (1 - sparse_bit_rate_native)
)
```

。

问题是：

```python
entry["ideal_sparse_upper_bound"]
```

几乎总是已经存在。

所以 fallback：

```python
1 / (1 - sparse_bit_rate_native)
```

实际上不会被用到。

因此 `module_sparsity.csv` 中最后那列虽然名叫：

```text
ideal_sparse_upper_bound
```

却仍然对应：

$$
S_{\text{reported}}
$$

而不是 Phase H 更应使用的：

$$
S_{\text{native}}
$$

### 直接改 export 即可

不要 `entry.get()`：

```python
upper_bound = (
    1.0 / (1.0 - sparse_bit_rate_native)
    if sparse_bit_rate_native < 1.0
    else float("inf")
)
```

Legacy per-layer summary 保留 reported upper bound 没问题，但 `module_sparsity.csv` 应明确使用 native。

这是 **P1，但建议跟 P0 一起修**。

---

# 7. `workload.csv` 暂时不要拿来做硬件 speedup

这个文件现在可以作为开发输出，但我建议 Phase H **暂时不要把其中的 BOP/MAC proxy 当正式结果**。

现在 exporter 对每一个 tensor role 都单独构造一个 workload row，然后：

```python
bit_sparsity = entry.get("sparse_bit_rate", 0.0)

b_a = a_bits * (1 - bit_sparsity)
b_b = b_bits * (1 - bit_sparsity)

bop_active = macs * b_a * b_b
```

。

这里有几个概念问题：

对于 Linear，A sparsity 与 W sparsity应该分别是：

$$
S_A,\quad S_W
$$

而不是用同一个 tensor-role 的：

$$
S
$$

同时修正 A 和 B。

真正更接近的是：

$$
BOP_{\text{active}}
\propto
MAC
\cdot
b_A(1-S_A)
\cdot
b_W(1-S_W)
$$

MatMul 也应该分别使用：

$$
S_A,\quad S_B
$$

而不是 A role 和 B role 各自生成一份完整 MAC workload。

现在每个 A/B/O role 都生成一行 MAC，因此如果后续把 workload rows 直接相加，还会**重复计算同一个算子的 MAC**。

所以本 Phase H 建议正式使用：

```text
module_sparsity.csv
weight_sparsity_static.csv
outlier_sidepath.csv
unit_sparsity.csv
quantization_manifest.csv
```

而：

```text
workload.csv
```

暂时标：

> experimental / not for hardware-speedup aggregation.

Phase I 再重新设计 workload join：

```text
Operator
├── A sparsity
├── B/W sparsity
├── O sparsity
├── M/K/N
├── call count
└── FP sidepath
```

每个 physical operator 一行，而不是每个 role 一行。

---

# 8. S0/S1 配置现在是对的

我重新检查了这两个 config。

S0 保持：

```text
G1-A
FP8 Linear
FP8 QK/PV
outlier_ratio=0.01
n_action_steps=10
num_steps=10
原 scale_dir
```

且只增加 sparsity 配置。

S1 保持：

```text
VLM raw
Expert Linear W4
A/O E4M3
QK/PV E4M3
outlier_ratio=0.01
n_action_steps=10
num_steps=10
原 G1-D scale_dir
```

。

`make_task_configs.py` 也已经使用 `deepcopy`，H0/H1/H2/H3 分别对应：

```text
1×task0
1×10 tasks
3×10 tasks
10×10 tasks
```

并只修改：

```text
output_dir
task_ids
n_episodes
```

。

这一部分不用再改。

---

# 9. 还有两个非阻塞项

### config 里的几个 flag 目前基本是 declarative

你写了：

```yaml
outlier_accounting:
  enabled: true
  export_reported: true
  export_native: true

structured:
  split_phase: true
  split_flow_step: true
  split_tensor_role: true
  split_attention_kind: true
```

但当前 `main.py` 并没有真正读取这些开关；实际行为是只要 `sparsity.enabled=true`，这些功能就默认全开。

对当前 H0/H1 没问题，因为你本来就全部设 true。

但长期最好二选一：

```text
真正 wire 到代码
```

或者：

```text
删掉这些“看似可配置但实际无效”的选项
```

避免以后设置 false 却仍然统计。

### T6 仍然比较弱

现在 T6 只是：

```text
collector 不修改输入 tensor
```

。

原本更强的要求是：

```text
same quantized layer
same input
same scales

stats OFF → output0
stats ON  → output1

assert output0 == output1
```

建议 H0 前或 H0 后补一个真实 QuantizedLinear forward test。

---

# 10. 当前最终 Gate

目前我会这样定：

| 项目                                      |            状态 |
| --------------------------------------- | ------------: |
| main wiring                             |             ✅ |
| static weight                           |             ✅ |
| flow-step structured                    |             ✅ |
| outlier partition                       |             ✅ |
| native export formula                   |             ✅ |
| structured unit                         |             ✅ |
| S0/S1 config                            |             ✅ |
| task generator                          |             ✅ |
| **structured tensor double collection** |      **❌ P0** |
| native upper-bound semantic             |         ⚠️ P1 |
| workload BOP semantics                  | ⚠️ Phase I 前修 |
| true stats ON/OFF equivalence test      |        ⚠️ 建议补 |

因此：

$$
\boxed{\text{H0 READY = NO，目前只差一个真正的 P0}}
$$

最重要的修改就是：

> **让 `collect_quant_activation()` 恢复成 audit-only，structured sparsity 只由 `_collect_linear_runtime()` / `_collect_matmul_runtime()` 收集一次。**

修完以后再把：

```python
module_sparsity.csv
```

的 upper bound 改成基于 `sparse_bit_rate_native`，然后跑：

```bash
python -m pytest tests/test_sparsity_accounting.py -v

bash experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/run_h0_smoke.sh
```

如果 H0 输出里每个 outlier A/B/O 的 `calls` 与对应 partition calls 能一一匹配、denoise `flow_step=0..9` 齐全、static weight 非空、native counter 全部非负，我就会认为 **Phase H 正式可以进入 H1 10-episode pilot**。
