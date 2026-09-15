我重新审计了最新 H2 的 `results.md`、`logs.md`、S0/S1 配置、`build_results_tables.py`、attention 数据流和 `stat_manager.py`。整体上，**H2 的原始统计链路已经稳定，output/O bug 也被 H2 正式闭环；H2 可以判 PASS，并可以进入 H3。** 但当前结果文档里还有几处会影响论文结论的分析口径问题，其中两处比较重要：**S0/S1 的 weight sparsity 不是同 scope 比较**，以及 **PV A 的物理含义解释写错了**。

| 审计项                              | 判定               | 重要性        |
| -------------------------------- | ---------------- | ---------- |
| H2 20/20 tasks 完成                | ✅                | —          |
| S0=90.0%，S1=86.7%                | ✅                | —          |
| H1→H2 sparsity 收敛                | ✅                | —          |
| output/O 修复后约 39.9%              | ✅                | —          |
| flow-step 0–9                    | ✅                | —          |
| numerator/denominator 聚合         | ✅                | —          |
| S0/S1 runtime headline 直接比较      | ⚠️ scope 不完全一致   | P1         |
| S0 41.52% vs S1 73.72% weight 比较 | ❌ scope 不一致      | **P1，必须修** |
| “PV A 是 value/RoPE 激活”           | ❌                | **P1，必须修** |
| unit sparsity 称作 native          | ⚠️ 不准确           | P1         |
| workload/BOP 66.8/67.9%          | ⚠️ 仅 debug proxy | P1         |
| “与 SR 无相关”                       | ⚠️ 结论过强          | P2         |
| 原 H0/H1 INVALID “解除”             | ⚠️ 表述不严谨         | P2         |

## 1. H2 本身是成功的

远端日志确认 H2 已完整完成：S0 与 S1 都是 10 tasks × 3 episodes，共 30 episodes，20/20 task pipeline 无错误；结果为 S0 90.0%、S1 86.7%。

从 episode 数看实际上就是：

$$
S0=\frac{27}{30}=90.0\%
$$

$$
S1=\frac{26}{30}=86.7\%
$$

因此目前二者只差 **1 个成功 episode**。

我按 30 episodes 计算 Wilson 95% CI，大约是：

$$
S0:\;74.4\%-96.5\%
$$

$$
S1:\;70.3\%-94.7\%
$$

重叠非常大。因此当前最合适的表述是：

> H2 与 Phase G 的 88%/84% 结果相容，S1 Expert-W4 在 30ep 下仍表现出接近 FP8 anchor 的闭环精度；但 30ep 不足以精确确定 3.3pp gap，H3 100ep 仍需要用于正式 accuracy claim。

不要写成：

> H2 已证明 Expert-W4 只损失 3.3pp。

H3 仍然有意义，而且现在它的主要任务确实已经从“统计 sparsity”变成了“确认 SR”。

---

## 2. Sparsity 收敛这部分是可信的

你现在使用了正确的 numerator/denominator 聚合，而不是平均 percentage。`build_results_tables.py` 的 `_ratio()` 明确是先 sum numerator / denominator。

排除旧的 pre-fix output/O 后：

$$
S0:\quad43.60\%\rightarrow43.59\%
$$

$$
S1:\quad44.01\%\rightarrow44.03\%
$$

逐 component 最大变化只有：

$$
0.14\text{ pp}
$$

出现在 S1 Expert.PV A。Unit 最大变化也低于 0.5pp。

所以：

$$
\boxed{
\text{sparsity characterization 在约 10ep 时已经非常稳定}
}
$$

这个结论我认为可以保留。

但有一个细微点：H1 和 H2 都使用 `seed=1000`，evaluation 最终把这个 seed 作为 `start_seed` 传给 rollout；H2 只是把每 task 从 1 episode 扩成了 3 episodes。也就是说 H2 很可能包含 H1 的第一条 episode，而不是独立的一批 30 episodes。

所以更准确的说法应是：

> 从每 task 1ep 扩展到 3ep 后，sparsity estimate 几乎不变。

而不是：

> 两组独立实验重复证明了收敛。

这不影响结论，只影响措辞。

---

# 3. 第一处必须修：当前 weight 41.52% → 73.72% 不是 apples-to-apples

这是现在 `results.md` 最大的方法学问题。

S0 配置：

```yaml
linear:
  include:
    - '*'
```

所以 S0 static weight 包括：

```text
VLM Linear FP8
+
Expert Linear FP8
```

。

而 S1：

```yaml
linear:
  include:
    - expert.*
```

所以 static weight 只有：

```text
Expert Linear W4
```

VLM Linear 根本没有进入 QuantizedLinear/static weight CSV。

这也解释了当前结果：

```text
S0 = 224 QuantizedLinear
S1 = 112 QuantizedLinear
```

因此当前：

$$
41.52\%\quad vs\quad73.72\%
$$

实际上比较的是：

$$
\boxed{
\text{VLM FP8 + Expert FP8}
}
$$

vs

$$
\boxed{
\text{Expert INT4}
}
$$

不是：

$$
\text{Expert FP8}\quad vs\quad\text{Expert INT4}.
$$

所以当前文档中的：

> “W4 使 weight sparse_bit_rate 从 41.52% 提升到 73.72%，+32.2pp”

**严格来说还没有被当前数据直接证明。** 

### 应该怎么修

`build_results_tables.py` 的 weight summary 增加一个 **Expert-common-scope** 表。

S0：

```python
s0_expert = [
    r for r in s0_weight_rows
    if r["component"].lower() == "expert"
]
```

S1：

```python
s1_expert = [
    r for r in s1_weight_rows
    if r["component"].lower() == "expert"
]
```

然后比较：

$$
S_{\rm W,Expert}^{FP8}
$$

与：

$$
S_{\rm W,Expert}^{INT4}.
$$

最终主表应该变成：

| Weight scope            |       S0 |       S1 |
| ----------------------- | -------: | -------: |
| All quantized Linear    |   41.52% |   73.72% |
| **Expert common scope** | **重新计算** | **重新计算** |

论文里的 W4 增益必须引用第二行。

我预计 Expert FP8 会和 41%左右比较接近，但**不要猜，重新从 CSV numerator/denominator 算**。

---

# 4. static weight 还重复统计了 10 次

当前结果自己已经显示：

```text
S0 rows = 2240 = 10 × 224
S1 rows = 1120 = 10 × 112
```

。

这是因为每个 task output 都重新 export 一份相同 static weights，汇总脚本又把 10 个 task 全加了起来。

由于每 task 权重完全相同：

$$
\frac{10N_{\rm sparse}}{10N_{\rm total}}
=
\frac{N_{\rm sparse}}{N_{\rm total}}
$$

所以**当前 41.52%/73.72% 的比例没有错**。

但是：

```text
total_bits
parameter count
weight BOP
memory volume
```

如果以后直接用这些加总，就会被放大 10 倍。

建议 weight summary 在读取后按：

```python
(config, module_id)
```

deduplicate，只保留一个 task 的 static weight row。

甚至更简单：

```python
_load_one_task(
    stage,
    cfg,
    "task00",
    "weight_sparsity_static.csv"
)
```

static weight 本来就是 episode-independent，不应该经过 task aggregation。

---

# 5. 第二处必须修：PV 的 A 不是 value activation

当前 `results.md` 写：

> PV 的 A 矩阵高稀疏是“RoPE/位置编码后 value 激活的固有结构”。

这个解释是错误的。

当前 attention 代码非常清楚：

```python
probs = softmax(masked_att_weights, dim=-1)

att_output = pv_matmul(
    probs,
    value_states.permute(...)
)
```

。

所以：

$$
PV:
\quad
A=P=\operatorname{softmax}(QK^T+\text{mask})
$$

$$
B=V
$$

因此当前：

$$
S_{\rm PV,A}
=
54\%-61\%
$$

对应的是：

$$
\boxed{\text{attention probability matrix }P}
$$

不是 value state。

这实际上比原解释更有研究价值。

Expert PV A：

$$
31.53\%\text{ element zeros}
$$

$$
60.99\%\text{ significand bit sparsity}
$$

而 VLM PV A：

$$
18.42\%\text{ element zeros}
$$

$$
54.26\%\text{ bit sparsity}
$$

。

正确解释应该改成：

> PV A 对应 softmax 后的 attention probability。其高 sparsity 与 attention probability 的长尾/集中分布、attention mask 以及 FP8 quantization 共同相关。

尤其不要再写：

> “不是量化产物”。

因为这里统计的是 **post-quant code**。小的非零 probability 很可能被 FP8 scale/quantization 映射为 0。

如果你要区分：

$$
\text{softmax 原生 zero/small-value structure}
$$

和：

$$
\text{quantization-induced zeros},
$$

需要额外收集 pre-quant `probs`。

---

# 6. PV A 的 flow-step 变化反而值得继续研究

这是 H2 中我认为最值得保留的新发现。

绝大多数 role：

$$
\Delta S < 0.02\text{ pp}
$$

几乎完全不随 denoise step 变化。

只有 PV A：

S0：

$$
61.02
\rightarrow
61.38
\rightarrow
59.85\%
$$

S1：

$$
60.29
\rightarrow
60.81
\rightarrow
59.41\%
$$

呈明显的倒 U 型，约 1.4–1.5pp range。

现在知道：

$$
PV_A=P=\text{attention probability}
$$

后，这个趋势应该理解成：

> **Action Expert 在 flow matching 不同 denoise step 上 attention distribution 的集中程度发生了变化。**

这比“activation bits 随 step 改变”更准确。

Phase I 或 appendix 我建议再统计：

$$
H(P)=-\sum_jP_j\log P_j
$$

也就是 attention entropy，以及：

```text
fraction(P < FP8 quantization threshold)
post-quant zero ratio
top-1/top-k attention mass
```

然后看它们是否和 PV-A 61→59% 的趋势同步。

这可能形成一个非常不错的小结论：

$$
\boxed{
\text{flow step 对大多数 GEMM operand sparsity 几乎无影响，
但会改变 attention-probability sparsity}
}
$$

---

# 7. Unit sparsity 的“native”命名还不准确

当前 bit sparsity 做了：

```text
reported
-
protected outlier positions
=
native
```

这一部分正确。

但 unit sparsity 没有做 native correction。

代码是直接把进入 normal quant path 的 `tensor` 送到：

```python
collect_unit_sparsity_structured(...)
```

计算 2×2 unit zero。

而 outlier forward 中 protected positions 已经在 normal tensor 里被 mask 为 0。

CSV exporter 只是：

$$
\text{unit\_zero\_rate}
=
\frac{N_{\rm zero-unit}}{N_{\rm total-unit}}
$$

没有和 `outlier_partition` 做任何 join/correction。

所以 H2 的：

```text
S0 unit = 7.36%
S1 unit = 7.99%
```

准确含义是：

$$
\boxed{
\text{masked normal quant-path 2×2 unit sparsity}
}
$$

而不是：

$$
\text{native unit sparsity after excluding protected elements}.
$$

结果表现在标题写：

> native = 扣除 FP protected 后

然后把 Unit sparsity 放在旁边，容易让人以为 unit 也经过 native correction。

建议改列名为：

```text
Quant-path unit sparsity (2×2, masked)
```

或者加脚注：

> Unit sparsity 未做 native outlier correction；它描述实际 normal quant datapath 中的 block-zero opportunity。

其实从硬件角度，这个指标仍然很有价值。

---

# 8. 当前 42.08% vs 42.42% 也不是完全同 scope

这一点比 weight scope 问题稍轻，但仍应说明。

S0 runtime 包括：

```text
VLM Linear
Expert Linear
VLM QK/PV
Expert QK/PV
```

S1 runtime 则是：

```text
Expert Linear
VLM QK/PV
Expert QK/PV
```

因为 S1 VLM Linear 是 raw FP，不会进入 quantized sparsity collector。结果文档自己也正确注明了 S1 无 VLM linear 行。

所以：

$$
42.08\%\quad vs\quad42.42\%
$$

是：

> 各自 **quantized workload** 内部的 aggregate sparsity。

不是：

> 同一 workload scope 下 S0/S1 sparsity 的严格差值。

因此“Δ=0.34pp”不能直接解释成“W4 只让 runtime sparsity 改变 0.34pp”。

建议增加：

### Common quantized runtime scope

统一只保留两边都有的：

```text
Expert Linear
VLM QK/PV
Expert QK/PV
```

S0 删除：

```text
VLM Linear
```

之后再比较：

$$
S0_{\rm common}
\quad vs\quad
S1_{\rm common}.
$$

这个数字才适合写：

> 配置变化对 runtime code sparsity 的影响。

---

# 9. BOP proxy 目前不能用于正式硬件结论

这一点文档已经有所警告，但实际问题比脚注写得更大。

当前 exporter 对**每个 role**单独生成 workload row：

```text
activation
output
A
B
O
```

然后取当前 role 的：

```python
bit_sparsity = entry["sparse_bit_rate"]
```

再同时用于：

```python
b_a = A_bits * (1-S)
b_b = B_bits * (1-S)
```

最后：

$$
BOP_{\rm active}
=
MAC\times b_a\times b_b.
$$

代码就是这样。

这有三个问题。

第一，Linear 应该是：

$$
MAC
\cdot
b_A(1-S_A)
\cdot
b_W(1-S_W)
$$

而不是同一个 \(S\) 同时用于 A/W。

第二，MatMul 应该用：

$$
S_A,\;S_B
$$

分别缩放，而不是 A row 用 \(S_A\) 同时缩两边、B row 又生成一次完整 operator MAC。

第三，目前这里取的是：

```python
entry["sparse_bit_rate"]
```

也就是 reported sparsity，而不是 Phase H headline 使用的 native corrected sparsity。

因此当前：

```text
S0 66.8%
S1 67.9%
```

**不要作为论文结果或 hardware reduction headline。**

建议把这一节标题直接改成：

> Legacy/debug BOP proxy — not physically additive

Phase I 重写为 **one physical operator, one row**：

$$
(operator,phase,step)
\rightarrow
(S_A,S_{W/B},S_O,M,K,N,R_{FP})
$$

再计算真实的 BOP opportunity。

---

# 10. “sparsity 与 task 无相关”结论应弱化

文档目前写：

> 相关性实质为 0。



但当前 `build_results_tables.py` 实际没有计算 Pearson/Spearman，它只是打印 task 表。

我按照当前表里的 20 个点重新算了一下 Pearson：

$$
r_{\rm S0}\approx0.003
$$

$$
r_{\rm S1}\approx0.138
$$

合并：

$$
r_{\rm pooled}\approx-0.053.
$$

所以“pooled 基本为 0”数值上确实成立。

但样本只有：

$$
20\text{ points}
$$

而每 task 只有 3 episodes，SR 只能取：

```text
0, 33.3, 66.7, 100%
```

所以科研表述应是：

> H2 pilot 中未观察到明显的 per-task sparsity–SR association；pooled Pearson \(r\approx-0.05\)。由于每 task 仅 3 episodes，不能据此证明两者统计独立。

同样：

> task06/task07 是“task 固有难度”

也太强。

它们两个 config 同时掉分，更严谨应写：

> 与共享 task difficulty 或相同 seed 下的 episode sampling effect 一致，暂不能归因于量化配置。

---

# 11. “H1/H0 INVALID 标注正式解除”也建议改一句

原 H0/H1 CSV 中的 output/O：

$$
0.3\%-0.6\%
$$

永远还是 invalid。

H2 只是证明：

> bug 已闭环，post-fix replacement measurement 有效。

所以：

```text
H1/H0 的 output/O INVALID 标注正式解除
```

容易让人误以为旧 CSV 现在变有效了。

建议改成：

> H1/H0 pre-fix output/O 原始测量仍标记 INVALID；该异常已通过 H1-Audit 与 H2 post-fix aggregate 完成闭环，正式结果由 post-fix 数据替代。

---

# 12. 我建议现在修改 `build_results_tables.py`

不需要重跑 H2，原始 H2 数据足够。

建议补三个 summary：

```python
# 1. common runtime scope
def _common_runtime_scope(rows, cfg):
    if cfg == "s0_fp8_all":
        rows = [
            r for r in rows
            if not (
                r["component"].lower() == "vlm"
                and r["operator"] in LINEAR_OPS
            )
        ]
    return rows
```

然后输出：

```text
Common-scope native significand sparsity:
S0 = ?
S1 = ?
```

第二个：

```python
# 2. static weight must be deduplicated + expert-only
def _expert_unique_weights(rows):
    unique = {}
    for r in rows:
        if r["component"].lower() != "expert":
            continue
        unique[r["module_id"]] = r
    return list(unique.values())
```

得到真正：

```text
Expert FP8 weight sparsity = ?
Expert W4 weight sparsity  = ?
Δ                         = ?
```

第三个是真实 correlation：

```python
pearson_r
spearman_r
```

然后 `results.md` 引用实际值，不再写定性“实质为0”。

---

# 13. H3 是否值得跑？

**值得，而且现在应该跑。**

但它的目标已经非常明确：

$$
\boxed{\text{H3 = accuracy confirmation，不再是 sparsity convergence experiment}}
$$

H2 已经证明 sparsity estimate 非常稳定，所以 H3 没必要为了统计精度再保存一堆昂贵的 audit histogram。

H3 重点确认：

$$
S0_{\rm Goal,100ep}
$$

$$
S1_{\rm Goal,100ep}
$$

以及尤其：

```text
task03
task06
task07
```

当前 30ep 里 S0/S1 只差一个 episode，因此 100ep 很必要。

如果你想节约计算/统计开销，我会建议 H3：

```text
保留：
- module sparsity counters（可选）
- flow-step tagging
- eval SR

关闭：
- fp_code_audit（已经闭环）
```

unit sparsity也可以保留，代价可接受的话用于最终 sanity check，但已经没有必要靠 100ep 提高其估计精度。

---

## 最终 Gate

我目前会给 Phase H：

$$
\boxed{\text{H2 PIPELINE = PASS}}
$$

$$
\boxed{\text{H2 SPARSITY CONVERGENCE = PASS}}
$$

$$
\boxed{\text{OUTPUT/O BUG = CLOSED}}
$$

$$
\boxed{\text{H3 READY = YES}}
$$

但在把 H2 结果用于组会/论文之前，我建议**先修四个结果解释问题**：Expert-only common-scope weight comparison、common-scope runtime aggregate、PV A=attention probabilities、unit sparsity改成 masked quant-path 语义。尤其前两个会直接改变你现在“W4 带来 +32.2pp weight sparsity、runtime 只差 0.34pp”这两句最核心的量化比较，因此应在 H3 前把汇总脚本补齐。
