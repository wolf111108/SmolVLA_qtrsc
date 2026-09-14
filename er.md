有问题，而且我认为当前 `results.md` 里“runtime activation/output 的 native bit sparsity ≈ 0% 是符合预期”的解释**不能成立**。

我重新检查了最新远端 H0 结果、`stat_manager.py`、`quant_methods.py`、`quant_spec.py` 和汇总脚本。H0 的结构完整性确实很好：S0/S1 都成功、flow step 0–9 齐全、A/B/O 齐全、sidepath 约 1.3%、static weight 行数正确；但结果文档随后把 runtime ≈0% 解释成“E4M3 非零 code 的 mantissa 几乎无全零 bit”，这一解释和实际统计代码是矛盾的。

## 1. 首先：按你现在的统计定义，FP8 activation 的 bit sparsity 不应该天然接近 0%

当前 `compute_sparse_stats_fp()` 对 E4M3 **不是统计完整 8-bit `S EEEE MMM`**，而是统计：

$$
1_{\text{hidden}} + 3_{\text{mantissa}}
$$

总共 **4 bit significand**。代码明确设置 `n = mant_bits + 1`，E4M3 因此就是 4 bit，然后逐个 bit 累加 zero-bit 数。

因此它统计的是：

$$
S_{\rm sig}
=
\frac{\text{4-bit significand 中的 0 bit 数}}
{4N}
$$

而不是“mantissa 是否整个为 000”。

举几个最简单的 E4M3 正常数：

| E4M3 数值 | 当前统计的 significand | zero-bit sparsity |
| ------: | ----------------: | ----------------: |
|     1.0 |            `1000` |               75% |
|   1.125 |            `1001` |               50% |
|    1.25 |            `1010` |               50% |
|     1.5 |            `1100` |               50% |
|    1.75 |            `1110` |               25% |
|   1.875 |            `1111` |                0% |

所以只有 significand 恰好接近：

```text
1111
```

时才是 0%。

要让几千万甚至几亿个 runtime activation/output 聚合后趋近 0%，相当于说这些 FP8 数值的 significand 几乎全部都是 `1.111`。对于 VLM + Action Expert 所有 Linear/MatMul 的 activation/output，这极不合理。

更强的内部 sanity check 是：H0 的 **S0 FP8 static weight 用的是同一个 `compute_sparse_stats_fp()`，结果却是 41.52%**。

因此：

$$
\boxed{\text{“E4M3 本身导致 runtime bit sparsity ≈0” 基本可以排除}}
$$

---

# 2. 当前 FP bit extractor 确实存在一个代码错误

当前代码先拆：

```python
sign = (raw >> sign_shift) & 0x1
mant = raw & ((1 << mant_bits) - 1)
exp  = (raw >> mant_bits) & ((1 << exp_bits) - 1)

sm = (sign << mant_bits) | mant
nonzero = (exp != 0) | (mant != 0)
```

然后：

```python
hidden = nonzero << mant_bits
sm_full = sm | hidden
```

但注释明明写的是：

```text
Add hidden leading 1 for normal numbers (exp != 0)
```

实际条件却用了：

```python
(exp != 0) | (mant != 0)
```

。

这至少造成两个错误。

### 错误 A：subnormal 被错误加 hidden 1

E4M3 最小正 subnormal：

```text
raw = 0000 001
```

应该统计为：

```text
0001
```

当前代码却变成：

```text
1001
```

因为虽然 `exp=0`，但 `mant != 0`，于是错误加入 hidden 1。

### 错误 B：`-0` 不会被统计成全零

对于：

```text
+0 -> 0x00
-0 -> 0x80
```

当前逻辑得到：

```text
+0 -> 0000
-0 -> 1000
```

这对你现在的 outlier-native correction 尤其危险。

---

# 3. `-0` 问题会直接破坏当前 native bit sparsity 的扣除公式

你的 outlier 路径现在确实是：

```python
x_normal_fp = x * (~x_channel_mask)
```

protected 的 activation 被 normal path 人工乘成 0，然后再进行 FP8 quantization。

如果原值是负数，例如：

```text
-2.5 × 0
```

IEEE 浮点很可能得到：

```text
-0.0
```

而当前 extractor：

```text
-0.0 -> 1000
```

因此这个人工 zero 实际在当前 4-bit statistic 中只有：

$$
3
$$

个 zero bits，不是 4 个。

但是 export 时直接假设：

```python
protected_bits = protected_elements * bitwidth

sparse_bits_native
    = sparse_bits_reported - protected_bits
```

即每个 protected E4M3 元素都被认为贡献了 **4 个人工 zero bits**。

所以对于负 protected value，实际：

$$
3
$$

却扣：

$$
4
$$

native numerator 被过度扣减。

这意味着：

$$
\boxed{\text{当前 FP sparse\_bit\_rate\_native 数值本身就不严格正确}}
$$

不过要注意：H0 总 sidepath 只有大约 1.3%。所以**仅仅这个 `-0` bug 通常不足以把一个约 40% 的 reported sparsity 一路打成 0%**。

这说明还必须继续检查 H0 CSV 中：

$$
S_{\rm reported}
$$

本身到底是多少。

---

# 4. 现在最关键的诊断：先区分 reported≈0，还是只有 native≈0

当前汇总脚本只读取：

```python
sparse_bits_native
total_bits_native
```

所以 `results.md` 现在只告诉我们：

```text
native ≈ 0
```

却没有告诉我们：

```text
reported 到底是多少
```

汇总脚本确实只聚合 native numerator/denominator。

这两个情况意义完全不同。

### 情况 A

如果实际是：

```text
reported sparse bit rate ≈ 40%
protected bit fraction   ≈ 1.3%
native sparse bit rate   ≈ 0%
```

那几乎可以直接判定：

> **native correction 的 partition 与 structured record 没有一一对应。**

例如：

```text
partition total_elements != structured total_elements
partition calls != structured calls
```

或者某些 partition 被多次累计。

### 情况 B

如果：

```text
reported sparse bit rate ≈ 1.3%
protected bit fraction   ≈ 1.3%
native sparse bit rate   ≈ 0%
```

那说明问题更靠前：

> **runtime FP code 的 sparse-bit extraction 就已经异常。**

因为几乎所有 reported zero bits 都只来自 outlier mask 人工制造的 zero。

### 情况 C

如果：

```text
reported ≈ 35~45%
native  ≈ 35~45%
```

那代码基本正常，只是 `results.md`/summary 的解释或读取字段出了问题。

---

# 5. 现在马上运行这个诊断脚本

我建议先**不要重跑 H0**。已有 CSV 就足够定位。

在仓库根目录运行：

```bash
python - <<'PY'
import csv
import glob
from collections import defaultdict

ROOT = "outputs/2026-09-13_phaseH_accuracy-preserving-sparsity/h0_smoke"

for cfg in ["s0_fp8_all", "s1_expert_w4"]:
    paths = glob.glob(
        f"{ROOT}/{cfg}/task*/sparsity/module_sparsity.csv"
    )

    print("\n" + "=" * 100)
    print(cfg, "files =", len(paths))
    print("=" * 100)

    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))

    acc = defaultdict(lambda: {
        "sb_r": 0,
        "tb_r": 0,
        "pb": 0,
        "sb_n": 0,
        "tb_n": 0,
        "pe": 0,
        "te": 0,
    })

    for r in rows:
        k = (
            r["component"],
            r["phase"],
            r["tensor_role"],
        )
        a = acc[k]

        a["sb_r"] += int(float(r["sparse_bits_reported"]))
        a["tb_r"] += int(float(r["total_bits_reported"]))
        a["pb"]   += int(float(r["protected_bits"]))
        a["sb_n"] += int(float(r["sparse_bits_native"]))
        a["tb_n"] += int(float(r["total_bits_native"]))
        a["pe"]   += int(float(r["protected_elements"]))
        a["te"]   += int(float(r["total_elements_reported"]))

    print(
        f"{'component':10s} {'phase':10s} {'role':12s} "
        f"{'reported':>10s} {'protected':>10s} {'native':>10s}"
    )

    for k, a in sorted(acc.items()):
        reported = a["sb_r"] / a["tb_r"] if a["tb_r"] else 0
        protected = a["pb"] / a["tb_r"] if a["tb_r"] else 0
        native = a["sb_n"] / a["tb_n"] if a["tb_n"] else 0

        print(
            f"{k[0]:10s} {k[1]:10s} {k[2]:12s} "
            f"{reported:10.4%} {protected:10.4%} {native:10.4%}"
        )
PY
```

这个输出基本就能一锤定音。

我最想看到的是类似：

```text
expert denoise activation
reported   = ?
protected  = ?
native     = ?

expert denoise output
reported   = ?
protected  = ?
native     = ?

expert denoise A/B/O
...
```

---

# 6. 同时检查 partition 和 structured record 是否严格一一对应

现在代码虽然使用相同的五维 key：

```text
module_id
phase
flow_step
tensor_role
attention_kind
```

，这是正确的。

但是 export 时只是：

```python
partition = self.outlier_partition.get(key, {})
```

随后直接做减法，没有验证：

```python
partition["calls"] == entry["calls"]

partition["total_elements"]
    == entry["total_elements"]
```

。

这是一个缺失的 correctness invariant。

应该加：

```python
if partition:
    assert int(partition["calls"]) == int(entry["calls"]), (
        key,
        partition["calls"],
        entry["calls"],
    )

    assert int(partition["total_elements"]) == total_elements, (
        key,
        partition["total_elements"],
        total_elements,
    )
```

再加：

```python
assert 0 <= protected_elements <= total_elements
assert zero_elements >= protected_elements
```

等 FP significand extractor 修完后，再要求：

```python
assert sparse_bits >= protected_bits
assert amp_zero_bits >= protected_bits
```

这样 native accounting 一旦失配，会直接 fail，而不是静默生成 `≈0%`。

---

# 7. FP significand extraction 建议直接修掉

你目前这个 4-bit metric 如果目标是后面的 bit-serial CIM，那么最合理的定义应该是：

> **unsigned significand，包括 hidden leading 1，但不包含 sign。**

也就是 E4M3：

```text
1MMM
```

normal number；

subnormal：

```text
0MMM
```

zero：

```text
0000
```

正负号另行处理，不应该挤进这 4 bit 中。

建议替换现在的 `_extract_sm_from_raw()` 核心逻辑为：

```python
def _extract_significand_from_raw(
    self,
    raw_int: torch.Tensor,
    fmt: str,
):
    fmt = fmt.lower().strip()

    if fmt == "e5m10":
        exp_bits = 5
        mant_bits = 10
    elif fmt == "e4m3":
        exp_bits = 4
        mant_bits = 3
    elif fmt == "e2m1":
        exp_bits = 2
        mant_bits = 1
    else:
        raise ValueError(fmt)

    raw = raw_int.to(torch.int64)

    mant = raw & ((1 << mant_bits) - 1)

    exp = (
        raw >> mant_bits
    ) & ((1 << exp_bits) - 1)

    # Hidden 1 ONLY for normal numbers.
    normal = exp != 0

    significand = (
        mant
        | (
            normal.to(torch.int64)
            << mant_bits
        )
    )

    return significand, mant_bits + 1
```

于是：

```text
+0          -> 0000
-0          -> 0000

+1.0        -> 1000
-1.0        -> 1000

+subnormal  -> 0MMM
-subnormal  -> 0MMM
```

这也使得你目前的 native correction：

$$
protected\_bits
=
N_{\rm protected}\times 4
$$

重新成立，因为 masked `+0/-0` 都真正统计为：

```text
0000
```

---

# 8. 加一个 T10，这次很重要

目前 H0 tests 对 INT native accounting 测得比较多，但没有真正卡住 FP8 raw encoding 语义。

建议：

```python
def test_t10_e4m3_significand_encoding():
    raw = torch.tensor([
        0x00,  # +0
        0x80,  # -0
        0x38,  # +1
        0xB8,  # -1
        0x01,  # + smallest subnormal
        0x81,  # - smallest subnormal
    ], dtype=torch.int64)

    sig, width = sm._extract_significand_from_raw(
        raw,
        "e4m3",
    )

    assert width == 4

    assert sig.tolist() == [
        0b0000,
        0b0000,
        0b1000,
        0b1000,
        0b0001,
        0b0001,
    ]
```

这个测试能同时防：

* negative zero；
* sign/magnitude 混淆；
* subnormal hidden-bit 错误。

---

# 9. 还有一个更大的概念问题：当前指标和你以前 EffLoc 的 “>85% FP sparsity” 不是同一个东西

这个也必须现在厘清，否则后面会越来越乱。

当前 Phase H 的：

```text
E4M3 sparse_bit_rate
```

实际只看 **原始 FP8 significand 的 4 bits**：

$$
1MMM
$$

代码自己也明确写的是 E4M3 → 4 bits。

而你之前 EffLoc 的高 FP bit sparsity 研究依赖的是：

> **不同 FP 值经过 exponent alignment 之后形成的 ineffective bits。**

EffLoc 的 LAU 会先比较 exponent，利用 \(\Delta E\) 对 mantissa 做逐轮 alignment；当 \(\Delta E\neq0\) 时直接产生 0 bit，之后才 shift mantissa。这种“alignment-induced zeros”正是高 bit sparsity 的主要来源，而不是简单看原始 E4M3 `1MMM`。因此不能直接把 Phase H 当前 4-bit significand statistic 和之前 >85% 的 aligned-FP sparsity 对比。

所以最好最终拆成三个概念：

| 指标                               | 统计对象                            | 用途                                  |
| -------------------------------- | ------------------------------- | ----------------------------------- |
| `significand_zero_bit_rate`      | E4M3 的 `1MMM`                   | 数值本身的 significand sparsity          |
| `encoding_zero_bit_rate`         | 完整 `S EEEE MMM` 8bit            | storage / memory representation     |
| `aligned_effective_bit_sparsity` | exponent alignment 后 bit stream | **EffLoc/CIM 真正 hardware-relevant** |

对于你现在的硬件研究，第三个其实最关键。

---

# 10. 所以 H0 当前应该怎么判

我建议把现在 `results.md` 的：

> “runtime native bit sparsity ≈0%，符合预期”

**删掉或标记为 INVALID / UNDER AUDIT**。

当前可以保留的 H0 结论是：

```text
PASS:
- rollout 正常
- SR 正常
- flow step 0..9 正常
- Linear activation/output 完整
- MatMul A/B/O 完整
- self/cross 正常
- static weight collection 正常
- sidepath collection 正常
- S0 static weight significand sparsity ≈41.5%
- S1 static weight sparsity ≈73.7%
```

但应该把：

```text
FP8 runtime native bit sparsity
```

暂时标成：

$$
\boxed{\text{INVALID — pending FP statistic/native-accounting audit}}
$$

另外，**H1 如果已经在跑，不一定需要杀掉**：rollout 和 SR 数据仍然有价值，原始 `module_sparsity.csv` 的 reported counters 也可能可用于事后重新分析。但在修正 FP significand extractor 前，不要把 H1 的 FP native bit sparsity 写进正式结论。

最关键的下一步不是重跑，而是先把上面诊断脚本的 **`reported / protected / native` 三列输出**拿出来。只要看到那三组数，我基本可以直接判断这个 `≈0%` 究竟是 **FP extractor、native subtraction，还是 H0 汇总/记录问题**。
