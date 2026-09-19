我重新读取了远端最新 `main`，包括四个 config、runner、汇总脚本、实验文档、scale manifest，以及当前 `quant_spec.py / quant_methods.py / stat_manager.py / model_wrapper.py`。结论是：

> **当前 quick-scan 的四组量化配置本身没有发现 P0 级数值错误，可以开跑。**
> INT8/INT16 路径、FP8W4 PoT、FP8 PoT 的量化 method 和 sparsity collector 都能正确接上当前框架。
> 但在正式运行前，我建议修 **3 个 P1**：一个文档口径错误、一个 secondary 汇总 bug、一个 Gate/幂等性问题。另外需要明确实验统计范围并不是整个 SmolVLA。

| 审计项                           |                判定 |
| ----------------------------- | ----------------: |
| Q0 INT8 config                |                 ✅ |
| Q1 INT16 config               |                 ✅ |
| INT16 当前框架支持                  |                 ✅ |
| Q2 FP8(A/O)-W4 + PoT(A/O)     |                 ✅ |
| Q3 FP8 PoT                    |                 ✅ |
| Q2/Q3 scale 副本策略              |              ✅ 很好 |
| `--skip-calibration` 防覆盖      |                 ✅ |
| element native sparsity       |                 ✅ |
| INT bit sparsity              |                 ✅ |
| FP8 significand sparsity      |                 ✅ |
| Linear output 旧 `mul_` bug    | ✅ 当前远端已是 `.mul()` |
| MatMul O 旧 `mul_` bug         |                 ✅ |
| prefill / denoise headline 聚合 |                 ✅ |
| Secondary QK/PV 分项            |              ❌ P1 |
| Gate 实际退出状态                   |             ⚠️ P1 |
| `reported` 的文档解释              |              ❌ P1 |
| 统计范围命名                        |             ⚠️ P1 |
| `README_EXPERIMENT.md` 远端引用   |             ⚠️ P2 |

## 1. 四个 config 的核心设置是正确的

Q0 确实是全范围：

```text
Linear A/W/O = INT8
QK/PV A/B/O  = INT8
method       = outlier
outlier      = 1%
```

并且校准明确 `recalibrate`，1 episode / batch 8 / stride 16。

Q1 只是把相同路径换成 INT16：

```text
Linear A/W/O = INT16
QK/PV A/B/O  = INT16
```

没有误把 `batch_size` 等字段也替换成 16。

当前 `parse_quant_spec()` 对任意 Python integer 都直接生成：

```python
QuantSpec(kind="int", bits=value)
```

所以 `16` 是真正的 INT16，不是 fallback 或伪 FP16。INT scale、clamp 和 fake-quant 路径也都按 `2^(bits-1)` 工作。

而 `model_wrapper` 的 method 解析是：

```python
layer_config.get(
    "method",
    quant_config.get("method", "per_tensor"),
)
```

因此 Q0/Q1 虽然每个 `q_proj` 节点没有重复写 `method: outlier`，也确实会继承顶层 `outlier`；MatMul 同样如此，并映射到 `matmul_outlier`。

这块不用改。

---

## 2. Q2/Q3 的 scale 处理比我最初建议的更安全

你没有直接让 Q2/Q3 指向 Phase-G/H 正在使用的 shared scale，而是复制到：

```text
scales/2026-09-15_sparsity-ratio-quickscan/
    q2_fp8w4_po2/
    q3_fp8_po2/
```

并记录了 864 个文件的 SHA256 manifest。864 这个数量也很合理：

$$
224\text{ Linear}\times3
+
64\text{ MatMul}\times3
=
864.
$$

文档同时规定 runner 对 Q2/Q3 强制：

```bash
--skip-calibration
```

因此不会意外改写副本。

Q2 的语义也写对了：

```text
A/O = E4M3 + PoT scale
W   = INT4 + continuous scale
```

因为 `pot_ao_outlier` 的实现确实只把 activation/output scale 投影到 PoT，weight scale 保持 continuous。

Q3 则是 Linear 和 MatMul 全 E4M3，`pot_fp8_outlier` 会检查 A/W/O 三个 scale 都是 2 的幂。

所以当前命名：

> FP8(A/O)-W4 + PoT(A/O)

是正确的。

---

# 3. 之前 H1-Audit 找到的 output aliasing bug，现在当前 main 是修复状态

这一点我特别重新确认了，因为如果这里回归，会直接毁掉这四组的 output sparsity。

当前 Linear 是：

```python
out_normal_dequant = (
    out_normal_quant
    .to(torch.float32)
    .mul(M_q)
    .to(x.dtype)
)
```

不是 `.mul_()`。collector 后面看到的还是原始：

```python
out_normal_quant
```

。

MatMul O 同样：

```python
out_normal_quant.to(torch.float32).mul(M_q)
```

没有原地覆盖 O code。

因此这次 INT8/INT16/FP8 的 `output/O` bit sparsity 不会重现之前 0.5% 的 instrumentation bug。

---

# 4. P1：文档中 `reported` 的方向写反了

这是确定的文字错误。

现在 `experiment_setup.md` 写：

> protected FP sidepath 会在 normal quant path 制造人工 0，用 `reported` 会**低估稀疏度**。



应该恰好反过来。

因为 protected position：

```text
真实值 ≠ 0
        ↓ outlier mask
normal quant path = 0
```

于是 reported tensor 中多出了人工零：

$$
S_{\rm reported}
>
S_{\rm native}
$$

一般意义上是**高估 sparsity**。

正确文字应改为：

> protected FP sidepath 会在 normal quant path 中制造人工 0，因此 `reported` 会**高估可归属于 native quantized workload 的 sparsity**；本实验统一使用 `native` counters，排除 protected positions。

`summarize_quick_sparsity.py` 顶部 docstring 也有同样一句：

```text
reported would understate sparsity
```

也应改成：

```text
reported would overstate native quant-path sparsity
```

。

这不影响当前计算，但**必须在结果出来前改**，否则以后很容易解释反。

---

# 5. P1：`quick_sparsity_by_role.csv` 现在分不开 QK 和 PV

这是当前真正的汇总脚本 bug。

你期望 secondary 表是：

```text
Linear activation
Linear output

QK A
QK B
QK O

PV A
PV B
PV O

Weight q/k/v/o/gate/up/down
```

`results.md` 也明确这么写。

但当前代码实际只按：

```python
rt_role[(stage, role)]
```

聚合。

所以：

```text
QK A
PV A
```

最终都会进入同一个：

```text
A
```

桶。

同理：

```text
QK B + PV B -> B
QK O + PV O -> O
```

所以最后 `quick_sparsity_by_role.csv` 实际只能得到：

```text
activation
output
A
B
O
```

无法回答你文档中已经承诺的 QK/PV 区别。

### 建议现在改

在读取 runtime row 时加入：

```python
operator = (r.get("operator") or "?").strip()
```

将：

```python
rt_role[(st, role)]
```

改成：

```python
rt_role[(st, operator, role)]
```

后面 merge：

```python
for (s, operator, role), acc in rt_role.items():
    if s != st:
        continue

    op = operator.lower()

    if op in {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }:
        label = f"Linear:{role}"
    elif "qk" in op:
        label = f"QK:{role}"
    elif "pv" in op:
        label = f"PV:{role}"
    else:
        label = f"{operator}:{role}"

    ...
    merged[label][k] += acc[k]
```

那么 secondary 才会真正输出：

```text
Linear:activation
Linear:output
QK:A
QK:B
QK:O
PV:A
PV:B
PV:O
```

其中当前 attention 实现也确认：

$$
PV_A=P=\mathrm{softmax}(...)
$$

$$
PV_B=V.
$$



**Primary 表不受这个 bug 影响。**

---

# 6. P1：当前 Gate 打了 `[FAIL]` 也可能最后 exit 0

现在：

```python
check_gate()
```

会产生：

```text
FAIL
WARN
OK
```

比如 native counter 出负数会：

```python
msgs.append(("FAIL", ...))
```

。

但是主循环里的：

```python
all_fail
```

只有这里会置 True：

```python
if not gate["has_summary"]:
    all_fail = True
```

也就是说：

```text
[FAIL] negative counters
```

屏幕上虽然出现 FAIL，程序最后仍可能：

```text
exit 0
```

runner 就会打印：

```text
Quick scan finished.
```

这是实验自动化上不够安全。

建议：

```python
gate_msgs = check_gate(cfg_dir, gate)

for level, msg in gate_msgs:
    print(f"  [{level}] {msg}")

if any(
    level == "FAIL"
    for level, _ in gate_msgs
):
    all_fail = True
```

同时我建议这两项由 WARN 提升到 FAIL：

```text
denoise flow_step 未覆盖 0..9
Linear/MatMul tensor roles 不完整
```

因为本实验的 contract 就要求它们必须存在。

---

# 7. Runner 的幂等判断也可以再稳一点

现在只检查：

```bash
if [[ -f "$out/sparsity/module_sparsity.csv" ]]; then
    SKIP
fi
```

。

但 `main.py` 的导出顺序是：

```text
module_sparsity.csv
...
weight_sparsity_static.csv
...
```

。

因此极端情况下：

```text
module CSV 已成功
↓
后续 export crash
```

下一次 runner 就会误判“已经完成”。

建议改成至少：

```bash
if [[
    -f "$out/sparsity/module_sparsity.csv" &&
    -f "$out/sparsity/weight_sparsity_static.csv"
]]; then
    echo "[SKIP] $name 已完成 sparsity primary outputs"
    continue
fi
```

这属于 robustness，不影响数值。

---

# 8. 一个研究口径要现在写清楚：这不是“整个 SmolVLA”的 sparsity

这是我认为最应该加到 issue 和 `results.md` 的一句话。

当前 wrapper 的显式 quantized Transformer targets 是：

```text
VLM text_model 16 layers:
    q/k/v/o + gate/up/down

Action Expert 16 layers:
    q/k/v/o + gate/up/down

VLM/Expert:
    QK + PV
```

。

也就是说这轮的：

```text
VLM prefill sparsity
```

严格含义是：

$$
\boxed{
\text{VLM text-Transformer quantized scope during prefill}
}
$$

不包括：

```text
SigLIP vision encoder
vision connector
```

也不应被描述成：

> 整个 SmolVLA prefill 的所有计算 sparsity。

尤其你之前已经分析过 vision encoder FLOPs 很大，所以这里必须避免读者误解。

建议结果表上方直接写：

> **Scope:** 当前量化框架覆盖的 VLM text Transformer + Action Expert Transformer 的 Linear/QK/PV；vision encoder、connector 及未包装的非 Transformer 算子不计入本 quick scan。

这样截图里的“SmolVLA sparsity ratio”才不会过度外推。

---

# 9. Primary headline 还有一个定义需要明确

你现在 Primary：

```text
Runtime element sparsity
Runtime bit sparsity
```

是把某 stage 下：

```text
Linear activation
Linear output
QK A/B/O
PV A/B/O
```

的所有 observed tensor elements 全部 sum numerator / denominator。这个实现本身没错。

但它是：

$$
\boxed{\text{observed-tensor-element-weighted aggregate}}
$$

不是：

$$
\text{MAC-weighted sparsity}
$$

也不是：

$$
\text{unique activation-memory sparsity}.
$$

因为同一语义数据可能先作为 Linear output，再作为下一个 operator input，被统计在不同 role 中。

因此建议结果表脚注写：

> Runtime headline 是所有已 instrumentation 的 quantized tensor observations 按 element/bit denominator 加权后的 aggregate；不等于 FLOP/MAC weighted sparsity，也不直接代表 hardware speedup。

这和你现在“不推导 speedup”的原则是一致的。

---

# 10. 当前静态校验做得很好

`logs.md` 里已经用 Phase-H H2 S0 task00 做了真值测试：

```text
VLM prefill
elem = 1.92%
bit  = 41.69%
wbit = 40.86%

Expert denoise
elem = 3.03%
bit  = 42.15%
wbit = 42.58%
```

其中 Expert FP8 weight 42.58% 与 Phase-H 独立 summary 完全一致。

所以至少可以确认：

$$
\boxed{
\text{Primary numerator/denominator aggregation implementation 正确}
}
$$

这一步做得很有价值。

另外当前 INT bit collector 对 INT16 也安全：使用 low-\(n\)-bit mask、INT32 临时值，\(n=16\) 不会越界；sign-aware 定义也正是文档现在描述的定义。

---

# 11. 一个小的文档问题：`README_EXPERIMENT.md` 当前远端似乎没有

四个 config 和多个 docs 都引用：

```text
README_EXPERIMENT.md
```

例如 Q0 第 3 行就指向它。

但我直接读取：

```text
experiments/2026-09-15_sparsity-ratio-quickscan/README_EXPERIMENT.md
```

当前远端返回 404。

如果你本地有这个文件，很可能漏 commit/push 了；如果你决定使用：

```text
docs/experiment_setup.md
```

作为唯一 protocol 文档，那就把所有 `README_EXPERIMENT.md §x` 引用替换掉。

不影响运行，只影响 provenance。

---

## 我现在给出的最终 Gate

目前没有必要碰 core code，也**不要在 H3 运行中修改 core code**。

当前状态：

```text
Q0 INT8 config                  PASS
Q1 INT16 config                 PASS
Q2 FP8W4 PoT config             PASS
Q3 FP8 PoT config               PASS
scale provenance                PASS
native element accounting       PASS
bit accounting                  PASS
output aliasing fix             PASS
primary aggregation             PASS

secondary QK/PV grouping        FIX BEFORE SUMMARY
gate exit semantics             RECOMMENDED FIX
reported/native wording         MUST FIX DOC
runner partial-run skip         RECOMMENDED FIX
experiment scope wording        MUST CLARIFY
README reference                CLEANUP
```

因此：

$$
\boxed{\text{QUICKSCAN NUMERICAL PIPELINE = READY}}
$$

如果 GPU 已经空出来，**即使不修 secondary grouping，也可以开始 Q0–Q3**，因为原始 `module_sparsity.csv` 不会丢信息，之后改 summary 脚本可以重新汇总；但我建议先花 5–10 分钟把上述脚本问题修掉，再一键跑，这样跑完即可直接得到可提交的 Primary + QK/PV Secondary 表。
