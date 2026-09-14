我重新审阅了远端最新提交 `e4034aa`、`quant_methods.py`、`stat_manager.py`、T10/T11、H1-Audit 配置和 `results.md`。这次 audit **成功定位到了真正的 bug，根因判断成立，修复方式也正确**。但当前 `results.md` 还有一个比较重要的实验记录问题：**不能把单次 S0 task0 audit 得到的 39.90% 直接回填成原 H1 的 S0/S1 10-episode aggregate 结果。**

## 审阅结论

| 项目                                    | 判定                   |
| ------------------------------------- | -------------------- |
| H1-Audit 设计                           | ✅ 正确                 |
| 独立 E4M3 raw-code audit                | ✅ 正确                 |
| output/O≈0.5% 的根因                     | ✅ 已定位                |
| `mul_ → mul` 修复                       | ✅ 正确                 |
| forward/SR 不受 bug 影响                  | ✅ 结论成立               |
| T10 FP significand 语义                 | ✅ 正确                 |
| T11 回归测试                              | ✅ 有效，但可加强            |
| 修复后 output/O≈39.9%                    | ✅ S0 task0 audit 可确认 |
| 原 H1 activation/A/B/SR                | ✅ 仍有效                |
| 原 H1 output/O                         | ❌ 原 CSV 已污染          |
| `results.md` 中“H1 S0/S1 output=39.9%” | ⚠️ 目前证据不足            |
| H2 是否可以开始                             | ✅ 可以，但建议先修正文档语义      |

### 1. 根因定位完全合理

当前 Linear 路径现在是：

```python
out_normal_quant = quant_awo(
    out_normal,
    layer.o_interval,
    layer.o_spec,
    out_dtype=out_normal.dtype,
)

out_normal_dequant = (
    out_normal_quant
    .to(torch.float32)
    .mul(M_q)
    .to(x.dtype)
)
```

而旧代码这里是 `.mul_(M_q)`。

关键点在于 `out_normal` 是 FP32，因此 scalar-scale 的 `quant_awo()` 最终创建的 `out_normal_quant` 也是 **FP32 tensor，只是数值已经被限制到 E4M3 可表示值**。`quant_awo()` 的 scalar FP 路径确实是先转成 FP8，再写回 `out_dtype`。

因此旧代码：

```python
out_normal_quant.to(torch.float32).mul_(M_q)
```

这里的：

```python
out_normal_quant.to(torch.float32)
```

当输入已经是 FP32 时，不需要创建新的 tensor；随后 `mul_()` 就可能直接修改 `out_normal_quant` 自身。

于是实际发生的是：

$$
q_O
\overset{\text{mul\_}}{\longrightarrow}
q_O M_q
$$

原本应该保留下来用于统计的：

```text
[-448,448] 内的 E4M3 code value
```

被覆盖成 dequant 中间值。

而 collector 又是在 dequant 之后读取：

```python
output_code=out_normal_quant
```

因此之前统计的根本不是 FP8 output code。这个分析与你 commit 描述完全一致。([GitHub][1])

---

## 2. 为什么 forward 精度没受影响，也是说得通的

这一点很容易让人疑惑：

> 既然 `out_normal_quant` 被改坏了，为什么 SR 没坏？

因为模型真正返回的是：

$$
O_{\rm dequant}
=
\frac{q_O M_q}{2^{16}}
$$

旧代码虽然把 `q_O` 原位覆盖成了 `q_OM_q`，但这正好也是 `out_normal_dequant` 接下来需要的中间结果。随后：

```python
torch.div(..., 2**16)
```

是 out-of-place 的，因此 forward 得到了正确的 dequantized output。

也就是说旧代码相当于：

```text
qO tensor
   ↓ in-place × Mq
dequant intermediate
   ↓ / 2^16
correct model output
```

真正受害的是：

```text
qO 本身已经不存在了
```

而 sparsity collector 恰好在后面还把它当作 `qO`。

所以：

$$
\boxed{\text{forward 正确，但 instrumentation 错误}}
$$

这是非常典型的“统计探针观察到了被复用的临时 buffer”问题。

因此 H1 已有的：

```text
SR
activation
A
B
FP sidepath
```

没有因为这个问题失效。

---

## 3. 三处 `mul_ → mul` 修复都对

你现在已经修了三个路径：

```text
Linear pot_fp8_outlier
Linear pot_ao_outlier_channel
MatMul pot_fp8_outlier
```

commit diff 中三处都是：

```python
.mul_(M_q)
```

变为：

```python
.mul(M_q)
```

。([GitHub][1])

这样：

```python
out_normal_quant
```

继续保持原始 quantized code，

而：

```python
out_normal_dequant
```

得到一个独立 tensor。

所以现在：

```text
out_normal_quant
        ├── collector → FP8 code sparsity
        │
        └── mul(Mq) → dequant → model forward
```

数据流终于是干净的。

我没有在 output quant code 上发现新的同类 mutation。

代码里还有：

```python
out_qa_qb.mul_(...)
out_fa_qb.mul_(...)
out_qa_fb.mul_(...)
```

这些 in-place 操作目前不是同一类问题，因为这些 tensor 本身就是 GEMM 临时 accumulator，之后没有被当成“原始 quantized operand code”送进 sparsity collector。

所以**不要机械地把所有 `mul_` 都替换掉**。

---

## 4. 独立 audit 本身设计是可信的

这一版 audit 最重要的地方是：它没有复用正式统计器的 `_extract_sm_from_raw()`。

它独立执行：

```text
FP8 raw uint8
    ↓
exp/mant 解析
    ↓
normal:    1MMM
subnormal: 0MMM
zero:      0000
```

而且 sign 不进入 4-bit significand。这和当前 Phase H 定义一致。

当前正式 `_unpack_sm_exp()` 也已经修正为：

```python
sm = mant
normal = exp != 0
```

并明确规定：

```text
+0/-0 -> 0000
subnormal 不加 hidden 1
```

。

T10 也确实锁住了：

```text
+0          -> 0000
-0          -> 0000
+1          -> 1000
-1          -> 1000
+subnormal  -> 0001
-subnormal  -> 0001
```

。

因此现在有两套独立路径：

$$
\text{正式 collector}
$$

和

$$
\text{raw-code audit}
$$

最后 output/O 都得到约 39.9%。

这是很强的交叉验证。

---

# 5. 所以 output/O 的正确结论是什么？

Audit 后：

$$
S_{\rm output,native}\approx39.90\%
$$

$$
S_{\rm O,native}\approx39.91\%
$$

而独立 raw-code audit：

$$
S_{\rm sig,nz}\approx39.90\%,39.91\%.
$$

`results.md` 记录了两边完全一致。

因此可以正式推翻之前：

$$
0.3\%-0.6\%
$$

的结果。

现在 runtime 的总体 picture 更合理：

$$
\boxed{
A/O/B/\text{activation significand sparsity}
\approx 40\%-56\%
}
$$

而不是：

```text
input 40~56%
output ~0%
```

这意味着我上一轮提出的“SmolVLA output 可能具有特殊 1.111 分布”的假设已经被 audit 否定。

**不需要再做 prequant/scale audit。**

问题不是模型 numerical distribution，也不是 output scale，而是 instrumentation aliasing。

---

# 6. 但现在 `results.md` 有一个重要问题

这是我认为你提交后还应该修的一项。

当前文档写：

```text
H1 结果（native sparse_bit_rate，10ep）

Expert denoise output:
S0 = 39.9%
S1 = 39.9%

QK/PV O:
S0 = 39.9%
S1 = 39.9%
```

。

但你的 H1-Audit config 明确只跑了：

```text
S0 FP8-all
libero_goal
task_id = 0
n_episodes = 1
```

。

也就是说真正 post-fix 测量到的证据是：

$$
\boxed{\text{S0 task0 × 1ep}}
$$

而不是：

$$
\boxed{\text{S0 10ep + S1 10ep}}
$$

原来的 H1 10ep 是在 bug 存在时跑的，所以那些 H1 CSV 里的：

```text
output
O
```

字段已经被污染。

**修代码不会自动修复之前生成的 CSV。**

因此现在不能把 audit 的：

```text
39.90%
```

直接当成：

```text
H1 S0 10ep aggregate = 39.90%
H1 S1 10ep aggregate = 39.90%
```

---

## 建议 `results.md` 改成这样

H1 表里：

| Metric                    |                 H1 S0 |                 H1 S1 | H1-Audit S0 task0 |
| ------------------------- | --------------------: | --------------------: | ----------------: |
| VLM prefill activation    |                40.70% |                40.65% |                 — |
| Expert denoise A          |                56.10% |                55.55% |                 — |
| Expert denoise activation |                40.58% |                40.65% |                 — |
| Expert denoise output     | **INVALID (pre-fix)** | **INVALID (pre-fix)** |        **39.90%** |
| QK/PV O                   | **INVALID (pre-fix)** | **INVALID (pre-fix)** |        **39.91%** |
| FP sidepath               |                 1.23% |                 1.27% |                 — |

然后写：

> H1-Audit 在 S0 Goal task0 ×1ep 上确认修复后 output/O native significand sparsity 为 39.90%/39.91%，并与独立 E4M3 raw-code audit 一致。原 H1 10ep 的 output/O 统计产生于修复前，因此不作为 H1 aggregate 数值使用；H2 将使用修复后的 collector 得到正式 multi-task aggregate。

这样最严谨。

---

# 7. H0 那张表也有相同问题

现在 §3.1 写：

```text
expert denoise output/O:
native = 39.90 / 39.91
```

。

如果 H0 没有在 commit `e4034aa` 后重新执行，那么这也不是原 H0 的真实测量值。

应写成：

```text
original H0 output/O:
INVALID — instrumentation bug
```

然后单独注明：

```text
H1-Audit post-fix:
output = 39.90%
O      = 39.91%
```

除非你已经在修复后重新跑了一次 H0。

从当前远端提交记录来看，我看到新增的是：

```text
s0_fp8_all_h1_audit_task0.yaml
```

没有看到 S1 audit 或完整 H0 rerun 的证据，所以建议按照“未 rerun”处理。([GitHub][1])

---

# 8. 不需要因此重跑整个 H1，H2 可以直接继续

这里不必为了这个问题浪费实验时间。

原 H1 仍然保留了很多有效信息：

$$
S0\,SR=90\%,\qquad S1\,SR=80\%
$$

以及：

```text
activation
A
B
sidepath
static weight
```

这些统计发生在 output code mutation 之前，或者来自独立 static collector，因此仍然有效。

所以最合理的路线是：

$$
\boxed{
\text{保留 H1 有效字段}
\rightarrow
\text{将 H1 output/O 标 INVALID}
\rightarrow
\text{直接跑修复后的 H2}
}
$$

H2 有：

```text
10 tasks × 3 episodes
```

它会自然给出真正可靠的：

```text
S0 output/O aggregate
S1 output/O aggregate
```

届时 H2 本身就能告诉你它是不是稳定在 39.9%左右。

如果你特别希望保留“H1→H2 收敛比较”中的 output/O，则需要重新跑：

```text
S0 10 tasks ×1ep
S1 10 tasks ×1ep
```

总共 20 episodes。

但从研究效率看，我认为**没有必要**。直接 H2 更划算。

---

# 9. T11 是好的 regression test，但还可以再加强

当前 T11 检查：

```python
rate > 0.10
```

目的是确保不会再次掉回约 1% 的异常值。

这足以防止本次 bug 回归，所以现在可以用。

但它实际上没有直接验证：

> `out_normal_quant` 没有被修改。

建议未来改成更直接的 invariant：

```text
collected output code:
all finite
abs(code) <= 448
```

例如：

```python
assert torch.isfinite(captured_output_code).all()
assert captured_output_code.abs().max() <= 448
```

然后 sparsity：

```python
assert 0.1 < rate < 0.8
```

作为第二层 sanity check。

这样比只检查 `rate > 0.10` 更能指向真正的 regression。

不过这属于 P2，不阻塞 H2。

---

# 10. 一个很重要的研究层面修正

当前结果应当表述为：

$$
\boxed{
\text{FP8 4-bit significand sparsity}
}
$$

而不是笼统说：

$$
\text{FP8 bit sparsity}.
$$

因为你统计的宽度明确是：

$$
M+1
$$

E4M3：

$$
1MMM=4\ bits
$$

而不是存储编码：

$$
SEEEEMMM=8\ bits.
$$

`_spec_bitwidth()` 当前也明确返回 `mant_bits + 1`。

所以目前得到：

```text
activation ≈40%
A          ≈50–56%
B          ≈40%
output/O   ≈40%
```

严格名称最好统一成：

```text
native significand sparse-bit rate
```

这与后续 EffLoc 的：

```text
exponent-aligned effective bit sparsity
```

又是不同指标。

现在 `results.md` 已经开始注明“4-bit significand（1MMM）口径”，这一点是正确的。

---

## 当前 Phase H Gate

我现在会正式给出：

```text
H0 structural pipeline          PASS
H1 pilot rollout               PASS
H1-Audit                       PASS
FP significand extractor        PASS
output/O aliasing root cause    CONFIRMED
mul_ -> mul fix                PASS
T10/T11 regression protection  PASS

H1 SR                           VALID
H1 activation/A/B              VALID
H1 weight sparsity             VALID
H1 sidepath                    VALID
H1 pre-fix output/O aggregate  INVALID

Audit S0 task0 output/O        VALID (~39.9%)

H2 READY                        YES
```

所以现在**不需要继续追 output/O 这个异常，也不需要做 scale audit**。这个问题已经闭环。

在启动 H2 前我只建议做一件事：把 `results.md` 里 H1/H0 的 `39.9%` 标记成 **H1-Audit post-fix measurement，而不是原 H1/H0 aggregate**。然后 H2 用现在的修复版本直接跑，H2 的 output/O 才作为正式 multi-task 数据。

这次 audit 实际上非常有价值：它确认了**模型本身的 runtime significand sparsity没有异常，异常来自 instrumentation 的 in-place aliasing**，因此 Phase H 后续的统计基础现在比之前可靠得多。

[1]: https://github.com/wolf111108/SmolVLA_qtrsc/commit/HEAD "fix: H1-Audit 修正 output/O 稀疏度统计 bug (0.5%→39.9%) · wolf111108/SmolVLA_qtrsc@e4034aa · GitHub"
