我重新按你远端 `main` 的最新状态审了一遍。当前仓库显示 **61 commits**。这次修改把我上轮指出的核心代码问题基本都修正了：**P0 routing、P0.5 `_should_wrap()`、Vision/Connector calibration 隔离，以及关键 routing unit tests 都已经到位。** ([GitHub][1])

现在的结论是：

> **代码层面的 V1–V3 Linear plumbing 可以 PASS。**
>
> 但还不能直接照现有实验手册复制 YAML 开跑，因为手册中的 V2/V3 配置已经和你新加的 `vision.linear.mlp / attn_proj` gate 不一致。这个是目前唯一我认为必须先修的 P0。

## 1. 上一轮几个问题现在都修对了

当前 wrapper 已经明确拆成：

```python
vision:
    enabled: true
    linear:
        mlp: ...
        attn_proj: ...
```

代码里：

```python
wrap_vision_mlp = vision_enabled and bool(
    vision_linear_cfg.get("mlp", False)
)

wrap_vision_attn_proj = vision_enabled and bool(
    vision_linear_cfg.get("attn_proj", False)
)
```

因此现在：

```text
vision.enabled=true
mlp=false
attn_proj=false
```

确实什么都不会 wrap。

而：

```text
mlp=true, attn_proj=false
→ 24 Vision MLP Linear

mlp=false, attn_proj=true
→ 48 Vision Attention projection

mlp=true, attn_proj=true
→ 72 Vision Linear
```

这已经严格符合我们要求的 V2/V3 实验语义。

`_replace_linear()` 里也已经加入：

```python
if not _should_wrap(module_id, quant_config):
    return
```

所以现在 `linear.enabled/include/exclude` 对 Vision 和 Connector 一样生效，这一点修正正确。

Calibration 问题也处理对了。Vision 和 Connector 默认：

```python
vision_cal_policy = vision_cfg.get(
    "calibration_policy", "recalibrate"
)

connector_cal_policy = connector_cfg.get(
    "calibration_policy", "recalibrate"
)
```

并且在创建完 `QuantizedLinear` 后显式覆盖：

```python
ql.calibration_policy = calibration_policy
```

这样不会再误继承旧的 `q_proj_0: reuse` 一类 component-blind policy。

---

## 2. Unit test 这一轮也明显完整了

现在你已经补到了 14 个 Vision routing tests，其中包括我上次要求的关键 case：

| Test                       | 当前结果语义 |
| -------------------------- | ------ |
| default off                | 0      |
| connector only             | 1      |
| `vision.enabled` alone     | 0      |
| Vision MLP only            | **24** |
| Vision attention proj only | **48** |
| Full Vision Linear         | **72** |
| Vision + Connector         | **73** |
| V2 cumulative              | **25** |
| module ID uniqueness       | ✅      |
| `linear.enabled=false`     | ✅      |
| attention exclude          | ✅      |
| StatManager parsing        | ✅      |
| override routing           | ✅      |
| calibration default        | ✅      |

尤其：

```python
assert n == 24
```

和：

```python
assert n == 48
```

现在都已经存在。

所以就**routing selection 本身**而言，我认为已经可以判定：

$$
\boxed{\text{PASS}}
$$

---

# 3. 现在真正的 P0：实验手册 YAML 已经落后于代码

这是这次审阅中新发现的最重要问题。

你现在代码规定：

```yaml
vision:
  enabled: true
  linear:
    mlp: true
    attn_proj: false
```

才会 wrap MLP。

但当前手册 V2 示例仍然写的是：

```yaml
quantization:
  vision:
    enabled: true
    quantize_matmul: false

  connector:
    enabled: true

  linear:
    overrides:
      - name: vision_mlp_fp8
        ...
```

这里**没有**：

```yaml
vision:
  linear:
    mlp: true
```

。

按照现在的新代码，这份 V2 YAML 实际会变成：

```text
Legacy VLM/Expert = 224
Connector         = +1
Vision MLP        = +0
----------------------
total             = 225
```

而不是你手册写的：

$$
224+1+24=249
$$

。

也就是说，如果现在直接照手册跑：

> **你以为自己在做 V2 Vision MLP FP8，实际上 Vision MLP 根本没有被 wrap。**

这个必须先修。

---

# 4. V3 手册也有同样的问题，而且更严重一点

当前 V3 示例：

```yaml
quantization:
  vision:
    enabled: true
    quantize_matmul: false

  linear:
    overrides:
      - name: vision_attn_proj_fp8
        target:
          module_id: "vision.layers.*.self_attn.*"
```

也没有：

```yaml
vision:
  linear:
    mlp: true
    attn_proj: true
```

。

因此新代码下这份 YAML：

```text
Vision MLP       0
Vision attn proj 0
```

两边都会是 0。

而且 V3 是 cumulative 实验，正确语义应该是：

```text
V1 Connector FP8
+
V2 Vision MLP FP8
+
V3 Vision Attention Projection FP8
```

所以 V3 的完整 config 不仅要打开两个 wrapper gate，还必须继续保留 V2 的 MLP precision override。

---

## 5. 建议立即把手册改成下面这样

### V2 正确形式

```yaml
quantization:
  linear_scale_granularity: per_site

  vision:
    enabled: true
    calibration_policy: recalibrate
    linear:
      mlp: true
      attn_proj: false
    quantize_matmul: false
    quantize_patch_embed: false

  connector:
    enabled: true
    calibration_policy: recalibrate

  linear:
    enabled: true
    overrides:
      - name: connector_fp8
        target:
          component: connector
        config:
          a_bit: e4m3
          w_bit: e4m3
          o_bit: e4m3
          method: pot_fp8_outlier

      - name: vision_mlp_fp8
        target:
          module_id: "vision.layers.*.mlp.*"
        config:
          a_bit: e4m3
          w_bit: e4m3
          o_bit: e4m3
          method: pot_fp8_outlier
```

这时才应该看到：

```text
existing = 224
connector = 1
vision MLP = 24

QuantizedLinear = 249
```

---

### V3 正确 cumulative 形式

```yaml
quantization:
  linear_scale_granularity: per_site

  vision:
    enabled: true
    calibration_policy: recalibrate
    linear:
      mlp: true
      attn_proj: true
    quantize_matmul: false
    quantize_patch_embed: false

  connector:
    enabled: true
    calibration_policy: recalibrate

  linear:
    enabled: true
    overrides:
      - name: connector_fp8
        target:
          component: connector
        config:
          a_bit: e4m3
          w_bit: e4m3
          o_bit: e4m3
          method: pot_fp8_outlier

      - name: vision_mlp_fp8
        target:
          module_id: "vision.layers.*.mlp.*"
        config:
          a_bit: e4m3
          w_bit: e4m3
          o_bit: e4m3
          method: pot_fp8_outlier

      - name: vision_attn_proj_fp8
        target:
          module_id: "vision.layers.*.self_attn.*"
        config:
          a_bit: e4m3
          w_bit: e4m3
          o_bit: e4m3
          method: pot_fp8_outlier
```

这时才应该看到：

$$
224+1+24+48
=
\boxed{297}
$$

---

# 6. README 也有一个小的语义不一致

README 现在写：

```text
vision_model:
quantization.vision.enabled: true 开启
```

但根据新实现：

```text
vision.enabled=true
```

本身已经**不会开启任何 Linear**。

实际上应该写成：

```text
quantization.vision.enabled=true
+
quantization.vision.linear.mlp=true
and/or
quantization.vision.linear.attn_proj=true
```

。README 已经正确把 Vision/Connector 标成 opt-in，但这里最好补清楚 sub-group gate，否则以后自己回来看很容易误解。([GitHub][1])

这个属于 P2 文档问题，不影响代码。

---

# 7. 代码本身还有没有新的 blocker？

我目前**没有再看到 V1–V3 routing 的代码级 P0**。

当前调用链已经是：

```text
_wrap_smolvla_linear_layers()
        ↓
legacy 224

_wrap_smolvlm_vision_linear_layers()
        ↓
0 / 1 / 24 / 48 / 72 / 73
```

并且两部分是独立统计的。`ModelWrapper._wrap_smolvla()` 已经真正调用新的 Vision wrapper，而不是只写了 helper 没接主路径。

这点非常重要，现在 plumbing 是完整接通的。

---

# 8. 但 real-model raw equivalence 仍然没有完成

这一点和 unit routing test 是两回事。

当前测试里的 mock 还是类似：

```text
fc1:       768 → 768
fc2:       768 → 768
connector: 768 → 768
```

而真实模型大致应该是：

```text
fc1       768  → 3072
fc2       3072 → 768
connector 12288 → 960
```

当前 tests 主要证明：

> module discovery / routing 是对的。

它还没有证明：

> **真实 SmolVLM 被 wrap 后，raw mode 和原模型数值完全等价。**

所以在 V1-FP8 之前仍然要完成这个 Gate。

建议至少比较：

```text
original connector output
vs
wrapped-raw connector output

original Vision MLP output
vs
wrapped-raw Vision MLP output
```

保存：

```text
max_abs_error
mean_abs_error
max_rel_error
```

对于单纯的 `nn.Linear → QuantizedLinear(mode=raw)`，理论上：

```python
F.linear(x, copied_weight, copied_bias)
```

应该达到 exact 或极接近 exact。

---

# 9. V0 仍然应该先跑

代码提前写好没关系，因为：

```text
vision.enabled = false
connector.enabled = false
```

时不会影响旧模型。

但实验顺序还是应该保持：

```text
routing code ready
        ↓
V0 workload audit
        ↓
确认真实 camera calls / tensor shapes / FLOPs
        ↓
V1 Connector raw
        ↓
V1 Connector FP8
        ↓
V2
```

手册自己也明确规定 V0 必须得到 runtime camera call 数、token 数和 FLOPs，并且不能 hard-code camera 数。

---

# 10. V4 目前继续不要动

当前：

```yaml
vision:
  quantize_matmul: true
```

仍然没有实现真正的 Vision `QK/PV` patch。

现有 `_inject_smolvla_quantized_matmul()` 仍然只创建：

```text
vlm × 16 × 2
expert × 16 × 2
= 64
```

，没有创建：

```text
vision × 12 × 2
= 24
```

。

这是**预期状态**，不是这轮需要修的 bug。

因此当前开发边界应该明确为：

```text
V1 ✅ code ready
V2 ✅ code ready
V3 ✅ code ready

V4 ❌ not implemented yet
V5 ❌ deferred
```

---

## 当前最终审计状态

| 项目                             | 状态                                              |
| ------------------------------ | ----------------------------------------------- |
| `vision.enabled` default-off   | ✅                                               |
| MLP / attn projection 独立 gate  | ✅                                               |
| Vision MLP only = 24           | ✅                                               |
| Vision attn only = 48          | ✅                                               |
| Full Vision = 72               | ✅                                               |
| Connector = 1                  | ✅                                               |
| `_should_wrap()`               | ✅                                               |
| include/exclude                | ✅                                               |
| component-specific calibration | ✅                                               |
| canonical module IDs           | ✅                                               |
| StatManager parser             | ✅                                               |
| main ModelWrapper integration  | ✅                                               |
| routing unit tests             | ✅                                               |
| **V2 manual YAML**             | **❌ P0：缺 `vision.linear.mlp=true`**             |
| **V3 manual YAML**             | **❌ P0：缺 group gate + cumulative MLP override** |
| README gate wording            | ⚠️                                              |
| real-model raw equivalence     | ⏳                                               |
| V0 runtime FLOPs audit         | ⏳                                               |
| Vision QK/PV V4                | ⏳，尚未实现                                          |

所以我这次的建议不是再改核心 wrapper。

**核心 wrapper 可以冻结。**

现在先做一个很小的文档/config 修复，把所有 V2/V3 YAML 改成新的 group-gate 语义。然后不要继续写 V4，直接进入：

```text
1. pytest routing
2. legacy 224 Gate
3. V0 workload audit
4. V1 raw equivalence
5. V1 calibration
```

这几项通过以后，Phase I 才算真正开始进入实验阶段。

[1]: https://github.com/wolf111108/SmolVLA_qtrsc?x=202609150526 "GitHub - wolf111108/SmolVLA_qtrsc · GitHub"
