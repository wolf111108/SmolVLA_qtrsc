我重新检查了你远端 `main` 的最新提交；仓库现在是 **60 commits**，`2026-09-15_phaseI_vision_quantization_experiment_manual.md`、Vision/Connector wrapper、README 更新和 `tests/test_vision_quant_routing.py` 都已经推上去了。整体方向是对的，而且官方 SmolVLM 的真实结构路径也匹配：Vision attention 确实是 `q_proj/k_proj/v_proj/out_proj`，MLP 是 `fc1/fc2`，Connector 确实是 `connector.modality_projection.proj`。([GitHub][1])

不过我发现了一个**会直接破坏 V2/V3 实验语义的 P0 问题**。所以目前我的判定是：

> **基础 plumbing 基本正确，Legacy 默认关闭也正确；但暂时不要开始 V2/V3 正式量化实验。先修 routing selection。**

### 当前做对的部分

你这次最核心的架构改法是合理的。新增了独立的 `_wrap_smolvlm_vision_linear_layers()`，没有把 Vision 逻辑硬塞进现有 VLM/Expert loop；`vision.enabled` 和 `connector.enabled` 也默认都是 `False`，因此旧 G5/G6 配置不写新字段时不会自动进入 Vision。这满足我们最重要的 backward-compatibility 要求。

Vision 的名字也全部写对了：

```python
VISION_ATTN_LINEAR_NAMES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
]

VISION_MLP_LINEAR_NAMES = [
    "fc1",
    "fc2",
]
```

这和 Transformers 官方 `SmolVLMVisionAttention` / `SmolVLMVisionMLP` 一致，特别是你没有把 Vision 的 `out_proj` 错写成 text LLM 的 `o_proj`。Connector 的：

```python
vlm_model.connector.modality_projection.proj
```

也和官方代码一致。

canonical module ID 的设计也很好：

```text
vision.layers.0.self_attn.q_proj
vision.layers.0.self_attn.out_proj
vision.layers.0.mlp.fc1
vision.layers.0.mlp.fc2

connector.layer.0.connector_proj
```

现有 `QuantStatManager._parse_module_id()` 本来就是取第一段作为 component、最后一段作为 operator，并从 `layers/layer` 后解析 index，因此 `vision` 和 `connector` 不需要增加新的 ContextVar 或 parser 特判。你新增的单测也已经验证这一点。

---

## P0：`vision.enabled=true` 目前会一次性 wrap 全部 72 个 Vision Linear

这是当前最需要修的地方。

你现在的代码是：

```python
if vision_enabled:
    for layer_idx, layer in enumerate(encoder.layers):
        for name in VISION_ATTN_LINEAR_NAMES:
            _replace_linear(...)

        for name in VISION_MLP_LINEAR_NAMES:
            _replace_linear(...)
```

也就是说：

$$
12\times(4+2)=72
$$

只要：

```yaml
vision:
  enabled: true
```

就会立刻把：

```text
48 attention projections
+
24 MLP
```

**全部变成 `QuantizedLinear`**。

但我们的实验手册明确要求：

```text
V1: Connector only          → +1
V2: Vision MLP only         → +24
V3: Vision attention proj   → 再 +48
```

因此 V2 应该是：

$$
224+1+24=\boxed{249}
$$

而不是：

$$
224+1+72=\boxed{297}
$$

手册的 V2 也明确规定“只改变 Vision MLP precision”。([GitHub][2])

### 为什么 `linear.overrides` 救不了这个问题

比如 V2 写：

```yaml
linear:
  overrides:
    - target:
        module_id: "vision.layers.*.mlp.*"
      config:
        a_bit: e4m3
        w_bit: e4m3
        o_bit: e4m3
```

这个 override **只是修改已经 wrapped module 的 quant config**，并不是告诉 wrapper：

> 只有这些 module 才 wrap。

`resolve_linear_quant_config()` 对不匹配的 Vision attention projection 只会得到原 base/default config；它们依然是 `QuantizedLinear`，然后 `switch_quantization_mode_all()` 又会统一把所有 QuantizedLinear 切成 `quant_forward`。因此 V2 实际上很可能变成：

```text
Vision MLP       → FP8（override）
Vision q/k/v     → 当前 q/k/v base precision
Vision out_proj  → 默认 precision
```

而不是“只量化 MLP”。这会让 V2 的结果完全无法解释。

你当前新增的测试也暴露了这一点：测试覆盖了 `connector only` 和 `full vision = 72`，但手册要求的两个关键 case：

```text
Vision MLP only       = 24
Vision attn proj only = 48
```

没有实现。当前 Test 3 实际测试的是 full Vision，而不是 manual 中的 MLP-only Test 3。

---

# 我建议这样修

不要把“precision override”和“是否 wrap”混在一起。新增独立的 Vision group selection。

我建议 YAML 定义成：

```yaml
quantization:
  vision:
    enabled: true

    linear:
      mlp: true
      attn_proj: false

    quantize_matmul: false
    quantize_patch_embed: false

  connector:
    enabled: true
```

于是：

### V1

```yaml
vision:
  enabled: false

connector:
  enabled: true
```

新增：

```text
1 connector
```

### V2

```yaml
vision:
  enabled: true
  linear:
    mlp: true
    attn_proj: false

connector:
  enabled: true
```

新增：

```text
24 MLP + 1 Connector
```

总数：

$$
\boxed{249}
$$

### V3

```yaml
vision:
  enabled: true
  linear:
    mlp: true
    attn_proj: true
```

总数：

$$
\boxed{297}
$$

这会让实验阶段和代码结构严格对应。

对应代码建议：

```python
vision_cfg = quant_config.get("vision", {}) or {}
vision_enabled = bool(vision_cfg.get("enabled", False))

vision_linear_cfg = vision_cfg.get("linear", {}) or {}

wrap_vision_mlp = (
    vision_enabled
    and bool(vision_linear_cfg.get("mlp", False))
)

wrap_vision_attn_proj = (
    vision_enabled
    and bool(vision_linear_cfg.get("attn_proj", False))
)
```

然后：

```python
if vision_enabled:
    for layer_idx, layer in enumerate(encoder.layers):

        if wrap_vision_attn_proj:
            for name in VISION_ATTN_LINEAR_NAMES:
                ...

        if wrap_vision_mlp:
            for name in VISION_MLP_LINEAR_NAMES:
                ...
```

我建议这里**不要默认两个都是 true**。Phase I 是科研实验框架，宁可显式一点：

```yaml
mlp: true
attn_proj: false
```

比因为少写一个字段就意外量化整个 Vision Encoder 安全很多。

---

## P0.5：Vision wrapper 没有调用 `_should_wrap()`

现有 VLM/Expert 路径每个 Linear 都会：

```python
if not _should_wrap(module_id, quant_config):
    continue
```

而 `_should_wrap()` 会处理：

```yaml
linear:
  enabled:
  include:
  exclude:
```

但新 Vision `_replace_linear()` 完全没有调用它。

这会产生两个问题。

第一，现在即使写：

```yaml
linear:
  enabled: false
vision:
  enabled: true
```

Vision 仍然会被 wrap。

这和整个框架的：

```text
linear.enabled=false
→ 禁止所有 QuantizedLinear wrapping
```

语义不一致。

第二，你也无法使用：

```yaml
linear:
  exclude:
    - "vision.layers.*.self_attn.*"
```

来临时隔离 Vision attention。

所以 `_replace_linear()` 至少应增加：

```python
if not _should_wrap(module_id, quant_config):
    return
```

推荐顺序：

```python
def _replace_linear(...):
    if mod is None or not isinstance(mod, nn.Linear):
        return

    if not _should_wrap(module_id, quant_config):
        return

    ql = create_quantized_linear(...)
```

这样 Vision/Connector 才真正遵守原框架 contract。

但注意：**只加这一句还不能自动解决 V2 routing**，因为默认 `include=["*"]` 仍然会 wrap 72 个。还是需要前面那个 `mlp / attn_proj` group gate。

---

# P1：`vision.quantize_matmul` 目前实际上没有作用

手册里我们特意把两个开关分开：

```yaml
quantize_matmul: true          # VLM / Expert

vision:
  quantize_matmul: false       # Vision
```

原因就是 Vision attention 是 Transformers 自己的 attention interface，而 VLM/Expert 是 LeRobot 自己的 attention interface。官方 SmolVLM 现在通过：

```python
ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation,
    eager_attention_forward,
)
```

执行 Vision `QKᵀ/PV`。([GitHub][3])

但是当前提交中，Vision config 只读取了：

```python
vision_enabled = bool(vision_cfg.get("enabled", False))
```

没有读取：

```python
vision_cfg["quantize_matmul"]
```

而 `_inject_smolvla_quantized_matmul()` 仍然只建立：

```text
vlm   × 16 × qk/pv
expert× 16 × qk/pv
= 64
```

没有 Vision 的 24 个 MatMul。

因此目前实际状态应该明确写成：

```text
V1 Connector       plumbing：已实现
V2 Vision Linear   plumbing：部分实现
V3 Vision Linear   plumbing：部分实现
V4 Vision QK/PV：尚未实现
V5 Patch Conv2d：尚未实现
```

这本身没问题——分阶段做更安全——但不要现在拿：

```yaml
vision:
  quantize_matmul: true
```

去跑实验，因为它目前不会做任何事情。

README 目前也正确地仍然把 QK/PV 写作“未量化”，这一点和代码是吻合的。([GitHub][1])

---

# P1：Calibration policy 还有一个潜在碰撞问题

这个问题比表面更隐蔽。

现在：

```python
resolve_calibration_policy(
    quant_config,
    layer_type,
    layer_idx,
)
```

calibration policy 的 key 是类似：

```text
q_proj_0
q_proj_1
...
```

它**不知道 component**。

以前只有：

```text
vlm q_proj_0
expert q_proj_0
```

现在又多了：

```text
vision q_proj_0
```

但三者的 scale identity 在 `per_site` 下却分别是：

```text
vlm_q_proj_0
expert_q_proj_0
vision_q_proj_0
```

如果某个 G6 配置里有：

```yaml
calibration_policy:
  per_layer_policy:
    q_proj_0: reuse
```

那么新的：

```text
vision.layers.0.self_attn.q_proj
```

也会继承：

```text
reuse
```

但 `vision_q_proj_*` scale 文件可能根本还不存在。

结果可能是：

```text
scale_inspection:
    reuse → 不采 scale

quant_forward:
    尝试读取 vision_q_proj scale
    → 文件不存在
```

所以 V3 前一定要解决这个问题。

最简单的 Phase-I 方案，不必马上大改 calibration resolver：

```python
# wrapping vision
ql.calibration_policy = str(
    vision_cfg.get("calibration_policy", "recalibrate")
)
```

Connector 同理：

```python
connector:
  calibration_policy: recalibrate
```

这样 Vision 第一次加入时强制 recalibrate。

等整个 Phase I 稳定后，再考虑把 calibration policy 也升级成完整的 module-aware selector：

```yaml
calibration_policy:
  overrides:
    - target:
        component: vision
      policy: recalibrate
```

---

# P1：测试虽然加了，但还不能作为实验 Gate

当前测试文件值得保留，尤其这几项是正确的：

```text
default off                  ✅
connector only = 1           ✅
full vision = 72             ✅
vision + connector = 73      ✅
module ID uniqueness         ✅
module ID parser             ✅
override glob matching       ✅
```



但是实验手册要求的关键测试还缺：

```text
Vision MLP only         = 24       ❌
Vision attn proj only   = 48       ❌
raw forward equivalence          ❌
connector shape equivalence      ❌
real SmolVLM module smoke        ❌
Vision MatMul routing = 24       ❌（V4未做）
Legacy full ModelWrapper=224     ❌
```

手册本来明确要求 Test 3 为 MLP-only、Test 4 为 attention-only，并要求 raw forward equivalence。([GitHub][2])

尤其目前 mock：

```python
fc1: Linear(768,768)
fc2: Linear(768,768)
connector: Linear(768,768)
```

不是实际模型：

```text
fc1       768 → 3072
fc2      3072 → 768
connector 12288 → 960
```

所以 mock 很适合检查 routing，但**不能代替真实 shape/raw-equivalence smoke**。

---

# 还有一个实验流程问题：V0 还没有先做

手册规定：

> V0 workload audit PASS 后才修改主量化框架。

V0 应该产出：

```text
vision_structure.json
vision_runtime_shapes.json
vision_flops.csv
vision_flops_summary.md
```

并通过真实 LIBERO inference 测 camera call 数，不能把两相机写死。([GitHub][2])

你现在已经先把 wrapper 写了，这倒不用回滚，因为它默认关闭，**不会污染 V0**。

所以现在正确的处理方式是：

```text
代码保留
vision.enabled=false
connector.enabled=false
       ↓
现在补跑 V0
       ↓
确认真实 runtime structure/FLOPs
       ↓
再开启 V1
```

即可。

---

## 建议你这次先只改 4 件事

我建议不要继续扩 V4，先做一个小修复 commit：

1. 给 Vision 增加 `mlp` / `attn_proj` 独立 wrapping gate。
2. `_replace_linear()` 中调用 `_should_wrap()`。
3. Vision/Connector 第一次实验显式 `calibration_policy=recalibrate`。
4. 补齐 routing tests：

```text
default off             0
connector only          1
vision MLP only        24
vision attention only  48
full vision            72
vision + connector     73
```

对应测试最好直接变成：

```python
def _cfg(
    vision_enabled=False,
    vision_mlp=False,
    vision_attn_proj=False,
    connector_enabled=False,
    ...
):
    ...
```

然后必须断言：

```python
assert wrap(...) == 24
assert wrap(...) == 48
```

---

## 修完后的 Gate 顺序

下一轮我建议严格这样走：

```text
Gate L0
旧 canonical G6
→ 224 QuantizedLinear
→ 原 MatMul 数不变
→ 0 vision
→ 0 connector
        │
        ▼
V0
真实 LIBERO inference
→ camera calls
→ actual tensor shape
→ FLOPs CSV
        │
        ▼
V1-R
224 + 1 = 225
Connector wrapped RAW
→ raw equivalence
        │
        ▼
V1-FP8
Connector only
→ calibration
→ Task0 ×1
→ Goal ×100
        │
        ▼
V2-R
224 + 1 + 24 = 249
Vision MLP RAW
→ raw equivalence
        │
        ▼
V2-FP8
只改变 fc1/fc2 precision
```

**只有看到 `249`，才允许把结果称为 V2。**

如果现在直接按当前代码打开：

```yaml
vision:
  enabled: true
```

看到的应该是：

```text
297
```

那其实已经相当于“V3 full Vision Linear wrapping”，不是 V2。

### 当前审计结论

| 项目                                           | 状态             |
| -------------------------------------------- | -------------- |
| 官方 Vision 路径定位                               | ✅              |
| `q/k/v/out_proj` 命名                          | ✅              |
| `fc1/fc2` 命名                                 | ✅              |
| Connector 路径                                 | ✅              |
| Default-off backward compatibility           | ✅              |
| canonical module ID                          | ✅              |
| QuantizedLinear 复用                           | ✅              |
| parser 支持 vision/connector                   | ✅              |
| Connector-only routing                       | ✅              |
| **Vision MLP-only routing**                  | **❌ P0**       |
| **Vision attn-only routing**                 | **❌ P0**       |
| `linear.enabled/include/exclude` 对 Vision 生效 | **❌**          |
| Vision calibration 独立性                       | **⚠️**         |
| V0 runtime workload audit                    | **待跑**         |
| V4 Vision QK/PV                              | **未实现，符合当前阶段** |
| V5 Conv2d                                    | **未实现，符合当前阶段** |

所以这次提交**不用推翻**。主体设计是正确的，问题集中在一个概念上：

> **现在把 `vision.enabled` 当成了“wrap 整个 Vision Linear”，而实验设计需要的是“Vision component enabled + operator group 可选择”。**

先把这个修掉，整个 V1→V2→V3 的实验链就会干净很多。之后再把新 commit 推上来，我建议下一次重点审这几个 count 和 calibration routing，不必马上碰 V4。

[1]: https://github.com/wolf111108/SmolVLA_qtrsc?rev=20260915T0526 "GitHub - wolf111108/SmolVLA_qtrsc · GitHub"
[2]: https://github.com/wolf111108/SmolVLA_qtrsc/blob/main/2026-09-15_phaseI_vision_quantization_experiment_manual.md "SmolVLA_qtrsc/2026-09-15_phaseI_vision_quantization_experiment_manual.md at main · wolf111108/SmolVLA_qtrsc · GitHub"
[3]: https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/modeling_smolvlm.py?utm_source=chatgpt.com "transformers/src/transformers/models/smolvlm/modeling_smolvlm.py at main · huggingface/transformers · GitHub"
