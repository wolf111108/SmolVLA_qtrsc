# SmolVLA 细粒度 Quantization / Sensitivity 修改方案

> 目标：在不修改 `lerobot_current` 上游源码的前提下，把当前 **Linear / MatMul 的“物理算子身份”与“scale 共享关系”彻底解耦**，从而同时支持：
>
> - VLM / Action Expert 独立 scale；
> - 每层独立 scale；
> - QK / PV 独立 scale；
> - global / component / layer / site 多种 scale granularity；
> - 精确到 `component → layer → operator → site` 的 noise sensitivity；
> - 保留当前“全局共享 scale 仍近似无损”的实验能力，作为正式 ablation，而不是由命名碰撞或共享对象偶然实现。

---

# 1. 当前问题

## 1.1 Linear：物理层身份和 scale 文件身份混在一起

当前 Linear 主要通过：

```text
layer_name + layer_idx
```

决定：

- calibration statistics key；
- scale 文件名；
- tensor dump；
- SQNR 日志；
- quant/test target。

对于：

```text
VLM layer 0 q_proj
Expert layer 0 q_proj
```

二者都可能落成：

```text
q_proj_0
```

因此会出现：

```text
VLM q_proj_0
       ┐
       ├── q_proj_a_scale_0.p
       ├── q_proj_w_scale_0.p
       └── q_proj_o_scale_0.p
Expert q_proj_0
```

这不是“可控 scale sharing”，而是 **identity aliasing**。

---

## 1.2 MatMul：所有 attention 共用一对 qk/pv quantizer

当前结构近似：

```text
all attention sites
       │
       ├── shared qk_matmul
       └── shared pv_matmul
```

因此 scale 文件只有：

```text
qk: A / B / O
pv: A / B / O
```

共 6 个。

这意味着：

```text
physical operator identity
=
scale-sharing identity
=
同一个 QuantizedMatMul object
```

无法进行：

- VLM vs Expert；
- layer 0 vs layer 15；
- QK vs PV per-layer；
- 单个 MatMul site sensitivity。

---

# 2. 总体设计原则

核心原则：

\[
\boxed{
\text{Physical Operator Identity}
\neq
\text{Scale Sharing Identity}
}
\]

即每个真正的算子都有唯一 `module_id`，但可以按配置决定它与谁共享 scale。

建议所有 Linear / MatMul 都统一拥有两套 identity。

---

## 2.1 Physical identity：`module_id`

用于：

- sensitivity targeting；
- test/noise injection；
- tensor dump；
- SQNR；
- call count；
- debug；
- layer/operator heatmap；
- 唯一标识真实计算位置。

例如：

```text
vlm.layers.0.self_attn.q_proj
expert.layers.0.self_attn.q_proj

vlm.layer.7.qk
vlm.layer.7.pv

expert.layer.7.qk
expert.layer.7.pv
```

`module_id` **永远唯一，不参与 scale sharing**。

---

## 2.2 Scale identity：`scale_group_name + scale_group_idx`

只用于：

- calibration 聚合；
- scale persistence；
- quant_forward scale 加载；
- 控制 scale 是否共享。

例如：

```text
module_id:
    expert.layer.7.qk

scale group:
    expert_qk_matmul_7
```

表示完全独立。

也可以：

```text
module_id:
    expert.layer.7.qk

scale group:
    qk_matmul_0
```

表示全局共享。

---

# 3. 目标 granularity

MatMul 建议至少支持以下四档。

| Granularity | QK/PV scale sharing | Scale 文件数 |
|---|---|---:|
| `global` | 所有 VLM/Expert 全共享 | 6 |
| `per_component` | VLM 一套、Expert 一套 | 12 |
| `per_layer` | VLM/Expert 相同 `layer_idx` 共享 | 96 |
| `per_site` | component + layer 完全独立 | 192 |

计算方式：

### global

```text
2 operators × 3 scales = 6
```

### per_component

```text
2 components × 2 operators × 3 scales = 12
```

### per_layer

```text
16 layers × 2 operators × 3 scales = 96
```

VLM/Expert 同 index 仍共享。

### per_site

```text
2 components
× 16 layers
× 2 operators
× 3 scales
= 192
```

这是完全不共享 scale 的配置。

---

# 4. 目标运行时结构

无论 scale granularity 是什么，**永远创建 64 个物理 MatMul object**：

```text
VLM:
    layer0.qk
    layer0.pv
    ...
    layer15.qk
    layer15.pv

Expert:
    layer0.qk
    layer0.pv
    ...
    layer15.qk
    layer15.pv
```

总数：

\[
2\times16\times2=64
\]

推荐：

```python
attention_module.quant_matmuls = nn.ModuleDict()
```

而不是根据 granularity 改物理 object 数量。

这样：

```text
Sensitivity identity
```

永远稳定。

只有：

```text
Scale group
```

随配置变化。

---

# 5. 修改 `QuantizedLinear`

文件：

```text
src/vla_tcs2/quant_linear.py
```

---

## 5.1 新增唯一 `module_id`

建议新增：

```python
self.module_id: str = ""
```

接口：

```python
def set_layer_info(
    self,
    layer_name: str,
    layer_idx: int,
    module_id: str | None = None,
):
    self.layer_name = layer_name
    self.layer_idx = layer_idx

    self.module_id = (
        module_id
        if module_id is not None
        else f"{layer_name}_{layer_idx}"
    )
```

例如 wrapper 中赋值：

```text
vlm.layers.3.self_attn.q_proj
expert.layers.3.self_attn.q_proj
```

---

## 5.2 Linear scale identity 也建议独立出来

新增：

```python
self.scale_group_name: str = ""
self.scale_group_idx: int = 0
```

以及：

```python
def set_scale_group(
    self,
    name: str,
    idx: int,
):
    self.scale_group_name = name
    self.scale_group_idx = idx
```

第一阶段可以直接：

```text
scale_group == module_id
```

即所有 Linear 独立。

以后如果想测试：

```text
VLM/Expert Linear shared scale
```

只改 scale group mapping，不再制造命名 collision。

---

## 5.3 scale 文件改用 scale group

原来的：

```text
q_proj_a_scale_0.p
```

改成类似：

```text
vlm_q_proj_a_scale_0.p
expert_q_proj_a_scale_0.p
```

或者进一步 sanitize 完整 module path。

原则：

```text
file identity = scale_group
```

而不是：

```text
file identity = layer_name + layer_idx
```

---

## 5.4 dump / SQNR / test statistics 改用 `module_id`

以下内容必须用 `module_id`：

```text
activation dump
weight dump
output dump
SQNR
test noise statistics
call count
sensitivity result
```

避免再次出现：

```text
VLM down_proj_0
Expert down_proj_0
```

相互覆盖。

---

# 6. 修改 `QuantizedMatMul`

文件：

```text
src/vla_tcs2/quant_matmul.py
```

---

## 6.1 新增 `module_id`

例如：

```python
self.module_id = ""
```

实际值：

```text
vlm.layer.7.qk
vlm.layer.7.pv
expert.layer.7.qk
expert.layer.7.pv
```

---

## 6.2 新增 scale group

```python
self.scale_group_name = ""
self.scale_group_idx = 0
```

接口：

```python
def set_scale_group(self, name, idx):
    self.scale_group_name = name
    self.scale_group_idx = idx
```

---

## 6.3 `_scale_file_paths()` 改为 scale group

示意：

```python
def _scale_file_paths(self):
    name = (
        self.scale_group_name
        or self.layer_name
    )
    idx = self.scale_group_idx

    return (
        f"{name}_A_scale_{idx}.p",
        f"{name}_B_scale_{idx}.p",
        f"{name}_O_scale_{idx}.p",
    )
```

---

## 6.4 calibration statistics 同样使用 scale group

不要再：

```python
collect_matmul_stats(
    self.layer_name,
    self.layer_idx,
    ...
)
```

改为：

```python
collect_matmul_stats(
    self.scale_group_name,
    self.scale_group_idx,
    ...
)
```

这样：

### `global`

64 个 physical MatMul 可以全部向：

```text
qk_matmul_0
pv_matmul_0
```

聚合。

### `per_site`

每个 MatMul 向独立 key 聚合。

---

# 7. 修改 `StatManager`

文件：

```text
src/vla_tcs2/quant/stat_manager.py
```

StatManager 要明确区分两个概念：

```text
module statistics
scale statistics
```

---

## 7.1 Scale statistics

继续按：

```text
scale_group_name + scale_group_idx
```

聚合。

用于：

```text
max calibration scale
scale persistence
```

---

## 7.2 Module statistics

另建：

```text
module_id
```

用于：

```text
call_count
SQNR
noise stats
sensitivity stats
tensor-level diagnostics
```

不要用 scale group 代替 module identity。

---

# 8. `model_wrapper.py` 是主要改造位置

文件：

```text
src/vla_tcs2/model_wrapper.py
```

主要负责四件事：

1. 给所有 Linear 唯一 `module_id`；
2. 创建 64 个独立 MatMul；
3. 根据 granularity 分配 scale group；
4. runtime routing 到正确 MatMul。

---

# 9. Linear wrapper 修改

当前创建 Linear 时不要只传：

```text
name
layer_idx
```

还需要传完整路径。

例如：

```python
module_id = (
    f"vlm.layers.{i}."
    f"self_attn.{name}"
)
```

或者：

```python
module_id = (
    f"expert.layers.{i}."
    f"self_attn.{name}"
)
```

MLP 同理：

```text
vlm.layers.3.mlp.down_proj
expert.layers.3.mlp.down_proj
```

---

# 10. MatMul bank 创建

建议：

```python
attention_module.quant_matmuls = nn.ModuleDict()
```

初始化：

```python
for component in ("vlm", "expert"):
    for layer_idx in range(16):

        qk = create_quantized_matmul(...)
        pv = create_quantized_matmul(...)

        qk.module_id = (
            f"{component}.layer.{layer_idx}.qk"
        )

        pv.module_id = (
            f"{component}.layer.{layer_idx}.pv"
        )

        attention_module.quant_matmuls[
            f"{component}_qk_{layer_idx}"
        ] = qk

        attention_module.quant_matmuls[
            f"{component}_pv_{layer_idx}"
        ] = pv
```

共 64 个。

---

# 11. Scale-group resolver

建议新增统一函数：

```python
def resolve_matmul_scale_group(
    component: str,
    layer_idx: int,
    op: str,
    granularity: str,
):
    if granularity == "global":
        return (
            f"{op}_matmul",
            0,
        )

    if granularity == "per_component":
        return (
            f"{component}_{op}_matmul",
            0,
        )

    if granularity == "per_layer":
        return (
            f"layer_{op}_matmul",
            layer_idx,
        )

    if granularity == "per_site":
        return (
            f"{component}_{op}_matmul",
            layer_idx,
        )

    raise ValueError(granularity)
```

---

# 12. YAML 新增 granularity

建议：

```yaml
quantization:
  quantize_matmul: true

  matmul_scale_granularity: per_site
```

可选：

```text
global
per_component
per_layer
per_site
```

默认建议：

```text
per_site
```

因为这是“正确、不共享”的 baseline。

以后其他模式都作为 scale-sharing ablation。

---

# 13. Runtime routing：不修改 LeRobot 源码

核心难点是：

```text
get_attention_interface()
```

没有 `layer_idx` 参数。

解决方法：

```text
ContextVar + runtime proxy
```

而不是修改：

```text
lerobot_current
```

---

# 14. 新增 Attention Context

在 `model_wrapper.py`：

```python
from contextvars import ContextVar
```

定义：

```python
_CURRENT_ATTN_SITE = ContextVar(
    "smolvla_attn_site",
    default=None,
)
```

保存：

```text
(component, layer_idx)
```

例如：

```text
("vlm", 7)
("expert", 7)
```

---

# 15. Wrap `forward_attn_layer`

保存原始 bound method：

```python
original_forward_attn_layer = (
    attention_module.forward_attn_layer
)
```

proxy 逻辑：

```python
def proxy(..., inputs_embeds, layer_idx, ...):

    if prefix exists and suffix is None:
        component = "vlm"

    elif prefix is None and suffix exists:
        component = "expert"

    else:
        component = "joint"

    token = _CURRENT_ATTN_SITE.set(
        (component, layer_idx)
    )

    try:
        return original_forward_attn_layer(...)
    finally:
        _CURRENT_ATTN_SITE.reset(token)
```

---

# 16. Wrap `forward_cross_attn_layer`

deployment / inference denoise path 下：

```text
component = expert
```

因此：

```python
token = _CURRENT_ATTN_SITE.set(
    ("expert", layer_idx)
)
```

然后调用原始 method。

同样：

```python
finally:
    reset(token)
```

防止异常后污染 runtime state。

---

# 17. Dispatcher

`get_attention_interface()` 继续保持无参数。

你的 custom attention interface 内：

```python
component, layer_idx = (
    _CURRENT_ATTN_SITE.get()
)

qk = attention_module.quant_matmuls[
    f"{component}_qk_{layer_idx}"
]

pv = attention_module.quant_matmuls[
    f"{component}_pv_{layer_idx}"
]
```

之后：

```python
att_weights = qk(
    query_states,
    key_states.transpose(2, 3),
)
```

和：

```python
att_output = pv(
    probs,
    value_states.permute(0, 2, 1, 3),
)
```

其余 attention 数学完全不改。

---

# 18. Runtime routing 数据流

## VLM prefix

```text
predict_action_chunk
        │
        ▼
inputs_embeds=[prefix, None]
        │
        ▼
forward_attn_layer(layer=7)
        │
        ▼
CURRENT_ATTN_SITE=("vlm", 7)
        │
        ▼
attention dispatcher
        │
        ├── vlm_qk_7
        └── vlm_pv_7
```

---

## Action Expert denoise

```text
denoise_step
        │
        ▼
inputs_embeds=[None, suffix]
        │
        ▼
forward_*_attn_layer(layer=7)
        │
        ▼
CURRENT_ATTN_SITE=("expert", 7)
        │
        ▼
attention dispatcher
        │
        ├── expert_qk_7
        └── expert_pv_7
```

---

# 19. `test_forward` / sensitivity 接入

已有：

```python
test_forward(...)
```

以及：

```text
test_methods.py
```

之后所有 sensitivity targeting 都基于：

```text
module_id
```

---

## 19.1 Linear sensitivity

例如：

```text
expert.layers.12.mlp.down_proj
```

配置：

```text
mode = test_forward
test_method = gaussian_rms_output
alpha = 0.03
```

其他 Linear：

```text
raw
```

即可测：

> Expert layer 12 down_proj output sensitivity。

---

## 19.2 MatMul sensitivity

例如：

```text
expert.layer.7.qk
```

只把这个 physical MatMul：

```text
mode = test_forward
```

其他 63 个 MatMul：

```text
raw
```

即可严格测：

> Expert layer 7 QK sensitivity。

即使：

```yaml
matmul_scale_granularity: global
```

也仍然能单独测 sensitivity。

因为 physical identity 和 scale sharing 已经解耦。

---

# 20. 建议 sensitivity target 配置

建议新增：

```yaml
test:
  enabled: true

  method: gaussian_rms_output

  alpha: 0.03
  seed: 0

  target:
    module_id: expert.layer.7.qk
```

以后可以扩展 selector：

```yaml
target:
  component: expert
  layer: 7
  operator: qk
```

或者：

```yaml
target:
  component: vlm
  operator: down_proj
  layers: [0, 1, 2, 3]
```

---

# 21. Mode switching

由于 Linear 和 64 个 MatMul 都是正式 `nn.Module`，现有：

```python
for module in model.modules():
```

应该可以统一切换：

```text
raw
scale_inspection
quant_forward
test_forward
```

要求：

```text
MatMul bank 必须使用 nn.ModuleDict / nn.ModuleList
```

不能用普通 Python dict/list。

---

# 22. Scale 文件预期

## global

```text
qk_matmul_A_scale_0.p
qk_matmul_B_scale_0.p
qk_matmul_O_scale_0.p

pv_matmul_A_scale_0.p
pv_matmul_B_scale_0.p
pv_matmul_O_scale_0.p
```

共：

```text
6
```

---

## per_component

```text
vlm_qk_matmul_*
vlm_pv_matmul_*

expert_qk_matmul_*
expert_pv_matmul_*
```

共：

```text
12
```

---

## per_site

layer 7：

```text
vlm_qk_matmul_A_scale_7.p
vlm_qk_matmul_B_scale_7.p
vlm_qk_matmul_O_scale_7.p

vlm_pv_matmul_A_scale_7.p
vlm_pv_matmul_B_scale_7.p
vlm_pv_matmul_O_scale_7.p

expert_qk_matmul_A_scale_7.p
expert_qk_matmul_B_scale_7.p
expert_qk_matmul_O_scale_7.p

expert_pv_matmul_A_scale_7.p
expert_pv_matmul_B_scale_7.p
expert_pv_matmul_O_scale_7.p
```

16 层共：

```text
192
```

---

# 23. Calibration

完成上述修改后必须重新 calibration。

旧 scale：

```text
6-file shared MatMul
```

不能直接用于：

```text
per_site
```

除非显式配置：

```text
global
```

---

## 推荐 scale_dir 按 granularity 分开

例如：

```text
scales/
  fp8_pot_matmul_global/
  fp8_pot_matmul_component/
  fp8_pot_matmul_layer/
  fp8_pot_matmul_site/
```

禁止不同 granularity 混用。

---

# 24. 必须增加 preflight audit

正式运行前检查：

```text
physical MatMul count
module_id unique count
scale-group count
expected scale file count
```

---

## per_site 应满足

```text
QuantizedMatMul physical modules = 64
unique module_id = 64
unique scale groups = 64
scale files = 64 × 3 = 192
```

---

## global 应满足

```text
physical modules = 64
unique module_id = 64
unique scale groups = 2
scale files = 2 × 3 = 6
```

---

# 25. Call-count audit

必须给 64 个 MatMul 独立记录：

```text
module_id → call_count
```

例如一次 rollout 后应该看到：

```text
vlm.layer.0.qk        > 0
vlm.layer.0.pv        > 0
...
expert.layer.15.qk    > 0
expert.layer.15.pv    > 0
```

这样可以直接证明：

> per-site MatMul 不是只创建了对象，而是真的被执行。

---

# 26. Raw-equivalence test

在启用新 routing 后，首先全部设：

```text
mode = raw
```

比较：

```text
原 SmolVLA
vs
新 MatMul dispatcher
```

要求：

```text
action output numerically equal / nearly equal
```

建议：

```text
max_abs_error
mean_abs_error
relative_error
```

都接近浮点计算误差。

如果 raw mode 已经改变输出，先不能做 quantization。

---

# 27. Quantization sanity test

建议按以下顺序：

### Step 1

```text
Linear raw
MatMul per_site raw
```

验证 routing。

### Step 2

```text
Linear raw
MatMul per_site FP8
```

验证 192 scale。

### Step 3

```text
Linear FP8
MatMul per_site FP8
```

验证完整量化。

### Step 4

切：

```text
per_site → per_layer → per_component → global
```

做 granularity ablation。

---

# 28. 预期正式 ablation

建议最终表：

| MatMul scale granularity | Scale files | Object SR |
|---|---:|---:|
| per-site | 192 | TBD |
| per-layer | 96 | TBD |
| per-component | 12 | TBD |
| global | 6 | 当前约 93% |

这样可以把当前偶然发现的：

> 6 个 scale 仍近似无损

正式转化为：

\[
\boxed{
\text{Scale metadata reduction}
\leftrightarrow
\text{Closed-loop SR}
}
\]

的实验。

---

# 29. Sensitivity 实验顺序

底层 identity 修好后建议：

### Component

```text
VLM
vs
Expert
```

### Layer

```text
VLM layer 0~15
Expert layer 0~15
```

### Operator

```text
q_proj/k_proj/v_proj/o_proj
qk/pv
gate/up/down
```

### Tensor site

Linear：

```text
input
weight
output
```

MatMul：

```text
A
B
output
```

最终得到：

```text
component
→ layer
→ operator
→ tensor site
```

四级 sensitivity map。

---

# 30. Deployment / Training scope

该方案优先针对当前 PTQ / evaluation inference path：

```text
predict_action_chunk
```

其中：

```text
VLM prefix:
    [prefix, None]

Expert denoise:
    [None, suffix]
```

因此：

```text
component = vlm / expert
```

可以清楚识别。

---

## Training-time caveat

训练时可能：

```text
[prefix, suffix]
```

同时存在。

并可能出现：

```text
joint self attention
vlm prefix attention
expert cross attention
```

此时单纯：

```text
(component, layer_idx)
```

不足以唯一描述所有 attention call site。

如果未来要支持 QAT / training-time quantization，需要把 identity 扩展成：

```text
(attention_role, component, layer_idx)
```

例如：

```text
joint_self
vlm_prefix
expert_cross
```

当前 PTQ / LIBERO evaluation 阶段可以暂不实现。

---

# 31. 推荐实施顺序

## Phase A — Identity 修复

1. Linear 增加 `module_id`；
2. 修复 VLM/Expert Linear scale collision；
3. dump/SQNR/test 全改用 `module_id`。

---

## Phase B — MatMul physical split

1. 创建 64 个 MatMul；
2. `ModuleDict` 注册；
3. ContextVar routing；
4. raw-equivalence test；
5. call-count audit。

---

## Phase C — Scale group 解耦

1. `scale_group_name / idx`；
2. StatManager 按 scale group 聚合；
3. scale file 按 scale group 保存；
4. 实现四种 granularity。

---

## Phase D — Calibration / Quantization 验证

依次验证：

```text
per_site
per_layer
per_component
global
```

确保 scale 文件数量分别正确。

---

## Phase E — Sensitivity

基于唯一 `module_id`：

```text
component
layer
operator
site
```

进行 Gaussian / OP / quant residual 实验。

---

# 32. 最终架构图

```text
                     SmolVLA
                        │
        ┌───────────────┴────────────────┐
        │                                │
      Linear                           MatMul
        │                                │
  unique module_id                 64 physical sites
        │                                │
        │                         unique module_id
        │                                │
        └──────────────┬─────────────────┘
                       │
                sensitivity/test
                 永远按 module_id
                       │
                       ▼
             ┌──────────────────┐
             │ scale group map  │
             └──────────────────┘
                       │
       ┌───────────────┼───────────────┬───────────────┐
       │               │               │               │
     global       per_component     per_layer        per_site
       │               │               │               │
   6 MatMul         12 MatMul        96 MatMul       192 MatMul
   scale files      scale files      scale files      scale files
```

注意：

```text
physical MatMul object 数始终 = 64
```

变化的只是：

```text
scale group 数
```

---

# 33. 修改完成后的核心收益

完成后可以同时回答：

1. **SmolVLA 是否需要 per-layer scale？**
2. **VLM 与 Expert 是否可以共享 scale？**
3. **QK/PV 是否可以全模型共享 scale？**
4. **6 个 MatMul scale 与 192 个 scale 的 SR 差多少？**
5. **VLM 与 Action Expert 谁更抗噪？**
6. **哪一层最敏感？**
7. **Linear 与 MatMul 谁更敏感？**
8. **QK 与 PV 谁更敏感？**
9. **哪些层可以进一步降到 FP6/FP4？**

---

# 34. 最关键的一句话

本次修改不应该只解决“scale 文件重名”。

真正应该建立的新抽象是：

\[
\boxed{
\text{module\_id}
\quad\text{负责“算子是谁”}
}
\]

\[
\boxed{
\text{scale\_group}
\quad\text{负责“和谁共享 scale”}
}
\]

这样才能同时支持：

```text
不共享 scale
可控共享 scale
per-layer sensitivity
operator-level sensitivity
hardware metadata ablation
```

并避免之后再次因为文件名、layer index 或 shared object 结构限制实验能力。
