# SmolVLA_qtrsc：Component-aware Linear Precision Routing 升级与 G5 重实验手册

> 日期：2026-09-14  
> 适用仓库：`wolf111108/SmolVLA_qtrsc`，`main`  
> 目标：在同一次 run 中支持 **VLM Linear 与 Expert Linear 使用不同精度**，并在升级后重新验证 G5 的 VLM selective-precision 结论。  
> 原则：**不修改量化数学内核，只升级 precision routing；先做路由/回归 Gate，再做闭环。**

---

## 0. 一句话结论

当前框架已经能：

- 按 `module_id` 选择是否 wrap；
- 分辨 `vlm.*` 与 `expert.*`；
- 为每个 physical Linear 使用独立 `per_site` scale；
- 对 Linear 执行 FP8 E4M3 或 A8W4O8；
- 对 QK/PV MatMul 执行 FP8；

但 **precision 配置仍由全局 operator block 决定**：

```yaml
q_proj:
  w_bit: ...
gate_proj:
  w_bit: ...
```

因此当前无法在同一次 run 中表达：

```text
VLM q_proj    = W4
Expert q_proj = FP8
```

本次升级增加：

```yaml
quantization:
  linear:
    include:
      - "vlm.*"
      - "expert.*"
    overrides:
      - name: vlm_attn_w4
        target:
          module_id: "vlm.layers.*.self_attn.*_proj"
        config:
          w_bit: 4
          method: pot_ao_outlier
```

使 precision resolution 变成：

```text
全局 method/default
        ↓
operator base config
        ↓
module_id/component/layer/operator override
        ↓
QuantizedLinear
```

升级后重新做一组 **G6：Expert-FP8 background 下的 VLM selective precision**：

| 组 | VLM Attention | VLM MLP | Expert Linear | QK/PV |
|---|---|---|---|---|
| G6-A | FP8 | FP8 | FP8 | FP8 |
| G6-B | W4 | FP8 | FP8 | FP8 |
| G6-C | FP8 | W4 | FP8 | FP8 |
| G6-D | W4 | W4 | FP8 | FP8 |

这不是覆盖旧 G5，而是验证：

> G5 得到的 `MLP > Attention` W4 sensitivity，在 **Expert 也量化为 FP8** 的实际部署背景下是否仍成立。

---

# 1. 当前代码状态与限制

## 1.1 当前 `create_quantized_linear()` 的 precision 来源

当前 `src/vla_tcs2/model_wrapper.py` 中：

```python
layer_config = quant_config.get(layer_type, {})
```

其中 `layer_type` 只有：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

随后：

```python
a_bit = layer_config.get("a_bit", 8)
w_bit = layer_config.get("w_bit", 8)
o_bit = layer_config.get("o_bit", 8)
```

因此配置：

```yaml
q_proj:
  w_bit: 4
```

会作用于**所有被 wrap 的 `q_proj`**。

如果同时：

```yaml
linear:
  include:
    - "vlm.*"
    - "expert.*"
```

那么：

```text
VLM q_proj    → W4
Expert q_proj → W4
```

目前不能把二者分开。

---

## 1.2 但是 wrapper 已经知道 physical identity

当前 `_wrap_smolvla_linear_layers()` 已经生成：

```text
vlm.layers.0.self_attn.q_proj
vlm.layers.0.mlp.gate_proj
...
expert.layers.0.self_attn.q_proj
expert.layers.0.mlp.gate_proj
...
```

即：

```python
module_id = f"{component}.layers.{layer_idx}.self_attn.{name}"
```

和：

```python
module_id = f"{component}.layers.{layer_idx}.mlp.{name}"
```

并且已有 `_matches_target()` 支持：

- `module_id`：exact / glob；
- `module_ids`：多个 glob；
- `component`：`vlm` / `expert`；
- `layer`：单层或 layer list；
- `operator`：q/k/v/o/gate/up/down 等。

所以**不需要重新设计 selector system**，只需把它从 sensitivity test 复用到 quantization routing。

---

## 1.3 Scale sharing 不是当前瓶颈

当前：

```yaml
linear_scale_granularity: per_site
```

对应：

```text
vlm_q_proj layer0
expert_q_proj layer0
```

使用不同 scale group。

所以只要继续固定：

```yaml
linear_scale_granularity: per_site
```

则：

```text
VLM q_proj W4 scale
Expert q_proj FP8 scale
```

不会相互污染。

**本次升级 v1 强制 component-aware override 只能与 `per_site` 一起使用。**

这样避免同时引入“mixed precision + scale sharing compatibility”的第二个变量。

---

# 2. 升级目标与非目标

## 2.1 必须实现

升级后必须能表达：

### Case A：VLM Attention W4，其他 Linear FP8

```text
VLM q/k/v/o      = A8W4O8
VLM gate/up/down = A8W8O8
Expert all       = A8W8O8
```

### Case B：VLM MLP W4，其他 Linear FP8

```text
VLM q/k/v/o      = A8W8O8
VLM gate/up/down = A8W4O8
Expert all       = A8W8O8
```

### Case C：Expert W4，VLM FP8

```text
VLM all    = A8W8O8
Expert all = A8W4O8
```

这就是后续 Phase H 真正想要的 deployment candidate。

---

## 2.2 本次不做

本次不要同时扩展：

- group-wise W4；
- component-specific MatMul precision；
- action head quantization；
- vision encoder quantization；
- 新量化 kernel；
- 新 scale method；
- G3 action horizon；
- G4 dynamic outlier mask。

原因：本次目标只是把 **precision routing 粒度** 从 operator-global 提升到 physical-module-aware。

---

# 3. 推荐的新 YAML 语义

## 3.1 基础 operator config 继续保留

例如默认全部 Linear FP8：

```yaml
quantization:
  method: pot_fp8_outlier

  q_proj:
    a_bit: e4m3
    w_bit: e4m3
    o_bit: e4m3
    d_bit: 4
    outlier_ratio: 0.01
    p: 4

  # k/v/o/gate/up/down 同理
```

这叫 **base operator config**。

---

## 3.2 新增 `linear.overrides`

建议格式：

```yaml
quantization:
  linear:
    enabled: true
    include:
      - "vlm.*"
      - "expert.*"
    exclude: []

    overrides:
      - name: vlm_attn_w4
        target:
          module_id: "vlm.layers.*.self_attn.*_proj"
        config:
          w_bit: 4
          method: pot_ao_outlier
```

### 设计理由

`module_id` glob 已经能一次覆盖：

```text
vlm.layers.0.self_attn.q_proj
vlm.layers.0.self_attn.k_proj
...
vlm.layers.15.self_attn.o_proj
```

不需要为 q/k/v/o 写四条 override。

MLP：

```yaml
target:
  module_id: "vlm.layers.*.mlp.*_proj"
```

全 VLM：

```yaml
target:
  module_id: "vlm.layers.*.*.*_proj"
```

全 Expert：

```yaml
target:
  module_id: "expert.layers.*.*.*_proj"
```

---

## 3.3 Override 允许修改的字段

v1 只允许：

```text
a_bit
w_bit
o_bit
d_bit
p
outlier_ratio
method
test_method
```

未知字段直接报错，避免 YAML typo 静默被忽略。

---

## 3.4 配置优先级

定义：

```text
1. operator base config
2. matching override #0
3. matching override #1
...
N. 最后一个 matching override
```

即：

> **后匹配的 override 覆盖前面的字段。**

例如：

```yaml
overrides:
  - name: all_vlm_w4
    target:
      component: vlm
    config:
      w_bit: 4
      method: pot_ao_outlier

  - name: keep_vlm_layer0_fp8
    target:
      module_id: "vlm.layers.0.*.*"
    config:
      w_bit: e4m3
      method: pot_fp8_outlier
```

则 layer0 最终 FP8。

但是 G6 不需要 stacking；G6 每个 config 只用一条 override，降低歧义。

---

# 4. Core code 升级方案

主要只改：

```text
src/vla_tcs2/model_wrapper.py
```

`quant_linear.py / scale_methods.py / quant_methods.py` 不应因本次 routing 升级而改变。

---

## 4.1 新增 config resolver

建议放在 `_matches_target()` 后面：

```python
_ALLOWED_LINEAR_OVERRIDE_KEYS = {
    "a_bit",
    "w_bit",
    "o_bit",
    "d_bit",
    "p",
    "outlier_ratio",
    "method",
    "test_method",
}


def resolve_linear_quant_config(
    quant_config: dict[str, Any],
    layer_type: str,
    module_id: str,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve effective quant config for one physical Linear.

    Precedence:
        operator base config
        -> matching linear.overrides in list order (later wins)

    Returns:
        resolved_config
        matched_override_names
    """
    base = dict(quant_config.get(layer_type, {}) or {})

    linear_cfg = quant_config.get("linear", {}) or {}
    overrides = linear_cfg.get("overrides", []) or []

    if not isinstance(overrides, list):
        raise TypeError("quantization.linear.overrides must be a list")

    matched_names: list[str] = []

    for idx, override in enumerate(overrides):
        if not isinstance(override, dict):
            raise TypeError(
                f"linear.overrides[{idx}] must be a dict"
            )

        target = override.get("target")
        patch = override.get("config")

        if not isinstance(target, dict) or not target:
            raise ValueError(
                f"linear.overrides[{idx}].target must be a non-empty dict"
            )

        if not isinstance(patch, dict):
            raise ValueError(
                f"linear.overrides[{idx}].config must be a dict"
            )

        unknown = set(patch) - _ALLOWED_LINEAR_OVERRIDE_KEYS
        if unknown:
            raise ValueError(
                f"Unsupported linear override fields at index {idx}: "
                f"{sorted(unknown)}"
            )

        if _matches_target(module_id, target):
            base.update(patch)
            matched_names.append(
                str(override.get("name", f"override_{idx}"))
            )

    return base, matched_names
```

---

## 4.2 修改 `create_quantized_linear()` signature

旧：

```python
def create_quantized_linear(
    original_layer,
    layer_type,
    layer_idx,
    quant_config,
    mode="scale_inspection",
):
```

新：

```python
def create_quantized_linear(
    original_layer,
    layer_type,
    layer_idx,
    quant_config,
    mode="scale_inspection",
    module_id: str | None = None,
):
```

然后把：

```python
layer_config = quant_config.get(layer_type, {})
```

替换为：

```python
physical_id = module_id or f"{layer_type}_{layer_idx}"

layer_config, matched_overrides = resolve_linear_quant_config(
    quant_config=quant_config,
    layer_type=layer_type,
    module_id=physical_id,
)
```

QuantizedLinear 创建完成后记录：

```python
quant_layer.quant_override_names = matched_overrides
quant_layer.effective_quant_config = dict(layer_config)
```

这两个字段只用于 debug / manifest，不进入 forward。

---

## 4.3 三个 call site 都传 `module_id`

Attention：

```python
ql = create_quantized_linear(
    mod,
    name,
    layer_idx,
    quant_config,
    mode,
    module_id=module_id,
)
```

MLP 同理。

misc/action-head 路径：

```python
ql = create_quantized_linear(
    mod,
    name,
    0,
    quant_config,
    mode,
    module_id=full_name,
)
```

虽然 G6 不 wrap action head，但必须保持函数完整。

---

## 4.4 v1 强制 `per_site`

在 `_wrap_smolvla_linear_layers()` 取得 granularity 后：

```python
overrides = (
    (quant_config.get("linear", {}) or {}).get("overrides", [])
    or []
)

if overrides and granularity != "per_site":
    raise ValueError(
        "component/module-aware linear overrides currently require "
        "linear_scale_granularity=per_site to prevent mixed-precision "
        "modules from sharing calibration scales."
    )
```

这样直接阻止：

```text
VLM q_proj W4
Expert q_proj FP8
```

却共享一个 scale group 的错误配置。

---

# 5. 不需要修改 calibration.py 的原因

G6 固定：

```yaml
linear_scale_granularity: per_site
```

因此 224 个 Linear 都有独立 scale identity。

Calibration 仍然：

```text
predict_action_chunk(calibration batch)
    ↓
VLM executes
    ↓
Expert executes
    ↓
每个 QuantizedLinear 自己使用 effective method/spec 收集 scale
    ↓
QuantStatManager 保存
```

G6 每个 config 使用独立：

```text
scale_dir
```

并且：

```yaml
calibration_policy:
  layer_policy:
    q_proj: recalibrate
    ...
```

所以全部 scale 重算。

**本轮禁止 reuse G5 scale。**

---

# 6. Upgrade 后必须新增的 routing audit

建议在新 task 下创建：

```text
experiments/2026-09-10_phaseG_w4-root-cause/
└── tasks/
    └── vlm-selective-expert-fp8/
        ├── configs/
        ├── scripts/
        │   ├── audit_routing.py
        │   ├── run_smoke.sh
        │   ├── run_goal.sh
        │   └── summarize.py
        └── docs/
```

创建命令：

```bash
cd ~/VLA_tcs2

bash experiments/new_experiment.sh \
  --task 2026-09-10_phaseG_w4-root-cause \
  vlm-selective-expert-fp8
```

---

## 6.1 `audit_routing.py` 必须检查

对每个 `QuantizedLinear` 导出：

```text
module_id
component
layer_idx
operator
a_bit
w_bit
o_bit
method
scale_group_name
scale_group_idx
override_names
```

CSV：

```text
outputs/2026-09-10_phaseG_w4-root-cause/
└── tasks/
    └── vlm-selective-expert-fp8/
        └── routing/
            └── <config_name>_routing.csv
```

---

## 6.2 G6 四组预期 routing count

### G6-A

```text
VLM FP8    = 112
Expert FP8 = 112
W4         = 0
Total Linear = 224
MatMul FP8 = 64
```

即：

```text
FP8 Linear = 224
W4 Linear  = 0
```

---

### G6-B

VLM Attention：

```text
16 layers × 4 proj = 64 W4
```

VLM MLP：

```text
16 × 3 = 48 FP8
```

Expert：

```text
112 FP8
```

所以：

```text
W4 Linear  = 64
FP8 Linear = 48 + 112 = 160
Total      = 224
```

---

### G6-C

VLM MLP：

```text
16 × 3 = 48 W4
```

其余：

```text
VLM Attention 64 FP8
Expert 112 FP8
```

所以：

```text
W4 Linear  = 48
FP8 Linear = 176
Total      = 224
```

---

### G6-D

VLM all：

```text
112 W4
```

Expert：

```text
112 FP8
```

所以：

```text
W4 Linear  = 112
FP8 Linear = 112
Total      = 224
```

---

## 6.3 MatMul 必须保持一致

所有四组：

```text
QuantizedMatMul = 64
```

即：

```text
VLM 16 × (QK + PV)    = 32
Expert 16 × (QK + PV) = 32
Total                  = 64
```

全部：

```text
A/B/O = E4M3
method = pot_fp8_outlier
```

---

# 7. Raw equivalence Gate

在任何 calibration / rollout 前，先检查 routing 升级没有破坏 Linear raw path。

对所有 224 个 wrapped Linear：

```python
module.mode = "raw"
```

固定随机输入：

```python
torch.manual_seed(1234)
```

比较：

```python
y1 = module(x)
y2 = F.linear(x, module.weight, module.bias)
```

Gate：

```text
FP32:
max_abs_diff == 0

若 bf16 kernel 存在实现差异：
max_abs_diff <= 1e-6
cosine >= 0.9999999
```

建议至少覆盖：

- q_proj；
- o_proj；
- gate_proj；
- down_proj；
- VLM；
- Expert；
- 2D input；
- 3D input；
- bias=True / False（如模型有）。

---

# 8. Config 设计：所有 operator 默认 FP8

G6 的关键做法是：

> **先把 VLM + Expert 全部 Linear wrap，并默认 FP8；再只用 override 把指定的 VLM 区域改 W4。**

共同部分：

```yaml
quantization:
  enabled: true
  mode: quant_forward

  method: pot_fp8_outlier

  quantize_matmul: true
  matmul_scale_granularity: per_site
  linear_scale_granularity: per_site

  outlier_ratio: 0.01
  weight_quant_granularity: per_tensor
  weight_group_size: null

  linear:
    enabled: true
    include:
      - "vlm.*"
      - "expert.*"
    exclude: []
```

特别注意：

**不要写 `include: ["*"]`。**

因为当前 wrapper 还有 misc/action-head Linear 遍历；`*` 会扩大研究范围。

---

# 9. G6-A：All Linear FP8 control

文件：

```text
configs/g6a_all_fp8_control.yaml
```

不设置 override：

```yaml
linear:
  enabled: true
  include:
    - "vlm.*"
    - "expert.*"
  exclude: []
  overrides: []
```

所有 7 类 operator：

```yaml
a_bit: e4m3
w_bit: e4m3
o_bit: e4m3
```

结果应与历史 G1-A（Goal 88%）同量级。

但必须重新跑，因为：

1. core routing code 已修改；
2. 这是 G6 自己的 exact control；
3. 后续 B/C/D 都要相对 G6-A 算 Δ。

---

# 10. G6-B：VLM Attention W4，Expert FP8

只增加：

```yaml
linear:
  enabled: true
  include:
    - "vlm.*"
    - "expert.*"
  exclude: []

  overrides:
    - name: vlm_attention_w4
      target:
        module_id: "vlm.layers.*.self_attn.*_proj"
      config:
        w_bit: 4
        method: pot_ao_outlier
```

所以最终：

```text
VLM q/k/v/o:
    A = E4M3
    W = INT4
    O = E4M3
    A/O scale = PoT
    W scale = continuous
    outlier = 1%

VLM gate/up/down:
    A/W/O = E4M3
    pot_fp8_outlier

Expert q/k/v/o/gate/up/down:
    A/W/O = E4M3
    pot_fp8_outlier
```

---

# 11. G6-C：VLM MLP W4，Expert FP8

Override：

```yaml
linear:
  enabled: true
  include:
    - "vlm.*"
    - "expert.*"
  exclude: []

  overrides:
    - name: vlm_mlp_w4
      target:
        module_id: "vlm.layers.*.mlp.*_proj"
      config:
        w_bit: 4
        method: pot_ao_outlier
```

最终：

```text
VLM q/k/v/o     = FP8
VLM gate/up/down= W4
Expert all      = FP8
```

---

# 12. G6-D：VLM all W4，Expert FP8

Override：

```yaml
linear:
  enabled: true
  include:
    - "vlm.*"
    - "expert.*"
  exclude: []

  overrides:
    - name: vlm_all_w4
      target:
        module_id: "vlm.layers.*.*.*_proj"
      config:
        w_bit: 4
        method: pot_ao_outlier
```

最终：

```text
VLM all Linear    = A8W4O8
Expert all Linear = A8W8O8
```

这是以前没有直接跑过的组合。

---

# 13. Calibration 固定设置

四组完全一致：

```yaml
calibration:
  dataset_repo_id: HuggingFaceVLA/libero
  dataset_revision: v3.0
  episodes: 8
  batch_size: 8
  frame_stride: 4
  seed: 42
```

并固定：

```yaml
calibration_policy:
  default: auto
  layer_policy:
    q_proj: recalibrate
    k_proj: recalibrate
    v_proj: recalibrate
    o_proj: recalibrate
    gate_proj: recalibrate
    up_proj: recalibrate
    down_proj: recalibrate
    qk_matmul: recalibrate
    pv_matmul: recalibrate
  per_layer_policy: {}
```

每个 config 独立 scale 目录，例如：

```text
scales/2026-09-10_phaseG_w4-root-cause/
└── vlm-selective-expert-fp8/
    ├── g6a_all_fp8_control/
    ├── g6b_vlm_attn_w4/
    ├── g6c_vlm_mlp_w4/
    └── g6d_vlm_all_w4/
```

禁止跨组 scale reuse。

---

# 14. Evaluation 固定设置

保持 G5 完全一致：

```yaml
model:
  overrides:
    n_action_steps: 10
    num_steps: 10

evaluation:
  env:
    type: libero
    task: libero_goal
    max_parallel_tasks: 1

  n_episodes: 10
  batch_size: 1
  use_async_envs: false
  seed: 1000
  max_episodes_rendered: 0

  rename_map:
    observation.images.image: observation.images.camera1
    observation.images.image2: observation.images.camera2
```

即：

```text
10 tasks × 10 episodes = 100 episodes / config
```

四组共：

```text
400 episodes
```

---

# 15. 正式开跑顺序

## Gate 0：代码静态检查

```bash
python -m py_compile src/vla_tcs2/model_wrapper.py
```

如果项目已有 lint/test 命令，追加执行。

---

## Gate 1：Resolver unit tests

至少覆盖：

### Test 1：没有 overrides

```text
VLM q_proj → base q_proj FP8
Expert q_proj → base q_proj FP8
```

### Test 2：VLM Attention override

```text
vlm.layers.3.self_attn.q_proj → W4
expert.layers.3.self_attn.q_proj → FP8
vlm.layers.3.mlp.gate_proj → FP8
```

### Test 3：VLM MLP override

```text
vlm.layers.7.mlp.down_proj → W4
vlm.layers.7.self_attn.o_proj → FP8
expert.layers.7.mlp.down_proj → FP8
```

### Test 4：unknown field

例如：

```yaml
config:
  wieght_bit: 4
```

必须 fail fast，而不是静默继续。

### Test 5：non-per-site

如果：

```yaml
linear_scale_granularity: global
linear:
  overrides: [...]
```

必须直接报错。

---

## Gate 2：Routing manifest

逐个 config：

```bash
python experiments/.../scripts/audit_routing.py \
  --config experiments/.../configs/g6a_all_fp8_control.yaml
```

必须严格得到：

```text
G6-A: FP8=224, W4=0
G6-B: FP8=160, W4=64
G6-C: FP8=176, W4=48
G6-D: FP8=112, W4=112
MatMul: 64 for every config
```

任何一个 count 不对都 STOP。

---

## Gate 3：Raw equivalence

四个 config 至少各做一次 module raw equivalence。

必须 PASS 后才 calibration。

---

## Gate 4：Calibration-only smoke

执行：

```bash
python main.py \
  --config <G6_CONFIG> \
  --skip-evaluation
```

检查：

```text
Calibration action summary
scale_dir
scale files
无 missing scale
无 method mismatch
无 NaN/Inf
```

期望全新目录下：

```text
Linear 224 个 physical sites
MatMul 64 个 physical sites
```

全部得到自己的 per-site scales。

---

## Gate 5：LIBERO task0 × 1 smoke

不要直接跑 100ep。

临时 smoke config：

```text
libero_goal task0
1 episode
```

四组都要求：

- model load PASS；
- calibration PASS；
- quant_forward PASS；
- rollout 无 crash；
- 输出非 NaN；
- routing manifest 与正式 config 相同。

---

## Gate 6：Goal ×100

顺序建议：

```text
G6-A
→ G6-B
→ G6-C
→ G6-D
```

不要四个并发占同 GPU，避免 timing / resource 干扰。

---

# 16. 运行脚本建议

`run_goal.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/VLA_tcs2"
TASK="$ROOT/experiments/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8"

cd "$ROOT"
conda activate smolvla_eval

configs=(
  g6a_all_fp8_control
  g6b_vlm_attn_w4_expert_fp8
  g6c_vlm_mlp_w4_expert_fp8
  g6d_vlm_all_w4_expert_fp8
)

for name in "${configs[@]}"; do
  cfg="$TASK/configs/${name}.yaml"
  out="$ROOT/outputs/2026-09-10_phaseG_w4-root-cause/tasks/vlm-selective-expert-fp8/${name}"

  if [[ -f "$out/result.json" ]]; then
    echo "[SKIP] $name already complete"
    continue
  fi

  echo "[RUN] $name"
  python main.py --config "$cfg"
done
```

正式脚本可再检查 `eval_info.json/result.json` 完整性，不要仅以目录存在判 done。

---

# 17. G6 结果应该如何分析

当前 G5：

```text
Expert Linear = raw FP
```

结果：

| 组 | SR |
|---|---:|
| G5-A VLM all FP8 | 90% |
| G5-B Attn W4 | 72% |
| G5-C MLP W4 | 39% |
| G5-D VLM all W4 | 21% |

G6 改成：

```text
Expert Linear = FP8
```

定义：

```text
SR_A = G6-A
SR_B = G6-B
SR_C = G6-C
SR_D = G6-D
```

### VLM Attention W4 loss

\[
L_{attn}=SR_A-SR_B
\]

### VLM MLP W4 loss

\[
L_{mlp}=SR_A-SR_C
\]

### VLM all W4 loss

\[
L_{all}=SR_A-SR_D
\]

---

## 17.1 判断 G5 是否在 Expert-FP8 背景下稳定

原 G5：

```text
L_attn = 18pp
L_mlp  = 51pp
L_all  = 69pp
```

若 G6：

```text
|L_attn - 18| <= 6pp
|L_mlp  - 51| <= 6pp
```

且仍有：

```text
L_mlp >> L_attn
```

则可正式写成：

> **VLM MLP > Attention 的 W4 sensitivity 对 Expert precision background 鲁棒。**

---

## 17.2 直接计算 Expert-FP8 interaction

例如：

\[
I_A = SR(G6A)-SR(G5A)
\]

\[
I_B = SR(G6B)-SR(G5B)
\]

\[
I_C = SR(G6C)-SR(G5C)
\]

\[
I_D = SR(G6D)-SR(G5D)
\]

若：

```text
I_A/I_B/I_C/I_D 均在 ±6pp 左右
```

则说明：

> Expert 从 raw FP → FP8 不会明显改变 VLM selective W4 的敏感性排序。

如果某一组出现：

```text
|I_X| >= 15pp
```

则说明 VLM/Expert quantization 存在明显 interaction，不能再把两部分独立解释。

---

# 18. G6 结束后的真正 deployment candidate

一旦 component-aware override 可用，最重要的下一配置是：

```text
VLM Linear    = FP8
Expert Linear = W4
QK/PV         = FP8
```

这正是之前框架难以自然表达、但硬件上最有希望的 mixed precision。

配置只需：

```yaml
linear:
  enabled: true
  include:
    - "vlm.*"
    - "expert.*"
  exclude: []

  overrides:
    - name: expert_all_w4
      target:
        module_id: "expert.layers.*.*.*_proj"
      config:
        w_bit: 4
        method: pot_ao_outlier
```

base operator config 仍全部 FP8。

这样得到：

```text
VLM 112 Linear    → FP8
Expert 112 Linear → W4
MatMul 64         → FP8
```

这应作为 Phase H 的新 **S2**，而不是用当前：

```text
S1 = VLM raw FP + Expert W4
```

代替。

---

# 19. 对 Phase H 的意义

当前 H0：

```text
S0 wrapped Linear = 224
S1 wrapped Linear = 112
```

原因：

```text
S0 = VLM FP8 + Expert FP8
S1 = VLM raw FP + Expert W4
```

两组 static quantized-weight coverage 不一样。

升级后可以建立：

```text
S2 = VLM FP8 + Expert W4
```

则：

```text
wrapped Linear = 224
```

而内部 precision：

```text
112 VLM FP8
112 Expert W4
```

这样：

- coverage 与 S0 一致；
- 可以做真正 whole-quantized-model 的 sparsity comparison；
- 也更接近实际 mixed-precision accelerator mapping。

所以 component-aware precision routing 不只是为了重做 G5，也是 Phase H 后续比较成立的基础设施。

---

# 20. 推荐 commit 划分

不要把框架升级和实验配置塞一个 commit。

## Commit A：Core routing

```text
quant: add module-aware Linear precision overrides
```

只包括：

```text
src/vla_tcs2/model_wrapper.py
tests / audit helper（如有）
```

要求：

- old config behavior unchanged；
- no override → old routing semantics；
- override resolution unit tests PASS；
- per-site guard PASS。

---

## Commit B：G6 experiment

```text
experiment: add Expert-FP8 VLM selective-precision validation
```

包括：

```text
experiments/.../tasks/vlm-selective-expert-fp8/
```

以及：

- configs；
- audit_routing.py；
- run_smoke.sh；
- run_goal.sh；
- experiment_setup.md；
- results.md。

---

## Commit C：Phase H S2（G6 后）

```text
experiment: add VLM-FP8 + Expert-W4 deployment candidate
```

不要在 G6 结果出来前提前把它声明为 accuracy-preserving。

---

# 21. 最终 Gate Checklist

## Framework Gate

- [ ] `linear.overrides` YAML 可解析
- [ ] base operator config 保持兼容
- [ ] `module_id` glob override 正确
- [ ] unknown override key fail-fast
- [ ] overrides + non-per-site fail-fast
- [ ] no-override old config routing 不变
- [ ] raw Linear equivalence PASS
- [ ] G6-A 224 FP8 Linear
- [ ] G6-B 160 FP8 + 64 W4
- [ ] G6-C 176 FP8 + 48 W4
- [ ] G6-D 112 FP8 + 112 W4
- [ ] 每组 MatMul = 64 FP8

## Calibration Gate

- [ ] 每组独立 scale_dir
- [ ] 全部 Linear/MatMul recalibrate
- [ ] VLM/Expert scale filenames 无碰撞
- [ ] 无 missing scales
- [ ] 无 NaN/Inf
- [ ] `--skip-evaluation` calibration-only PASS

## Rollout Gate

- [ ] task0 × 1 四组 PASS
- [ ] Goal ×100 G6-A 完成
- [ ] Goal ×100 G6-B 完成
- [ ] Goal ×100 G6-C 完成
- [ ] Goal ×100 G6-D 完成
- [ ] task-wise SR 保存
- [ ] 与 G5 90/72/39/21 并排分析

---

# 22. 最终研究问题

这次升级后，G6 回答的不是：

> “Expert FP8 能不能跑？”

而是更严格的问题：

\[
\boxed{
\text{当 VLM 与 Expert 都进入量化部署域后，
VLM 内部 MLP > Attention 的 W4 敏感性结论是否仍成立？}
}
\]

如果成立，则 Phase G 可以把结论从：

```text
在 Expert raw FP 背景下，VLM MLP 最敏感
```

升级为：

```text
在 deployment-like Expert FP8 背景下，
VLM MLP 仍是 W4 的主要 accuracy bottleneck
```

然后 Phase H 使用：

```text
VLM FP8 + Expert W4 + MatMul FP8
```

作为真正的 accuracy/sparsity/hardware mixed-precision candidate。
