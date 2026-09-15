# SmolVLA Vision Quantization & Workload Experiment Manual

> 日期：2026-09-15  
> 建议实验名：`2026-09-15_phaseI_vision-quantization`  
> 目标：在**不破坏现有 VLM / Action Expert 量化主线**的前提下，将 SmolVLA 的 Vision Encoder 与 Connector 按阶段纳入 workload 统计、PTQ、稀疏统计和硬件分析。  
> 原则：**一次只改变一个变量；每一阶段必须通过 raw-equivalence / routing / calibration / smoke / closed-loop gate 后才能进入下一阶段。**

---

# 0. 为什么现在必须补 Vision Quantization

当前项目的量化覆盖范围主要是：

- VLM text transformer：Q/K/V/O projection + MLP；
- Action Expert：attention projection + MLP；
- 可选的 VLM / Expert `QK^T` 与 `P×V` MatMul；
- Vision Encoder：当前未纳入；
- Connector：当前未纳入；
- Patch embedding Conv2d：当前未纳入。

而对当前 `lerobot/smolvla_libero` + LIBERO 双相机、512×512 输入，理论 dense FLOPs 粗估为：

| 组件 | FLOPs / `sample_actions()` | 约占总 inference |
|---|---:|---:|
| Vision Encoder + Connector | ~430.6 GFLOPs | ~72% |
| VLM 16-layer prefix | ~57.6 GFLOPs | ~10% |
| Expert 10-step denoise | ~107.5 GFLOPs | ~18% |
| 总计 | ~595.7 GFLOPs | 100% |

Vision 内部继续拆分：

| Vision 子模块 | 双相机 FLOPs | Vision 内占比 |
|---|---:|---:|
| MLP `fc1/fc2` | ~231.9 G | ~53.9% |
| Attention Q/K/V/out projection | ~116.0 G | ~26.9% |
| Attention `QK^T + P×V` | ~77.3 G | ~18.0% |
| Patch embedding Conv2d | ~2.42 G | ~0.56% |
| Connector projection | ~3.02 G | ~0.70% |
| Pixel shuffle | ~0 arithmetic FLOPs | ~0% |

因此：

> **只补 Vision MLP + Vision attention projection，就能覆盖约 80.8% 的 Vision compute；再加入 Vision QK/PV 后，V1–V4 基本覆盖 Vision+Connector 99% 以上的主要 arithmetic FLOPs。**

Patch Conv2d 的算力占比很小，因此最后再处理。

---

# 1. 官方代码中的 Vision 数据流

当前 SmolVLA 推理的数据流应理解为：

```text
LIBERO RGB image(s)
        │
        ▼
SmolVLM Vision Encoder
  patch embedding Conv2d
        │
        ▼
12 × Vision Transformer
  ├─ LayerNorm
  ├─ q_proj / k_proj / v_proj
  ├─ QK^T
  ├─ Softmax
  ├─ P×V
  ├─ out_proj
  ├─ residual
  ├─ LayerNorm
  ├─ fc1
  ├─ GELU
  └─ fc2
        │
        ▼
Pixel Shuffle
1024 vision tokens → 64 visual tokens / camera
        │
        ▼
Connector / modality projection
        │
        ▼
64 × VLM hidden visual tokens / camera
        │
        ▼
VLM prefix transformer × 16
        │
        ▼
VLM KV cache
        │
        ▼
Action Expert × 10 flow steps
```

对于当前 500M Video backbone，实验前应由脚本再次确认实际 runtime config，而不是把以下数值写死进 profiler：

```text
Vision hidden ≈ 768
Vision FFN ≈ 3072
Vision layers ≈ 12
Vision heads ≈ 12
patch_size ≈ 16
image_size = runtime 实际 resize，当前 LIBERO 路线通常 512×512
pixel_shuffle factor ≈ 4
VLM hidden ≈ 960
```

**重要：当前 checkpoint 配置里可能声明 3 个 camera feature，但当前 LIBERO runtime 实际只提供两张图。FLOPs 必须按实际 `vision_model.forward()` 调用次数统计，不能按 config 里的 camera key 数量猜。**

---

# 2. 实验总体顺序

整个 Vision 扩展按下面顺序执行：

```text
V0  Vision workload / shape / FLOPs audit
 │   不量化，只建立可信 workload 基线
 ▼
V1  Connector Linear quantization
 │   最小改动，验证 Vision 侧 QuantizedLinear plumbing
 ▼
V2  Vision MLP fc1/fc2 quantization
 │   最大单一 FLOPs 块，优先级最高
 ▼
V3  Vision attention projection quantization
 │   q/k/v/out_proj
 ▼
V4  Vision attention MatMul quantization
 │   QK^T + P×V
 ▼
V5  Patch embedding Conv2d quantization（可选）
 │   FLOPs 很小，仅在“完整 Vision quantization”需要时做
 ▼
V6  Integrated candidate + full benchmark
```

每个阶段都必须保留一个 **raw-wrapped equivalence** 配置，证明“加入 wrapper 本身没有改变模型行为”。

---

# 3. 这次代码修改的核心设计原则

## 3.1 不改变现有 VLM / Expert 行为

Vision 扩展必须默认关闭：

```yaml
quantization:
  vision:
    enabled: false
    quantize_matmul: false
    quantize_patch_embed: false
  connector:
    enabled: false
```

这样旧的 G5/G6 配置在不增加任何字段时，行为保持原样。

**这是本次改动最重要的 backward-compatibility gate。**

预期旧配置仍然得到：

```text
QuantizedLinear   = 224
QuantizedMatMul   = 64    # 若旧配置原本开启 VLM/Expert matmul
Vision wrappers   = 0
Connector wrappers= 0
```

任何偏差都应 STOP，不进入 Vision 实验。

---

## 3.2 不新建 VisionQuantizedLinear

当前 `QuantizedLinear` 最终使用 `F.linear(x, weight, bias)`，可接受：

```text
[B, T_text, D]
[B, T_action, D]
[B, 1024, 768]
```

因此 Vision `q/k/v/out_proj` 与 `fc1/fc2` 不需要新算子类。

本轮只扩展：

- module discovery；
- canonical module_id；
- component routing；
- YAML target；
- manifest / counts；
- Vision attention MatMul hook。

---

## 3.3 phase 不拆，component 拆

当前 runtime phase 已经有：

```text
prefill
 denoise
```

Vision 与 Connector 都发生在 `sample_actions()` 的 prefix 路径，因此仍然属于：

```text
phase = prefill
```

不要为了 Vision 重写当前 `CURRENT_PHASE` 逻辑。

通过 module id / StatManager component 区分：

```text
phase=prefill, component=vision
phase=prefill, component=connector
phase=prefill, component=vlm
phase=denoise, component=expert
```

当前 StatManager 已经会从 module id 第一段解析 component，因此第一阶段**不需要增加 `CURRENT_COMPONENT` ContextVar**。

---

# 4. 目录结构

按照当前项目实验管理规范建立：

```bash
cd ~/SmolVLA_qtrsc   # 以实际远端仓库路径为准
bash experiments/new_experiment.sh 2026-09-15 phaseI vision-quantization
```

推荐目录：

```text
experiments/
└── 2026-09-15_phaseI_vision-quantization/
    ├── README.md
    ├── configs/
    │   ├── v0_workload_audit.yaml
    │   ├── v1_connector_raw.yaml
    │   ├── v1_connector_fp8.yaml
    │   ├── v2_vision_mlp_raw.yaml
    │   ├── v2_vision_mlp_fp8.yaml
    │   ├── v2_vision_mlp_w4.yaml
    │   ├── v3_vision_attnproj_raw.yaml
    │   ├── v3_vision_attnproj_fp8.yaml
    │   ├── v3_vision_attnproj_w4.yaml
    │   ├── v4_vision_attnmatmul_raw.yaml
    │   ├── v4_vision_attnmatmul_fp8.yaml
    │   └── v6_integrated_candidate.yaml
    ├── scripts/
    │   ├── audit_vision_structure.py
    │   ├── audit_vision_routing.py
    │   ├── compare_raw_equivalence.py
    │   ├── run_v0.sh
    │   ├── run_v1.sh
    │   ├── run_v2.sh
    │   ├── run_v3.sh
    │   ├── run_v4.sh
    │   └── summarize_vision_quant.py
    ├── docs/
    │   ├── CODE_CHANGE_PLAN.md
    │   ├── GATES.md
    │   └── RESULTS.md
    └── tasks/
        ├── v0-workload-audit/
        ├── v1-connector/
        ├── v2-vision-mlp/
        ├── v3-vision-attnproj/
        ├── v4-vision-attnmatmul/
        └── v6-integrated/
```

输出镜像：

```text
outputs/2026-09-15_phaseI_vision-quantization/
```

---

# 5. Phase V0 — Vision workload audit

## 5.1 目的

V0 不改模型计算，只回答：

1. 当前 checkpoint 实际 Vision Encoder 是多少层？
2. 实际 `q/k/v/out_proj`、`fc1/fc2` 路径是什么？
3. Connector projection 路径是什么？
4. 当前 LIBERO 一个 `sample_actions()` 实际处理几张图？
5. Vision 输入 token 是多少？pixel shuffle 后是多少？
6. 每一类算子的理论 MAC/FLOPs 是多少？
7. 当前 Vision 占整次 inference 的 FLOPs 比例是多少？

## 5.2 新增脚本

新增：

```text
experiments/.../scripts/audit_vision_structure.py
```

脚本必须做四类审计。

### A. 静态结构审计

伪代码：

```python
policy = ... load exact experiment checkpoint ...
vlm_expert = locate_smolvlm_with_expert(policy)
vlm_model = vlm_expert.get_vlm_model()
vision_model = vlm_model.vision_model
connector = vlm_model.connector

print(type(vision_model))
print(vision_model.config)
print(len(vision_model.encoder.layers))

for i, layer in enumerate(vision_model.encoder.layers):
    assert isinstance(layer.self_attn.q_proj, torch.nn.Linear)
    assert isinstance(layer.self_attn.k_proj, torch.nn.Linear)
    assert isinstance(layer.self_attn.v_proj, torch.nn.Linear)
    assert isinstance(layer.self_attn.out_proj, torch.nn.Linear)
    assert isinstance(layer.mlp.fc1, torch.nn.Linear)
    assert isinstance(layer.mlp.fc2, torch.nn.Linear)
```

不要只打印，要把审计结果写成 JSON。

### B. 实际 camera call 计数

在 `vision_model.forward` 上挂临时 hook：

```python
vision_forward_calls += 1
record pixel_values.shape
```

运行**一个真正的 LIBERO policy inference**。

当前预期：

```text
vision_model.forward calls / sample_actions ≈ 2
```

但脚本不允许把 2 写死。

### C. token shape 审计

记录：

```text
Vision input image shape
Patch embedding output shape
Vision encoder output shape
Pixel shuffle output shape
Connector output shape
```

当前典型预期：

```text
[B,3,512,512]
→ 1024 patch tokens
→ 1024×768
→ 64×12288
→ 64×960
```

### D. FLOPs 计算

统一采用：

```text
1 MAC = 2 FLOPs
```

Linear：

```text
FLOPs = 2 × M × K × N
```

Vision attention QK：

```text
FLOPs_QK = 2 × B × H × Tq × Tk × Dh
```

PV：

```text
FLOPs_PV = 2 × B × H × Tq × Tk × Dh
```

Self-attention 二者合计：

```text
4 × B × H × T² × Dh
= 4 × B × T² × d
```

Conv2d：

```text
2 × B × Hout × Wout × Cout × (Cin/groups) × Kh × Kw
```

Pixel shuffle：

```text
arithmetic FLOPs ≈ 0
```

但建议额外报告：

```text
read bytes
write bytes
```

因为它是数据搬移，不代表完全免费。

## 5.3 V0 输出

必须产出：

```text
vision_structure.json
vision_runtime_shapes.json
vision_flops.csv
vision_flops_summary.md
```

`vision_flops.csv` 推荐列：

```text
component
layer_id
op_type
operator
input_shape
weight_shape
output_shape
macs
flops
percent_of_vision
percent_of_full_inference
```

## 5.4 V0 PASS 条件

必须全部满足：

- [ ] Vision 层数与 checkpoint config 一致；
- [ ] 每层确实存在 q/k/v/out + fc1/fc2；
- [ ] Connector projection 路径确认；
- [ ] runtime camera call 数通过真实 inference 得到；
- [ ] token 数通过 runtime 得到；
- [ ] FLOPs 脚本输出无 hard-coded camera 数；
- [ ] 理论 FLOPs 与手算数量级一致；
- [ ] 没有修改模型权重或 forward 行为。

V0 PASS 后才修改主量化框架。

---

# 6. 主框架修改 A — Vision/Connector 显式 opt-in

目标文件：

```text
src/vla_tcs2/model_wrapper.py
```

## 6.1 增加配置读取

建议：

```python
vision_cfg = quant_config.get("vision", {})
vision_enabled = bool(vision_cfg.get("enabled", False))
vision_quantize_matmul = bool(vision_cfg.get("quantize_matmul", False))
vision_quantize_patch_embed = bool(vision_cfg.get("quantize_patch_embed", False))

connector_cfg = quant_config.get("connector", {})
connector_enabled = bool(connector_cfg.get("enabled", False))
```

默认 false。

---

## 6.2 Vision operator name

新增：

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

注意：

```text
VLM text:  o_proj
Vision:    out_proj
```

不要把二者混淆。

---

## 6.3 新函数 `_wrap_smolvlm_vision_linear_layers`

建议独立函数，不要把 Vision 逻辑塞进 text/expert loop。

伪代码：

```python
def _wrap_smolvlm_vision_linear_layers(...):
    vlm_model = vlm_expert.get_vlm_model()
    vision_model = vlm_model.vision_model

    for layer_idx, layer in enumerate(vision_model.encoder.layers):
        # Attention projections
        for name in VISION_ATTN_LINEAR_NAMES:
            old = getattr(layer.self_attn, name)
            module_id = f"vision.layers.{layer_idx}.self_attn.{name}"

            new = create_quantized_linear(
                old,
                module_id=module_id,
                component="vision",
                layer_idx=layer_idx,
                layer_type=name,
                ...,
            )
            new.set_stat_manager(stat_manager)
            new.set_layer_info(...)
            setattr(layer.self_attn, name, new)

        # MLP
        for name in VISION_MLP_LINEAR_NAMES:
            old = getattr(layer.mlp, name)
            module_id = f"vision.layers.{layer_idx}.mlp.{name}"
            ...
```

`resolve_linear_scale_group()` 可以继续复用：

```python
resolve_linear_scale_group(
    "vision",
    layer_idx,
    name,
    granularity,
)
```

推荐本 phase 强制：

```yaml
linear_scale_granularity: per_site
```

避免新的 Vision 模块与 VLM/Expert 意外共享 scale。

---

## 6.4 Connector wrapping

官方 connector 的核心 projection 是一个普通 `nn.Linear`。

建议 canonical ID 不直接照抄 Python object path，而保持当前 StatManager 易解析的格式：

```text
connector.layer.0.connector_proj
```

伪代码：

```python
connector_proj = vlm_model.connector.modality_projection.proj

qproj = create_quantized_linear(
    connector_proj,
    module_id="connector.layer.0.connector_proj",
    component="connector",
    layer_idx=0,
    layer_type="connector_proj",
    ...,
)

vlm_model.connector.modality_projection.proj = qproj
```

这样已有 parser 可得到：

```text
component = connector
layer_idx = 0
operator = connector_proj
```

---

# 7. 修改后的 wrapper count 预期

假设当前背景仍然是现有 16-layer VLM + 16-layer Expert：

```text
现有 VLM/Expert Linear:
16 × 7 + 16 × 7
= 224
```

Vision：

```text
12 layers × (4 attention projection + 2 MLP)
= 72 Linear
```

Connector：

```text
1 Linear
```

所以：

| 阶段 | 新增 Linear | 累计 Linear |
|---|---:|---:|
| Legacy | 0 | 224 |
| V1 Connector | +1 | 225 |
| V2 Vision MLP | +24 | 249 |
| V3 Vision attention proj | +48 | 297 |

若现有 VLM/Expert QK/PV 已开启：

```text
16×2 + 16×2 = 64 QuantizedMatMul
```

Vision QK/PV：

```text
12×2 = 24
```

V4 后：

```text
QuantizedMatMul = 88
```

这些 count 必须进入 routing gate。

---

# 8. Gate L0 — Legacy regression，必须最先跑

代码改完但 Vision 仍默认 disabled 后：

```bash
python -m py_compile \
  src/vla_tcs2/model_wrapper.py \
  src/vla_tcs2/quant_linear.py \
  src/vla_tcs2/quant_matmul.py
```

然后用当前**原封不动的 canonical G6 配置** build 一次模型。

必须确认：

```text
QuantizedLinear == 224
QuantizedMatMul == 原来数量
no module_id startswith "vision."
no module_id startswith "connector."
```

如果旧实验 routing 发生任何变化：

```text
STOP
```

不要继续 V1。

---

# 9. Phase V1 — Connector quantization

## 9.1 为什么先做 Connector

Connector 只有一个 Linear：

```text
pixel shuffle output
→ Linear(大输入通道 → VLM hidden)
```

FLOPs 很少，但最适合检查：

- 新 component 是否能被 resolver 识别；
- scale group 是否独立；
- calibration 是否能正常保存；
- StatManager 是否能导出 connector；
- raw-wrapped 是否数值完全等价。

---

## 9.2 V1-R：Connector raw wrapper

配置：

```yaml
quantization:
  enabled: true

  vision:
    enabled: false

  connector:
    enabled: true

  linear:
    enabled: true
```

把 connector wrapper 切到 raw/test mode，不做 fake quant。

### V1-R PASS

比较未修改模型与 wrapped-raw 模型：

```text
connector output
prefix embedding
final action chunk
```

至少报告：

```text
max_abs_error
mean_abs_error
max_rel_error
```

理想情况：

```text
exact equal 或 dtype rounding 级误差
```

---

## 9.3 V1-FP8

建议先：

```text
A = FP8 E4M3
W = FP8 E4M3
O = FP8 E4M3
```

沿用当前项目已经验证过的 PoT / outlier 方法，不在 Vision phase 同时发明新量化算法。

示意：

```yaml
quantization:
  linear_scale_granularity: per_site

  connector:
    enabled: true

  linear:
    overrides:
      - name: connector_fp8
        target:
          component: connector
        config:
          a_bit: e4m3
          w_bit: e4m3
          o_bit: e4m3
          method: pot_fp8_outlier
```

`method` 字段名称必须以当前代码实际支持值为准，上面只是模板。

---

## 9.4 V1 实验 gate

顺序固定：

```text
Gate 1 routing count: 225 linears
Gate 2 raw equivalence
Gate 3 calibration-only
Gate 4 task0 × 1 smoke
Gate 5 Goal × 100
```

如果 V1 Connector FP8 都导致明显 SR 崩溃，先审计 calibration / outlier / connector output，不要进入 V2。

---

# 10. Phase V2 — Vision MLP quantization

## 10.1 目标

量化 12 个 Vision Transformer block 中的：

```text
fc1
fc2
```

总计：

```text
12 × 2 = 24 Linear
```

这是 Vision FLOPs 最大的单一部分：约 54%。

---

## 10.2 V2-FP8

Primary cumulative path：

```text
V1 Connector FP8
+
Vision MLP FP8
+
现有 canonical VLM/Expert quant background
```

YAML target 示例：

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
        target:
          module_id: "vision.layers.*.mlp.*"
        config:
          a_bit: e4m3
          w_bit: e4m3
          o_bit: e4m3
          method: pot_fp8_outlier
```

如果现有 matcher 不支持 glob `module_id`，则使用当前框架已经支持的 component/operator selector，或扩展 matcher，但必须加 unit test。

---

## 10.3 V2-W4 sensitivity

FP8 稳定后，再测：

```text
A8 / W4 / O8
```

或者严格使用当前项目 G5/G6 已有的 W4 配置方式。

不要在这一阶段同时改：

- outlier ratio；
- block size；
- scale granularity；
- calibration dataset；
- eval seed。

只改变 Vision MLP precision。

---

## 10.4 V2 routing 预期

如果 cumulative：

```text
224 existing
+1 connector
+24 vision MLP
=249 QuantizedLinear
```

必须逐项确认：

```text
vision.layers.0.mlp.fc1
vision.layers.0.mlp.fc2
...
vision.layers.11.mlp.fc1
vision.layers.11.mlp.fc2
```

没有漏层、没有重复。

---

# 11. Phase V3 — Vision Attention Projection quantization

## 11.1 目标

再加入每层：

```text
q_proj
k_proj
v_proj
out_proj
```

共：

```text
12 × 4 = 48 Linear
```

这部分约占 Vision FLOPs 27%。

---

## 11.2 配置

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
        config:
          a_bit: e4m3
          w_bit: e4m3
          o_bit: e4m3
          method: pot_fp8_outlier
```

注意 `out_proj` 需要明确进入 resolver/operator list。

---

## 11.3 V3 routing 预期

Cumulative：

```text
249 + 48 = 297 QuantizedLinear
```

并验证：

```text
12 q_proj
12 k_proj
12 v_proj
12 out_proj
```

---

## 11.4 V3 sensitivity

推荐只做两组：

```text
V3-FP8
V3-W4
```

如果 V3-W4 明显比 V2-W4 更敏感，说明 Vision attention projection 比 MLP 更需要精度保护，后续 integrated candidate 可以采用：

```text
Vision MLP: W4
Vision attention projection: FP8
```

不要因为 MLP 可以 W4 就默认 attention projection 也应该 W4。

---

# 12. Phase V4 — Vision QK / PV MatMul quantization

这是本轮最容易引入实现错误的一步。

---

## 12.1 为什么不能直接复用当前 LeRobot MatMul hook

现有 VLM/Expert `QuantizedMatMul` hook 针对的是 LeRobot 自己的 attention interface。

Vision Encoder 属于 Transformers `SmolVLMVisionAttention` 路径，不经过当前 text/expert hook。

因此必须单独注入 Vision attention。

---

## 12.2 V4 开始前先审计 attention backend

输出：

```python
print(vision_model.config._attn_implementation)
```

推荐第一版只支持：

```text
eager
```

如果 runtime 是：

```text
sdpa
flash_attention_2
其他 fused backend
```

不要直接 patch。

先建立一个独立：

```text
EAGER-BACKEND EQUIVALENCE GATE
```

证明切换到 eager 不改变闭环输入输出到不可接受程度，然后再量化 QK/PV。

---

## 12.3 推荐 patch 方法

第一版不要重新注册一个全新的 Transformers AttentionInterface backend，因为 mask / backend routing 容易发生版本相关差异。

更安全的方式是：

1. 保留当前 attention backend 选择逻辑；
2. 只在确认使用 eager 时，process-local patch SmolVLM vision eager function；
3. 给每个 Vision attention module 挂两个 `QuantizedMatMul`。

每层：

```text
vision.layer.0.qk
vision.layer.0.pv
...
vision.layer.11.qk
vision.layer.11.pv
```

共 24 个。

---

## 12.4 伪代码

示意，不应直接盲拷贝到生产代码：

```python
from transformers.models.smolvlm import modeling_smolvlm as smolvlm_mod

_original_vision_eager_attention_forward = smolvlm_mod.eager_attention_forward


def quantized_vision_eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    if not hasattr(module, "_vla_qk_matmul"):
        return _original_vision_eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            **kwargs,
        )

    attn_weights = module._vla_qk_matmul(
        query,
        key.transpose(-1, -2),
    ) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_probs = torch.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)

    attn_probs = torch.nn.functional.dropout(
        attn_probs,
        p=dropout,
        training=module.training,
    )

    attn_output = module._vla_pv_matmul(attn_probs, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

**必须逐行对照当前安装的 Transformers 版本中原始 eager attention 实现。**

绝对不要根据这份文档里的伪代码替换官方实现而不做 diff。

---

## 12.5 patch 必须可恢复、幂等

实现：

```text
if already patched:
    do not patch again

save original function

restore on teardown / rebuild when possible
```

避免 notebook / repeated build 中重复包裹。

---

## 12.6 V4 raw-equivalence 是强制 gate

首先用：

```text
QuantizedMatMul = raw mode
```

检查：

```text
Vision layer output
Vision final hidden
Connector output
policy action chunk
```

至少保存：

```text
max_abs
mean_abs
max_rel
```

如果 raw patch 本身产生明显差异：

```text
STOP
```

不能用后面的 quantization 结果掩盖 patch error。

---

## 12.7 V4-FP8

先只做：

```text
QK: FP8
PV: FP8
```

不要第一组就 W4 activation MatMul。

因为 QK/PV 都是 runtime activation × activation，和 weight-only/linear 的数值性质不同。

---

## 12.8 V4 routing count

Cumulative 预期：

```text
QuantizedLinear = 297
QuantizedMatMul = 64 existing +24 vision =88
```

如果当前 background 没有开启 VLM/Expert matmul，那么 Vision 自己应精确为：

```text
24
```

所以 routing report 必须同时给：

```text
by component
by op_type
by operator
```

而不能只看总数。

---

# 13. Phase V5 — Patch embedding Conv2d（可选）

## 13.1 为什么最后做

当前双相机条件下，Patch Conv2d 只有约：

```text
2.4 GFLOPs / sample_actions
```

只占 Vision 约：

```text
0.56%
```

所以 V1–V4 已经基本覆盖主要算力。

除非：

- 论文需要声称“全 Vision quantized”；
- reviewer 明确要求 Conv；
- 目标硬件对 Conv2d 有特殊 mapping；
- 后续扩展到更大的 vision front-end；

否则 V5 可以 deferred。

---

## 13.2 若实现，不要硬套 QuantizedLinear

新增：

```text
src/vla_tcs2/quant_conv2d.py
```

至少支持：

```text
raw
scale_inspection
quant_forward
```

forward：

```python
F.conv2d(...)
```

权重 shape：

```text
[Cout, Cin/groups, Kh, Kw]
```

per-output-channel scaling 的维度定义也要重新确认，不能直接复制 Linear `[out,in]` 的 axis 假设。

---

# 14. Calibration 设置

为避免 Vision 实验与旧 scale 污染：

> **每个实验配置必须使用独立的 `scale_dir`。绝不跨 precision / component 复用 calibration scale。**

推荐 calibration dataset 延续当前主线：

```text
HuggingFaceVLA/libero
revision: 与当前主线一致
```

建议保持当前已固定的：

```text
episodes
batch_size
stride
seed
```

不在 Vision phase 修改 calibration protocol。

---

## 14.1 Vision calibration coverage audit

每次 calibration 结束后输出：

```text
expected_modules
seen_modules
missing_modules
call_count per module
```

V2：

```text
24 / 24 vision MLP sites seen
```

V3：

```text
48 / 48 vision attention projection sites seen
```

V4：

```text
24 / 24 vision matmul sites seen
```

Connector：

```text
1 / 1
```

任何 expected site call_count=0：

```text
FAIL
```

---

# 15. Closed-loop evaluation protocol

这里必须区分两条线。

## 15.1 Vision quant 主线：与当前 G5/G6 保持可比

建议继续使用当前量化研究的 canonical deployment protocol，而不是此时切换到 Table-2 strict protocol。

如果当前 G6 使用：

```text
model = lerobot/smolvla_libero
n_action_steps = 10
num_steps = 10
LIBERO Goal
10 tasks × 10 episodes = 100 episodes
seed = 1000
batch = 1
max_parallel_tasks = 1
async = false
```

Vision phase 继续保持完全一致。

理由：

> Vision precision 是本阶段唯一变量。

如果在这里把 `n_action_steps=10` 改成 1，你将无法把 SR 变化归因给 Vision quantization。

---

## 15.2 Table-2 strict track 单独保留

论文严格 simulation track：

```text
n_action_steps = 1
num_steps = 10
```

可以在 V6 final candidate 成型之后单独运行。

但结果必须明确标：

```text
Strict Table-2-style protocol
```

不要和 `n_action_steps=10` 的量化主线 SR 直接混成一个表。

---

# 16. 每个阶段统一 Gate

推荐把所有阶段都固定成以下 Gate。

## Gate 0 — Manifest

保存：

```bash
git rev-parse HEAD
git status --short
python --version
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import lerobot; print(lerobot.__version__)"
python -c "import mujoco; print(mujoco.__version__)"
```

再保存：

```text
checkpoint repo + revision
calibration dataset + revision
vision_model.config
_attn_implementation
GPU
CUDA
```

---

## Gate 1 — Static code

```bash
python -m py_compile ...
pytest <new vision routing tests>
```

---

## Gate 2 — Legacy regression

旧 config count 与行为完全不变。

---

## Gate 3 — Routing manifest

逐 component count。

---

## Gate 4 — Raw equivalence

wrapper/patch raw mode 与未修改模型一致。

---

## Gate 5 — Calibration-only

```bash
python main.py \
  --config <CONFIG> \
  --skip-evaluation
```

必须确认：

```text
all expected modules calibrated
no NaN
no Inf
scale nonzero
outlier stats sane
```

---

## Gate 6 — Task0 × 1 smoke

只检查：

```text
model load
rollout complete
no exception
no NaN action
behavior not obviously broken
```

1 episode 不是 accuracy 结论。

---

## Gate 7 — Goal × 100

正式闭环结论。

---

## Gate 8 — Workload / sparsity / latency

导出：

```text
quantization_manifest
workload.csv
module_sparsity
outlier stats
latency profile
```

---

## Gate 9 — Full four-suite

只对最终 integrated candidate 做：

```text
Spatial
Object
Goal
Long
```

不要每个 ablation 都跑 400 episodes。

---

# 17. 推荐实验矩阵

## 17.1 Primary cumulative path

| ID | Connector | Vision MLP | Vision Attn Proj | Vision QK/PV | Patch Conv | 目的 |
|---|---|---|---|---|---|---|
| V-CTRL | raw | raw | raw | raw | raw | 当前 canonical quant background |
| V1 | FP8 | raw | raw | raw | raw | plumbing |
| V2-FP8 | FP8 | FP8 | raw | raw | raw | MLP 主算力 |
| V2-W4 | FP8 | W4 | raw | raw | raw | MLP sensitivity |
| V3-FP8 | FP8 | FP8 | FP8 | raw | raw | all Vision Linear |
| V3-W4 | FP8 | W4/FP8* | W4 | raw | raw | projection sensitivity |
| V4-FP8 | FP8 | chosen | chosen | FP8 | raw | full major Vision compute |
| V6 | chosen | chosen | chosen | chosen | raw | integrated candidate |

`*` 具体使用 V2 最优结果，不要强制所有模块同 bit。

---

## 17.2 Isolation groups（只有出现异常时再跑）

如果 cumulative path 某一步突然大幅掉点，再运行：

```text
V2-ISO: 仅 Vision MLP quant，其余 Vision raw
V3-ISO: 仅 Vision attention projection quant
V4-QK-ISO: 仅 QK quant
V4-PV-ISO: 仅 PV quant
```

不要一开始就把实验矩阵爆炸成几十组。

---

# 18. Accuracy 判定

Primary metric：

```text
Success Rate
```

同时报告：

```text
ΔSR vs immediate parent
ΔSR vs V-CTRL
```

例如：

```text
V2-FP8 vs V1
V2-FP8 vs V-CTRL
```

当前项目可继续沿用之前的工程阈值作为**调试启发式**：

```text
≤ ~6 pp: relatively stable
> ~15 pp: high sensitivity / likely unacceptable
```

但文档中必须写清：

> 这是工程筛选阈值，不是统计显著性检验。

---

# 19. FLOPs、BOPs、稀疏度与 latency 必须分开

后续论文图至少同时有：

## A. Dense theoretical FLOPs

回答：

```text
结构上谁最贵？
```

## B. Precision-weighted BOP proxy

例如：

```text
MACs × activation_bits × weight_bits
```

对 activation×activation MatMul 定义独立 BOP 公式并写清。

## C. Sparsity-adjusted effective ops

使用当前 StatManager 的：

```text
zero ratio
bit sparsity
outlier side path
```

但不要把 sparsity-adjusted op 数冒充真实硬件 latency。

## D. Measured CUDA latency

回答：

```text
实际上谁耗时？
```

---

# 20. Latency profiling 设置

建议分别测：

```text
full predict_action_chunk / sample_actions
vision-only
VLM prefix
10-step expert
```

CUDA timing：

```python
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

# warmup >= 20
# measured >= 100

torch.cuda.synchronize()
starter.record()
...
ender.record()
torch.cuda.synchronize()
ms = starter.elapsed_time(ender)
```

固定：

```text
batch=1
same image count
same image resolution
same tokenizer length policy
same flow steps=10
```

不要一边 profile 一边跑其他 GPU workload。

报告：

```text
mean
median
p50
p90
p99
std
```

并计算：

```text
Effective TFLOP/s = theoretical FLOPs / measured latency
```

但明确：

> theoretical FLOPs 和 hardware executed FLOPs 不是同一个概念。

---

# 21. StatManager 需要确认/补充的内容

现有 StatManager 已能从 module id 解析 component，因此 Vision wrapper 后应天然得到：

```text
vision
connector
vlm
expert
```

需要新增 unit test：

```python
_parse_module_id("vision.layers.3.mlp.fc1")
# expected component=vision, layer=3, operator=fc1

_parse_module_id("vision.layers.7.self_attn.out_proj")
# component=vision, layer=7, operator=out_proj

_parse_module_id("connector.layer.0.connector_proj")
# component=connector, layer=0, operator=connector_proj
```

`workload.csv` 最终应该能够按 component 聚合：

```text
vision
connector
vlm
expert
```

---

# 22. 新增测试建议

新增：

```text
tests/test_vision_quant_routing.py
```

至少包含：

### Test 1 — default off

```text
vision.enabled=false
connector.enabled=false
→ legacy counts unchanged
```

### Test 2 — connector only

```text
+1 QuantizedLinear
```

### Test 3 — vision MLP only

```text
12×2=24
```

### Test 4 — vision attn projection only

```text
12×4=48
```

### Test 5 — full vision linear

```text
72
```

### Test 6 — module IDs unique

```text
len(ids) == len(set(ids))
```

### Test 7 — raw forward equivalence

固定随机 tensor，原始 Linear 与 wrapped raw Linear exact/close。

### Test 8 — connector shape

确保 wrapper 不改变：

```text
[..., connector_in] → [..., VLM hidden]
```

### Test 9 — vision matmul routing

V4 开启后：

```text
12 qk
12 pv
```

---

# 23. 推荐 YAML 设计

下面不是完整 config，而是新字段模板。

```yaml
quantization:
  enabled: true

  linear_scale_granularity: per_site

  # Existing VLM / Expert behavior remains controlled by current config.
  quantize_matmul: true

  vision:
    enabled: true
    quantize_matmul: false
    quantize_patch_embed: false

  connector:
    enabled: true

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
          method: <USE_CURRENT_VALID_METHOD>

      - name: vision_mlp_fp8
        target:
          module_id: "vision.layers.*.mlp.*"
        config:
          a_bit: e4m3
          w_bit: e4m3
          o_bit: e4m3
          method: <USE_CURRENT_VALID_METHOD>

      - name: vision_attn_proj_fp8
        target:
          module_id: "vision.layers.*.self_attn.*"
        config:
          a_bit: e4m3
          w_bit: e4m3
          o_bit: e4m3
          method: <USE_CURRENT_VALID_METHOD>
```

如果 matcher 当前不支持 `module_id` glob：

> 不要为了文档形式强行改配置 parser；优先使用现有 selector 能表达的 component/operator 组合。只有确实表达不了时再扩展 matcher，并添加测试。

---

# 24. Vision MatMul YAML

建议把 Vision MatMul 与现有 text/expert MatMul 开关解耦：

```yaml
quantization:
  quantize_matmul: true        # existing vlm/expert semantics

  vision:
    enabled: true
    quantize_matmul: true      # only vision qk/pv
```

这样：

```text
vision.quantize_matmul=false
```

不会意外关闭现有 VLM/Expert MatMul，也不会因为顶层 `quantize_matmul=true` 就自动量化 Vision。

---

# 25. Calibration scale policy

新增 Vision 后必须检查 `layer_policy` / scale mapping 是否显式包含：

```text
q_proj
k_proj
v_proj
out_proj
fc1
fc2
connector_proj
qk
pv
```

如果当前 resolver 对未知 operator 会 fallback 到全局默认，这在 smoke 阶段可能不报错，但会造成不可控配置。

因此 routing manifest 必须打印每个 module 最终解析到的：

```text
a_bit
w_bit
o_bit
method
scale_group
outlier config
```

---

# 26. Quantization manifest 的新预期

最终 manifest 示例：

```text
vision.layers.0.self_attn.q_proj
  component=vision
  operator=q_proj
  precision=FP8
  scale_group=vision_q_proj_layer0

vision.layers.0.mlp.fc1
  component=vision
  operator=fc1
  precision=W4A8
  scale_group=vision_fc1_layer0

connector.layer.0.connector_proj
  component=connector
  operator=connector_proj
  precision=FP8
```

论文/汇报时就能直接聚合：

```text
component → operator family → layer → precision
```

---

# 27. Sparsity 统计

Vision 加入后建议沿用当前 sparsity framework，但额外单独聚合：

```text
Vision MLP activation sparsity
Vision attention projection activation sparsity
Vision QK input sparsity
Vision attention probability sparsity
Vision PV value sparsity
Connector activation sparsity
```

特别注意：

```text
Softmax probability 的 exact zero ratio
```

在 BF16/FP32 下可能很低，不能因为 QK/VLM 其他 activation 有 bit sparsity 就默认 attention probability 也具备同样可跳过的结构稀疏。

---

# 28. V6 Integrated Candidate

只有 V1–V4 分别通过后才组合最终 precision map。

可能出现的候选，例如：

```text
Vision Connector            FP8
Vision MLP                  W4A8
Vision Attention Proj       FP8
Vision QK/PV                FP8
VLM                         current chosen config
Action Expert               current chosen config
Patch Embedding             BF16
LayerNorm / Softmax / GELU  BF16/FP32
```

这只是示例，不预设结果。

真实 V6 必须由 V2/V3/V4 sensitivity 结果决定。

---

# 29. Full benchmark 触发条件

只有同时满足：

- [ ] Goal×100 退化在可接受区间；
- [ ] 无 NaN/Inf；
- [ ] raw wrapper equivalence 全过；
- [ ] routing count 全对；
- [ ] scale coverage 100%；
- [ ] latency 至少没有明显反常增加；
- [ ] workload manifest 完整；

才跑：

```text
Spatial ×100
Object ×100
Goal ×100
Long ×100
```

---

# 30. 每阶段建议结果表

## 30.1 Accuracy

| Config | Connector | V-MLP | V-AttnProj | V-QKPV | Goal SR | Δ parent | Δ CTRL |
|---|---|---|---|---|---:|---:|---:|
| CTRL | raw | raw | raw | raw | | | |
| V1 | FP8 | raw | raw | raw | | | |
| V2-FP8 | FP8 | FP8 | raw | raw | | | |
| V2-W4 | FP8 | W4 | raw | raw | | | |
| V3-FP8 | FP8 | FP8 | FP8 | raw | | | |
| V4-FP8 | FP8 | chosen | chosen | FP8 | | | |

## 30.2 Workload

| Component | Dense GFLOPs | % total | Precision | BOP proxy | latency ms | % latency |
|---|---:|---:|---|---:|---:|---:|
| Vision MLP | | | | | | |
| Vision attn proj | | | | | | |
| Vision QK/PV | | | | | | |
| Connector | | | | | | |
| VLM | | | | | | |
| Expert | | | | | | |

## 30.3 Sparsity

| Component/op | native zero | bit sparsity | effective op reduction | outlier ratio |
|---|---:|---:|---:|---:|
| Vision fc1 | | | | |
| Vision fc2 | | | | |
| Vision q_proj | | | | |
| Vision qk | | | | |
| Vision pv | | | | |

---

# 31. 常见错误与 STOP 条件

## Error A — 旧 G6 wrapper count 从 224 变了

说明 backward compatibility 破坏。

```text
STOP
```

---

## Error B — config 里声明 3 cameras，所以 profiler 乘 3

错误。

当前 FLOPs 必须按 runtime Vision call 数。

---

## Error C — 把 Pixel Shuffle 当成 70% FLOPs 来源

错误。

真正昂贵的是 Pixel Shuffle **之前** 1024-token Vision Transformer。

---

## Error D — V4 patch 后 raw 模式都不一致

说明 attention patch 实现错误或 backend 不等价。

```text
STOP
```

---

## Error E — calibration scale 被不同阶段复用

会让结果不可归因。

每个 config 独立 scale_dir。

---

## Error F — V2 同时换 bit、outlier ratio、scale method

不可解释。

一次只改 precision target。

---

## Error G — 用 theoretical FLOPs 推 GPU speedup

错误。

必须单独 measured latency。

---

# 32. 推荐执行顺序

严格按下面顺序：

```text
[ ] 1. 创建 phaseI 实验目录
[ ] 2. 运行 V0 静态 + runtime workload audit
[ ] 3. 保存 Vision config / shapes / camera calls / FLOPs
[ ] 4. 修改 model_wrapper：显式 vision/connector opt-in
[ ] 5. 跑 legacy regression，确认旧 config 224/64 不变
[ ] 6. 实现 Connector wrapping
[ ] 7. V1 raw equivalence
[ ] 8. V1 calibration-only
[ ] 9. V1 task0×1
[ ] 10. V1 Goal×100
[ ] 11. 实现 Vision MLP wrapper
[ ] 12. V2 routing=24 new MLP sites
[ ] 13. V2 raw/calibration/smoke/Goal
[ ] 14. V2 FP8 vs W4
[ ] 15. 实现 Vision attention projection wrapper
[ ] 16. V3 routing=48 new projection sites
[ ] 17. V3 raw/calibration/smoke/Goal
[ ] 18. V3 FP8 vs W4
[ ] 19. 审计 vision attention backend
[ ] 20. 实现 eager-only Vision QuantizedMatMul hook
[ ] 21. V4 raw patch equivalence
[ ] 22. V4 routing=24 Vision MatMul
[ ] 23. V4 FP8 calibration/smoke/Goal
[ ] 24. 生成 integrated V6 precision map
[ ] 25. Goal×100 final check
[ ] 26. full 4-suite evaluation
[ ] 27. latency / sparsity / BOP / FLOPs 总结
[ ] 28. 决定是否值得做 V5 Conv2d
```

---

# 33. 本阶段最终应该回答的研究问题

完成后，应能定量回答：

1. SmolVLA 一次 inference 中 Vision / VLM / Expert 分别占多少 dense FLOPs？
2. Vision 内部 MLP、attention projection、QK/PV 各占多少？
3. Vision FP8 对 LIBERO closed-loop SR 的影响是多少？
4. Vision MLP 是否可以比 Vision attention 更激进地降到 W4？
5. Vision QK/PV 是否比 Linear 更敏感？
6. Vision activation 是否存在可利用的 zero/bit sparsity？
7. 把 Vision 纳入量化后，全模型 precision-weighted BOP 可以降低多少？
8. 理论 BOP 降低是否真正转化为 measured latency speedup？
9. Vision 是否已经成为量化后新的 latency / bandwidth bottleneck？
10. Patch Conv2d 只有约 0.5% Vision FLOPs，是否值得为完整量化单独设计硬件支持？

---

# 34. 建议的最终论文图

## Figure A — SmolVLA inference FLOPs breakdown

```text
Vision Encoder
 ├─ MLP
 ├─ Attention projection
 └─ QK/PV
VLM prefill
Expert denoise
```

## Figure B — Quantization coverage progression

```text
Current framework
→ +Connector
→ +Vision MLP
→ +Vision Attn Proj
→ +Vision QK/PV
```

横轴阶段，纵轴：

```text
% total inference FLOPs covered by quantization
```

## Figure C — SR vs effective BOP

每个点：

```text
CTRL / V1 / V2 / V3 / V4 / V6
```

## Figure D — Dense FLOPs vs measured latency

用于说明：

```text
FLOPs 占比 ≠ latency 占比
```

## Figure E — Per-component sparsity

```text
Vision MLP
Vision Attn
VLM
Expert
```

---

# 35. 证据锚点与代码来源（2026-09-15 核查）

实验实施前建议把这些 revision / URL 写入本次 manifest：

## 当前项目

```text
https://github.com/wolf111108/SmolVLA_qtrsc
```

重点核查：

```text
src/vla_tcs2/model_wrapper.py
src/vla_tcs2/quant_linear.py
src/vla_tcs2/quant_matmul.py
src/vla_tcs2/runtime_context.py
src/vla_tcs2/quant/stat_manager.py
main.py
experiments/
```

## 官方 LeRobot SmolVLA

```text
https://github.com/huggingface/lerobot
```

重点：

```text
src/lerobot/policies/smolvla/modeling_smolvla.py
src/lerobot/policies/smolvla/smolvlm_with_expert.py
```

## Transformers SmolVLM

```text
https://github.com/huggingface/transformers
```

重点：

```text
src/transformers/models/smolvlm/modeling_smolvlm.py
```

实施 V4 时必须以**远端实际安装版本**中的该文件为准，不要只以 GitHub main 为准。

---

# 36. 一句话执行原则

> **先把 Vision 当 workload 准确测清，再从最安全的 Connector → 最大算力的 MLP → Attention projection → QK/PV 逐层扩大量化覆盖；每一步都以 legacy regression、raw equivalence、calibration coverage 和 closed-loop SR 为 gate，最后才组合成全模型候选。**
