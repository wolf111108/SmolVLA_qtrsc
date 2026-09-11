# SmolVLA 量化稀疏（Quantized Sparsity）研究与工程改造手册

> 项目：`wolf111108/SmolVLA_qtrsc`  
> 基准日期：2026-09-08  
> 研究目标：面向硬件部署，把 SmolVLA 的 **量化、稀疏、执行阶段、算子形状和硬件可跳过工作量** 连成一条可验证链路。
>
> 核心目标不是简单报告“稀疏率”，而是建立：
>
> \[
> \boxed{\text{phase} \rightarrow \text{module} \rightarrow \text{quantized tensor} \rightarrow \text{sparsity} \rightarrow \text{skippable work} \rightarrow \text{hardware gain}}
> \]

---

# 1. 当前项目状态与前置动作

## 1.1 当前公开仓库与本地 H100 工作树存在差异

截至 2026-09-08，公开 GitHub `main` 中的：

```text
src/vla_tcs2/quant/stat_manager.py
```

仍然是较旧的 bookkeeping 版本，其注释明确写着：

```text
prefill/decode phase sparsity accounting: dropped
CSV export helpers: dropped
unit sparsity counters: dropped
collect_quant_activation: bookkeeping-only stub
```

但 2026-09-08 H100 工作日志已经记录，本地工作树中已经完成：

```text
compute_sparse_stats
compute_sparse_stats_fp
compute_unit_sparsity_int/fp
collect_quant_activation
collect_weight_sparsity
collect_model_weight_sparsity
全局 + per-phase counters
CSV export
smoke tests
```

因此第一步应先：

```text
本地 sparsity StatManager
    ↓
提交正式 commit
    ↓
push 到主仓库
    ↓
再继续 phase / flow-step / workload 扩展
```

否则不同机器从 GitHub 拉取后会得到不同的统计语义。

建议先执行：

```bash
python scripts/test_sparsity_stat_smoke.py
python -m py_compile src/vla_tcs2/quant/stat_manager.py
git status
git rev-parse HEAD
```

并把 commit hash 写入后续每次 sparsity profile 的 metadata。

---

# 2. 研究问题

本阶段建议围绕以下四个核心研究问题展开。

## RQ1：SmolVLA 稀疏性在两个运行阶段如何分布？

比较：

```text
VLM prefill
vs
Action Expert denoise
```

分别统计：

```text
Linear activation/output
QK A/B/O
PV A/B/O
weight
```

---

## RQ2：10 个 flow-matching denoise step 的稀疏性是否随时间变化？

研究：

\[
S_{expert}(t),\quad t=0,1,\ldots,9
\]

因为每一步的 action state `x_t` 都不同，不能默认 10 次 Expert forward 具有相同分布。

---

## RQ3：不同量化方案如何改变稀疏性？

重点比较：

```text
FP8 PoT per-tensor no-OP
FP8 PoT + dynamic outlier protection
FP8 blockwise PoT
A8/W4/O8
scale granularity: per_site / per_component / ...
```

关注：

```text
精度
元素零率
bit sparsity
unit sparsity
side-path 比例
active BOP
```

---

## RQ4：观察到的稀疏能否真正转化成硬件收益？

最终必须从：

```text
统计稀疏
```

推进到：

```text
可跳 MAC
可跳 bit-op
带宽减少
SRAM/HBM traffic
latency
energy
```

不能直接用 “90% bit sparsity = 10× speedup” 这种推断。

---

# 3. 必须区分的稀疏定义

## 3.1 Element sparsity

\[
S_{elem}=\frac{\#\{x_i=0\}}{N}
\]

这是量化 code 中 exact zero 的比例。

对于硬件来说，它可以作为理论 MAC skip 的上界，但是否能真正跳过取决于数据布局和调度。

---

## 3.2 Bit sparsity

\[
S_{bit}=\frac{N_{zero\ bits}}{N_{total\ bits}}
\]

适合分析：

```text
bit-serial
bit-slice
CIM
shift-and-add
```

当前本地移植已经包含多种 INT bit 统计和 FP significand 统计。

### FP 口径必须明确

当前日志说明 FP E4M3/E2M1/E5M10 的统计更接近：

```text
1MMM
```

而不是完整：

```text
SEEEE MMM
```

因此正式图表建议写：

```text
FP8 significand-bit sparsity (1MMM convention)
```

不要含糊写成“FP8 全 8-bit sparsity”。

---

## 3.3 Unit sparsity

例如硬件一次处理：

```text
2×2
1×4
4×1
```

unit，可以统计 unit 是否全零、unit 内 active bits 等。

### 原则

unit shape 必须对应未来真实硬件 datapath。

不要仅因为旧工程默认 2×2 就永久沿用 2×2。

---

## 3.4 Structured sparsity

第二阶段建议增加：

```text
block-zero ratio
channel-zero ratio
N:M sparsity
```

例如：

```text
2:4
4:8
block16
block32
```

因为硬件常常更容易利用规则稀疏，而不是随机 unstructured zeros。

---

## 3.5 Effective hardware sparsity

最终真正关心的是：

\[
S_{MAC-skip}
\]

或：

\[
S_{bitop-skip}
\]

它必须结合：

```text
tensor sparsity
GEMM shape
operand reuse
hardware execution granularity
```

一起计算。

---

# 4. SmolVLA 推理数据流与统计阶段

当前 deployment inference 主路径：

```text
Images / Language / State
          │
          ▼
      embed_prefix
          │
          ▼
┌───────────────────────┐
│ VLM prefill × 1       │
│ 16 transformer layers │
└───────────────────────┘
          │
          ▼
      KV cache
          │
          ▼
Initial action noise x1
          │
          ▼
┌────────────────────────────┐
│ flow matching             │
│ denoise step 0            │
│ denoise step 1            │
│ ...                       │
│ denoise step 9            │
│                           │
│ 每步 Expert ×16 layers     │
└────────────────────────────┘
          │
          ▼
     action_out_proj
          │
          ▼
      action chunk
```

默认：

```text
num_steps = 10
```

因此一次 action generation 的主要 Transformer 工作量近似：

\[
1\times VLM\ prefill + 10\times Expert\ denoise
\]

---

# 5. 第一阶段正确 phase 划分

## 5.1 Prefill

Inference 中：

```text
inputs_embeds = [prefix_embs, None]
```

因此：

```text
vlm.layers.*.*
vlm.layer.*.qk
vlm.layer.*.pv
```

打标签：

```text
phase = prefill
component = vlm
flow_step = -1
```

---

## 5.2 Denoise

每个 flow step：

```text
inputs_embeds = [None, suffix_embs]
```

因此：

```text
expert.layers.*.*
expert.layer.*.qk
expert.layer.*.pv
```

打标签：

```text
phase = denoise
component = expert
flow_step = 0..9
```

---

# 6. 为什么必须进一步拆 10 个 denoise step

如果只输出：

```text
Expert average sparsity = X%
```

会丢掉 flow-matching 最重要的时间维度。

因为：

```text
step 0: x_t 接近初始 action noise
step 1: Euler 更新一次
...
step 9: 越来越接近最终 action
```

建议得到：

\[
S_{expert}(0),S_{expert}(1),\ldots,S_{expert}(9)
\]

后续可能产生：

```text
early denoise → lower precision / different sparse schedule
late denoise  → higher precision
```

这种 VLA 特有的 temporal mixed precision / temporal sparsity 设计。

---

# 7. Weight sparsity 与 activation sparsity 必须分开

## 7.1 Weight

Weight 是静态数据：

```text
VLM weight
Expert weight
```

所以：

```text
每个 physical module 统计一次
```

不能因为 Expert 被调用 10 次就重复累计 10 次 weight sparsity。

---

## 7.2 Activation

Activation 是 runtime-dependent：

```text
observation-dependent
flow-step-dependent
quantization-dependent
```

因此应该按真实 forward 累计。

---

## 7.3 Hardware workload 再引入 call_count

Weight sparsity只存一次。

硬件成本计算时再使用：

```text
Expert weight reused ×10
```

进行访存/计算建模。

---

# 8. MatMul 必须分别统计 A / B / O

QK：

\[
A=Q,\quad B=K^T
\]

PV：

\[
A=P=softmax(QK^T),\quad B=V
\]

因此不能只写：

```text
expert.layer.7.qk sparsity
```

必须至少有：

```text
expert.layer.7.qk.A
expert.layer.7.qk.B
expert.layer.7.qk.O

expert.layer.7.pv.A
expert.layer.7.pv.B
expert.layer.7.pv.O
```

---

# 9. MatMul operand origin 建议一起记录

## VLM prefill

QK：

```text
A_origin = vlm_query
B_origin = vlm_key
```

PV：

```text
A_origin = attention_probability
B_origin = vlm_value
```

---

## Expert self-attention

建议：

```text
A_origin = expert_suffix_query
B_origin = prefix_plus_suffix_kv
```

---

## Expert cross-attention

建议：

```text
A_origin = expert_suffix_query
B_origin = prefix_cache_projected_kv
```

这对硬件很重要，因为 B 可能来自 KV cache 读，而不是普通 streamed activation。

---

# 10. 不要用 layer parity 猜 attention 类型

即使当前配置通常按 self/cross 交错，也不要写死：

```python
if layer_idx % 2 == 0:
    attention_kind = "self"
```

应该利用真实函数调用：

```text
wrapped_forward_attn_layer       → attention_kind=self
wrapped_forward_cross_attn_layer → attention_kind=cross
```

这样以后修改模型配置仍然正确。

---

# 11. 新增统一 Runtime Context

建议新建：

```text
src/vla_tcs2/runtime_context.py
```

内容：

```python
from contextvars import ContextVar

CURRENT_PHASE = ContextVar(
    "vla_phase",
    default="unknown",
)

CURRENT_FLOW_STEP = ContextVar(
    "vla_flow_step",
    default=-1,
)

CURRENT_ATTN_KIND = ContextVar(
    "vla_attn_kind",
    default="unknown",
)

CURRENT_GENERATION_ID = ContextVar(
    "vla_generation_id",
    default=-1,
)


def get_runtime_context():
    return {
        "phase": CURRENT_PHASE.get(),
        "flow_step": CURRENT_FLOW_STEP.get(),
        "attention_kind": CURRENT_ATTN_KIND.get(),
        "generation_id": CURRENT_GENERATION_ID.get(),
    }
```

优点：

```text
不修改 lerobot_current
与现有 MatMul ContextVar 路由一致
try/finally 可安全恢复
未来可拓展 thread/reentrant 场景
```

---

# 12. 自动设置 prefill / denoise，而不是由 runner 手工 set_phase

外层 evaluation 并不知道：

```text
predict_action_chunk 内部什么时候 prefill
什么时候进入 10 次 denoise
```

所以 phase 必须自动 instrumentation。

---

# 13. `model_wrapper.py` 中 monkey-patch `sample_actions`

保存：

```python
original_sample_actions = flow_model.sample_actions
```

进入一次新的 generation 时：

```python
CURRENT_PHASE.set("prefill")
CURRENT_FLOW_STEP.set(-1)
```

并把 denoise counter 清零。

`sample_actions()` 本身先执行 prefix VLM forward，再调用 `euler_integrate()`，因此默认 context 可以设为 `prefill`。

---

# 14. monkey-patch `denoise_step`

保存：

```python
original_denoise_step = flow_model.denoise_step
```

每次进入：

```python
phase_token = CURRENT_PHASE.set("denoise")
step_token = CURRENT_FLOW_STEP.set(step_idx)
```

结束：

```python
finally:
    CURRENT_FLOW_STEP.reset(step_token)
    CURRENT_PHASE.reset(phase_token)
```

并：

```text
step_idx += 1
```

每次新的 `sample_actions()` 必须重置成 0。

---

# 15. 推荐由 StatManager 管理 generation / step counter

提供：

```python
def begin_generation(self):
    self.current_generation += 1
    self.current_flow_step = -1


def begin_denoise_step(self):
    self.current_flow_step += 1
    return self.current_flow_step
```

这样 runtime context 只负责传递标签，不负责持久状态。

---

# 16. Attention wrapper 增加 `attention_kind`

当前项目已经 wrapper：

```text
forward_attn_layer
forward_cross_attn_layer
```

只需在现有 ContextVar site routing 上增加：

```text
CURRENT_ATTN_KIND=self
CURRENT_ATTN_KIND=cross
```

MatMul 的 physical routing 不需要改变。

---

# 17. 稀疏统计 identity 必须用 `module_id`

当前工程已经有：

```text
vlm.layers.0.self_attn.q_proj
expert.layers.0.self_attn.q_proj
vlm.layer.0.qk
expert.layer.0.qk
```

所有 sparsity key 应使用 `module_id`。

绝对不要重新退化到：

```text
q_proj_0
```

否则 VLM/Expert 会再次混合。

---

# 18. `scale_group` 不能作为 sparsity identity

例如 global scale 下：

```text
64 physical MatMul
```

可能共享少量 scale group，但它们 runtime distribution 仍然不同。

所以：

```text
module_id  → 运行/稀疏 identity
scale_group → 量化 scale sharing identity
```

必须继续保持解耦。

---

# 19. 重构 `collect_quant_activation` API

当前历史接口依赖 positional args，越来越难维护。

建议新增统一接口：

```python
collect_quant_tensor(
    *,
    module_id,
    operator,
    tensor_role,
    tensor_code,
    spec,
    shape,
    runtime_context,
    original_tensor=None,
    metadata=None,
)
```

Linear 示例：

```python
collect_quant_tensor(
    module_id=layer.module_id,
    operator="down_proj",
    tensor_role="activation",
    tensor_code=x_code,
    spec=layer.a_spec,
    runtime_context=get_runtime_context(),
)
```

MatMul 示例：

```python
collect_quant_tensor(
    module_id=layer.module_id,
    operator="qk",
    tensor_role="A",
    tensor_code=A_sim,
    spec=layer.A_spec,
    metadata={"operand_origin": "expert_suffix_query"},
    runtime_context=get_runtime_context(),
)
```

---

# 20. 保留旧接口作为 compatibility adapter

短期不要一次性删掉：

```text
collect_quant_activation(...)
```

可以：

```python
def collect_quant_activation(...):
    # parse legacy signature
    # convert to structured fields
    return self.collect_quant_tensor(...)
```

这样现有 quant_methods 不会一次性全部失效。

---

# 21. 推荐统一统计 key

在线 aggregate key：

```python
(
    module_id,
    phase,
    flow_step,
    attention_kind,
    tensor_role,
    quant_format,
)
```

value 保存累计计数。

不要每次 forward 保存一整行 dataframe，否则 full eval 数据量太大。

---

# 22. 推荐累计 counter

```text
call_count
sum_total_elements
sum_zero_elements
sum_total_bits
sum_zero_bits
sum_sm_total_bits
sum_sm_zero_bits
sum_unit_total
sum_unit_zero
```

最后再计算 ratio。

---

# 23. Global / phase sparsity 必须“先累加分子分母，再求比例”

错误：

\[
S=\frac{1}{L}\sum_l S_l
\]

正确：

\[
S_{elem}=\frac{\sum_l Z_l}{\sum_l N_l}
\]

\[
S_{bit}=\frac{\sum_l Z_{bit,l}}{\sum_l B_l}
\]

Prefill / denoise 同理。

---

# 24. Weight statistics 数据结构

建议单独：

```text
per_module_weight_sparsity[module_id]
```

记录：

```text
module_id
component
layer
operator
weight_shape
quant_format
scale_group
scale_value
zero elements
bit sparsity
unit sparsity
```

---

# 25. `collect_model_weight_sparsity()` 对 dynamic OP 的 caveat

对于：

```text
per_tensor
pot_fp8_per_tensor
```

可以静态执行：

```text
weight → quant_awo → code → sparsity
```

这是准确的。

但当前 dynamic OP 的 weight 路径包括：

```text
activation outlier channel 对应的 weight columns → FP
+
weight element top-k → FP
```

第一部分取决于 runtime activation mask。

因此 dynamic OP 下不存在唯一静态“实际量化 weight tensor”。

所以静态 weight profile 必须标记为：

```text
potential main-quantizer weight sparsity
```

而 runtime OP path 要另做 mask/sidepath 统计。

---

# 26. Dynamic outlier protection 必须新增 side-path 统计

每次 forward 至少记录：

```text
activation_protected_fraction
weight_protected_fraction
output_protected_fraction
```

不能只统计 normal FP8/W4 code，然后假装整个算子都在这条稀疏路径上。

---

# 27. 当前 OP 的四条计算路径也要记录

当前实现近似：

\[
(X_q+X_f)(W_q+W_f)
\]

展开：

\[
X_qW_q + X_fW_f + X_fW_q + X_qW_f
\]

软件里确实有四项贡献。

建议 collector 记录理论工作量：

```text
qa_qb_MACs
fa_fb_MACs
fa_qb_MACs
qa_fb_MACs
```

这样才能评估：

> 当前 dynamic OP 如果原样硬件实现，是否抵消了量化/稀疏收益？

---

# 28. 新增 outlier mask 稳定性统计

这是决定 dynamic OP 能否改成 static OP 的关键。

## 28.1 Channel hit frequency

\[
f_j=\frac{\#\text{calls where channel j is outlier}}{\#\text{calls}}
\]

如果某些 channel：

```text
f_j ≈ 0.8~1.0
```

说明 outlier 很稳定，适合离线固定。

---

## 28.2 Jaccard similarity

两个 runtime mask：

\[
J=\frac{|M_a\cap M_b|}{|M_a\cup M_b|}
\]

高 Jaccard：

```text
dynamic top-k → 可考虑 static mask
```

---

## 28.3 Energy coverage

对 static mask S：

\[
C_E=\frac{\sum_{i\in S}x_i^2}{\sum_i x_i^2}
\]

比较：

```text
dynamic top1%
static top1%
static top2%
```

看能覆盖多少 activation energy。

---

# 29. 为什么 outlier 稳定性应列为高优先级

当前 dynamic OP 的软件 accuracy 很好，但硬件代价包括：

```text
runtime absmax
top-k
mask
高精度 side path
cross terms
output 再 top-k
```

如果 outlier channel 高度稳定，就可改成：

```text
offline static channel list
```

硬件变成：

```text
main FP8/W4 GEMM
+
small FP16 sidecar
```

明显更规则。

---

# 30. QuantizedLinear 需要怎样改

在：

```text
src/vla_tcs2/quant/quant_methods.py
```

逐渐把现有：

```python
collect_quant_activation(
    layer.layer_name,
    layer.layer_idx,
    ...
)
```

改成：

```python
collect_linear_runtime(
    module_id=layer.module_id,
    layer=layer,
    input_code=x_code,
    input_fp=x,
    output_code=out_code,
    runtime_context=get_runtime_context(),
)
```

---

# 31. Linear 建议统计 input + output

最少：

```text
input activation code
output quantized code
```

原因：output 会影响：

```text
buffer write traffic
下一阶段输入分布
是否可压缩
```

只统计 input 不够完整。

---

# 32. Weight 不要每次 forward 重新累计

`w_code` 虽然在 quant_forward 中每次生成，但统计应该：

```text
per module once
```

可用：

```python
if not stat_manager.weight_seen(module_id):
    ...
```

或者完全由 `collect_model_weight_sparsity()` 管理。

---

# 33. QuantizedMatMul collector 重构

建议：

```python
collect_matmul_runtime(
    module_id=layer.module_id,
    A_code=A_sim,
    A_spec=layer.A_spec,
    B_code=B_sim,
    B_spec=layer.B_spec,
    O_code=out_code,
    O_spec=layer.O_spec,
    runtime_context=get_runtime_context(),
    metadata={
        "attention_kind": ...,
        "A_origin": ...,
        "B_origin": ...,
    },
)
```

---

# 34. PV-A（softmax probability）需要特殊分析

PV 的 A 是 softmax probability：

```text
值通常 >= 0
大量值可能非常小但非零
```

量化后这些小值可能被变成 exact zero。

建议同时记录：

```text
FP near-zero ratio
quantized exact-zero ratio
```

例如：

```text
|P| < 1e-4
|P| < 1e-3
```

注意：near-zero 只是 pruning 潜力分析，不能直接当成无损 hardware zero。

---

# 35. 建议新增 `near_zero` 可选统计

```yaml
sparsity:
  metrics:
    near_zero: true
  near_zero_thresholds:
    - 1.0e-4
    - 1.0e-3
```

对 raw FP tensor 使用。

真正无损硬件 skip 仍以：

```text
quantized code == 0
```

为准。

---

# 36. 建议 YAML 配置

```yaml
sparsity:
  enabled: true

  mode: quant_code   # raw | quant_code | both

  collect_runtime: true
  collect_weight: true

  phases:
    - prefill
    - denoise

  split_denoise_steps: true

  collect:
    linear_input: true
    linear_output: true

    matmul_A: true
    matmul_B: true
    matmul_output: true

    weight: true

  metrics:
    element_zero: true
    bit_zero: true
    sm_bit_zero: true
    unit_sparsity: true
    near_zero: false

  unit:
    rows: 2
    cols: 2

  near_zero_thresholds:
    - 1.0e-4
    - 1.0e-3

  outlier:
    collect_masks: true
    collect_channel_frequency: true
    collect_jaccard: true
    collect_energy_coverage: true

  sampling:
    max_calls_per_key: 0
    closed_loop_stride: 1

  export:
    dir: outputs/sparsity/fp8_pot_profile
    module_csv: true
    phase_csv: true
    flow_step_csv: true
    weight_csv: true
    unit_csv: true
    outlier_csv: true
    workload_csv: true
```

---

# 37. Sparsity 子系统不要与 quantization.enabled 强绑定

应该允许：

```text
quantization.enabled=false
sparsity.enabled=true
```

用于 raw FP distribution 分析。

同时也允许：

```text
quantization.enabled=true
sparsity.mode=quant_code
```

用于真正量化 code 稀疏。

---

# 38. 建议明确三种 sparsity mode

## `raw`

统计 FP tensor：

```text
exact zero
near-zero
raw distribution
```

## `quant_code`

统计真实：

```text
FP8/INT4/INT8 code
```

这是硬件无损 skip 的主要依据。

## `both`

同时保留：

```text
raw tensor
quantized code
```

用于分析：

> 量化本身创造了多少 exact zero？

---

# 39. `main.py` 修改方案

当前主流程：

```text
build
→ calibration
→ quant_forward
→ evaluation
```

建议加入 Sparsity Setup。

Build 后：

```python
sp_cfg = config.get("sparsity", {})

if sp_cfg.get("enabled", False):
    wrapper.stat_manager.configure_sparsity(sp_cfg)
```

Calibration/scale 可用后：

```python
if sp_cfg.get("collect_weight", False):
    wrapper.stat_manager.collect_model_weight_sparsity(model)
```

Evaluation 前：

```python
wrapper.stat_manager.enable_sparsity()
```

Evaluation 后：

```python
wrapper.stat_manager.export_sparsity_bundle(output_dir)
```

---

# 40. 更推荐增加独立 profile runner

不要所有 profile 都跑完整 LIBERO 100 episodes。

建议新增：

```text
scripts/run_sparsity_profile.py
```

流程：

```text
load config
build wrapped model
reuse/calibrate scales
enable sparsity
prepare deployment-faithful observation batches
predict_action_chunk
export stats
```

这样能快速扫描整个模型。

---

# 41. Profile 数据源建议

## 41.1 Offline deployment-faithful profile

复用 calibration-style 数据采样：

```text
4~8 episodes
frame_stride 4~8
batch=1
```

但每个 observation 都执行：

```text
predict_action_chunk
```

即真实：

```text
prefill ×1
+ denoise ×10
```

---

## 41.2 Closed-loop validation profile

再在真实 LIBERO rollout 中抽样：

```text
每 task 前 N 个 replanning points
```

或：

```text
每 10 个 replan 采 1 个
```

目的：

```text
验证 offline sparsity 是否存在 distribution shift
```

---

# 42. 必须研究 offline vs closed-loop distribution shift

定义：

\[
\Delta S=S_{closed}-S_{offline}
\]

如果差异很小：

```text
后续硬件研究可大量使用便宜 offline trace
```

如果差异明显：

```text
必须保留 closed-loop profile
```

---

# 43. Batch size 建议使用 deployment batch=1

Scale calibration 可以 batch=8。

但 sparsity/hardware profile 建议：

```text
batch=1
```

因为 LIBERO 实际 evaluation 就是 batch1。

不要混淆：

```text
scale calibration batch
vs
hardware profile batch
```

---

# 44. Action noise seed 也必须控制

`sample_actions` 会采 action noise。

建议 profile：

```text
seed 0
seed 1
seed 2
```

检查 flow-step sparsity 对初始 noise realization 的方差。

---

# 45. 推荐输出目录

```text
outputs/sparsity/<experiment_name>/
```

包含：

```text
config.yaml
metadata.json

module_sparsity.csv
phase_sparsity.csv
flow_step_sparsity.csv
weight_sparsity.csv
unit_sparsity.csv
matmul_operand_sparsity.csv
outlier_mask_stats.csv
workload.csv
summary.json
```

---

# 46. `module_sparsity.csv` 推荐列

```text
module_id
component
layer
operator

phase
flow_step
attention_kind

tensor_role
operand_origin

quant_kind
quant_format
bitwidth

calls
shape

total_elements
zero_elements
element_sparsity

total_bits
zero_bits
bit_sparsity

sm_total_bits
sm_zero_bits
sm_bit_sparsity

unit_total
unit_zero
unit_sparsity
```

---

# 47. `weight_sparsity.csv`

```text
module_id
component
layer
operator
weight_shape
quant_format
scale_group
scale_value

total_elements
zero_elements
element_sparsity

total_bits
zero_bits
bit_sparsity

unit_total
unit_zero
unit_sparsity
```

---

# 48. `outlier_mask_stats.csv`

```text
module_id
phase
flow_step
calls

channel_count
protected_channels_per_call
mean_protected_ratio
mean_jaccard

static_top1_energy_coverage
dynamic_top1_energy_coverage
```

对于 hit frequency，建议另存 top channels 或 JSON summary。

---

# 49. `workload.csv` 是硬件研究最重要的输出

每个 operator / phase / flow step 记录：

```text
module_id
phase
flow_step
call_count
op_type

M
K
N
batch
heads

MACs
FLOPs

A_elements
B_elements
O_elements
A_bytes
B_bytes
O_bytes
weight_bytes

scale_bytes
outlier_metadata_bytes

A_element_sparsity
B_element_sparsity
A_bit_sparsity
B_bit_sparsity

estimated_zero_MACs
estimated_active_MACs
estimated_dense_bitops
estimated_active_bitops
```

---

# 50. Linear GEMM shape

如果：

```text
x: [B,T,K]
weight: [N,K]
```

则：

\[
M=B\times T
\]

\[
GEMM=[M,K]\times[K,N]
\]

记录：

```text
M,K,N
```

以及真实 token count。

---

# 51. Activation-zero 的理论 MAC skip

若 activation 零元素数为：

\[
Z_X
\]

每个 activation 沿 N 个 output 重用。

理论可跳 MAC：

\[
MAC_{skip,A}=Z_XN
\]

总 MAC：

\[
MAC_{total}=MKN
\]

比例等于 activation element sparsity。

但这是 upper bound，不等价于真实 speedup。

---

# 52. Weight-zero theoretical skip

Weight 中零元素数：

\[
Z_W
\]

每个 weight 被 M 行 activation 使用。

\[
MAC_{skip,W}=Z_WM
\]

---

# 53. Activation + Weight 稀疏不能简单相加

不能：

\[
S_{skip}=S_A+S_W
\]

若近似独立：

\[
S_{skip}\approx1-(1-S_A)(1-S_W)
\]

真实硬件最好使用 unit/block 统计而不是 independence approximation。

---

# 54. QK/PV workload 要保留 batch/head 维度

QK：

\[
[B,H,T_q,D]\times[B,H,D,T_k]
\]

PV：

\[
[B,H,T_q,T_k]\times[B,H,T_k,D]
\]

不要在 exporter 中只记录 M/K/N 而丢掉：

```text
B
H
Tq
Tk
D
```

这些信息对 attention accelerator 很重要。

---

# 55. Bit-operation proxy

如果平均 active bits：

```text
A = bA
B = bB
```

定义 first-order：

\[
BOP_{active}=MAC\times b_A\times b_B
\]

Dense baseline：

\[
BOP_{dense}=MAC\times B_A\times B_B
\]

则：

\[
Reduction=1-\frac{b_Ab_B}{B_AB_B}
\]

必须在结果中标记：

```text
hardware proxy, not measured latency
```

---

# 56. StatManager 不应重新塞入硬件 simulator

当前从旧 SACIM/EffLoc 只移植统计而不移植 Mapping_stat / CIM latency 是正确方向。

建议：

```text
SmolVLA_qtrsc
→ 输出真实 workload.csv

hardware simulator
→ 读取 workload.csv
→ 做 mapping / cycles / HBM / SRAM / energy
```

保持模型与硬件解耦。

---

# 57. Phase-aware workload 总公式

一次 action generation：

\[
C_{gen}=C_{prefill}+\sum_{t=0}^{9}C_{denoise,t}
\]

而一次 episode 如果执行 T 个动作、每次执行 `n_action_steps=n_a`：

\[
N_{gen}\approx\left\lceil\frac{T}{n_a}\right\rceil
\]

\[
C_{episode}=N_{gen}\cdot C_{gen}
\]

这样可以把你已有的 `n_action_steps` accuracy ablation 和硬件成本连接起来。

---

# 58. Expert weight reuse 必须进入硬件建模

Expert 相同 weights 在一次 action generation 内被复用约 10 次。

因此关键问题：

> Expert weights 能否驻留 SRAM / CIM array，在 10 个 flow steps 中复用？

这会让 W4 的价值不只是参数压缩，还包括：

```text
更容易 on-chip residency
更低 SRAM bandwidth
更少 HBM traffic
```

---

# 59. KV cache 应成为下一阶段统计对象

Prefix KV：

```text
生成一次
→ 10 个 denoise step 重复读取
```

后续建议统计：

```text
KV shape
bytes
read count
raw distribution
FP8 code sparsity
bit sparsity
```

并研究：

```text
persistent FP8 KV cache
```

这对部署很可能比减少几百个 scale metadata 更重要。

---

# 60. S0：统计系统正确性实验

第一阶段必须先做 correctness。

三个 run：

```text
A. sparsity disabled
B. sparsity enabled，仅 call counting
C. 完整 bit sparsity enabled
```

同一固定 input / noise。

要求：

```text
action output numerically equal
```

统计不能改变模型输出。

---

# 61. S0 Call-count audit

单次：

```text
predict_action_chunk
```

检查：

```text
VLM physical modules 有 prefill calls
Expert physical modules 有 denoise calls
Expert calls 约随 10 个 flow steps 累积
```

任何：

```text
module call=0
unexpected ×2 / ×20
```

都需要先解释。

---

# 62. S0 Phase audit

抽查：

```text
vlm.layers.3.mlp.down_proj
expert.layers.3.mlp.down_proj
vlm.layer.3.qk
expert.layer.3.qk
```

要求：

```text
vlm.*    → prefill only
expert.* → denoise only
```

Expert 必须观察到：

```text
flow_step 0..9
```

---

# 63. S0 Weight once-only audit

跑一次 action generation。

断言：

```text
每个 physical weight sparsity record = 1
```

不能因为 Expert forward 10 次变成 10 份 static weight record。

---

# 64. S0 MatMul A/B audit

必须确认：

```text
qk A/B/O 都有统计
pv A/B/O 都有统计
```

并验证 shape 可以构成真实 matmul。

---

# 65. 第一批正式实验 S1：FP8 PoT no-OP

建议先使用最干净的：

```text
pot_fp8_per_tensor
no outlier
per_site scale
```

原因：

```text
没有 FP sidepath
统计语义最清楚
```

得到第一张 reference sparsity map。

---

# 66. S1 必须输出

```text
VLM prefill sparsity
Expert denoise overall
Expert step0..9

Linear q/k/v/o/gate/up/down
QK A/B/O
PV A/B/O

VLM weight
Expert weight
active-BOP proxy
```

---

# 67. S2：FP8 dynamic OP vs no-OP

比较：

```text
pot_fp8_per_tensor
vs
pot_fp8_outlier
```

不只比较 Success Rate，还比较：

```text
normal-path sparsity
protected fraction
FP side-path work
4-term cross-path work
estimated active bitops
```

最终回答：

> OP 带来的 accuracy gain 是否值得硬件成本？

---

# 68. 如果 Phase F 的 no-OP 与 OP 精度接近

那应该直接把 dynamic OP 从主硬件候选中降级。

优先：

```text
plain FP8 PoT
blockwise PoT FP8
```

因为更规则。

---

# 69. S3：W4 sparsity

当前重要候选：

```text
A = FP8
W = INT4
O = FP8
```

重点比较：

```text
per_site
per_component
```

记录：

```text
weight element zero
weight bit sparsity
weight code histogram
activation sparsity
active BOP
```

---

# 70. W4 granularity failure 应用 sparsity 数据解释

已知：

```text
per_layer/global → SR 0%
```

原因与跨 VLM/Expert scale mismatch 有关。

建议对四档额外比较：

```text
code histogram
zero ratio
saturation/clipping
mean active bits
```

看看 scale 抬升如何导致 Expert 有效表示塌缩。

---

# 71. FP8 scale granularity × sparsity 也值得正式测

虽然 FP8 SR 对：

```text
per_site
per_layer
per_component
global
```

不敏感，但量化 code sparsity 可能不同。

因为更大的 scale 可能让更多小值进入：

```text
zero / subnormal
```

潜在结论可能是：

> 更粗 scale 不伤 SR，反而增加 hardware sparsity。

这个方向很有价值，必须实测而不是猜。

---

# 72. S4：Flow-step sparsity

固定两个候选：

```text
FP8 no-OP
A8/W4/O8
```

对：

```text
step0..9
```

分别统计。

重点：

```text
Expert q_proj
Expert down_proj
QK A/B
PV A/B
```

---

# 73. Flow-step × sensitivity 联合分析

可把 Phase E 的 quantization/noise sensitivity 和 sparsity 合并。

四象限：

```text
高 sparsity + 低 sensitivity
→ 最理想低精度/稀疏区域

高 sparsity + 高 sensitivity
→ 可 sparse，但 precision 要保守

低 sparsity + 低 sensitivity
→ 更适合降 bit

低 sparsity + 高 sensitivity
→ 高精度保留
```

---

# 74. S5：Outlier mask stability

对每个 Linear 统计：

```text
hit frequency
Jaccard
energy coverage
```

并按：

```text
VLM prefill
Expert denoise
flow step
```

分组。

---

# 75. 如果 static mask 可行

实现新方法：

```text
static_channel_outlier
```

Calibration：

```text
累计 channel importance
→ 固定 top-r channels
```

Inference：

```text
直接查静态 mask
不做 runtime top-k
```

---

# 76. Static OP 更硬件友好的形式

重新排列 channels：

```text
normal channels | outlier channels
```

然后：

\[
Y=X_NW_N^T+X_OW_O^T
\]

只需要：

```text
main low-precision GEMM
small high-precision sidecar GEMM
```

比当前四项动态 cross-term 规则得多。

---

# 77. S6：Blockwise PoT FP8

建议：

```text
per_tensor
block128
block64
block32
block16
```

同时测：

```text
SR
SQNR
element sparsity
bit sparsity
block-zero ratio
metadata bytes
```

---

# 78. Blockwise 方法应新增 block sparsity

如果量化 block size = 32：

\[
S_{block-zero}=\frac{\#all-zero\ blocks}{\#blocks}
\]

这个指标对 block scheduler 比单个 element zero 更有用。

---

# 79. 推荐正式算法对比

| 方法 | runtime top-k | FP side path | 规则性 | 目标 |
|---|---|---|---|---|
| FP8 per-tensor | no | no | 高 | 最简 baseline |
| Dynamic OP | yes | yes | 低 | accuracy oracle |
| Static channel OP | no | small | 中高 | hardware-friendly rescue |
| Block32 PoT FP8 | no | no | 很高 | 首选替代方案 |
| A8/W4/O8 | no/可选 | 可选 | 高 | weight bandwidth/compute |

---

# 80. 推荐输出图

## 图 1：Component × Operator sparsity heatmap

行：

```text
VLM
Expert
```

列：

```text
q k v o qk.A qk.B pv.A pv.B gate up down
```

颜色：

```text
bit sparsity / active-bit ratio
```

---

## 图 2：Layer heatmap

两个面板：

```text
VLM layer0..15
Expert layer0..15
```

列 operator。

---

## 图 3：Flow-step sparsity curve

横轴：

```text
0..9
```

纵轴：

```text
active-bit ratio
```

多条 operator 曲线。

---

## 图 4：Quantization method vs active BOP

比较：

```text
FP8 per-tensor
FP8 block64
FP8 block32
FP8 static OP
W4
```

同时标出 LIBERO SR。

---

## 图 5：Accuracy–Hardware Cost Pareto

横轴：

```text
normalized hardware cost
```

纵轴：

```text
LIBERO SR
```

候选：

```text
FP
FP8
FP8+dynamic OP
block FP8
W4
W4+static OP
mixed precision
```

---

# 81. Reproducibility metadata

每次 profile 必须保存：

```text
git commit
git dirty
checkpoint
dataset revision
calibration seed
profile seed
action-noise seed

num_steps
n_action_steps

quant method
scale granularity
scale_dir

sparsity metric version
unit shape
```

---

# 82. `num_steps` 与 `n_action_steps` 必须分开

```text
num_steps
= 一次 action generation 内 denoise 次数
= 10
```

```text
n_action_steps
= 生成 chunk 后执行多少 action 再 replan
= 当前常用 10
```

Sparsity 的 flow-step 主要由：

```text
num_steps
```

决定。

Episode 总硬件成本同时受：

```text
n_action_steps
```

影响。

---

# 83. 建议新增 sparsity 子模块

如果后续继续扩展，不建议把所有代码塞在 `stat_manager.py`。

建议：

```text
src/vla_tcs2/sparsity/
    __init__.py
    metrics.py
    runtime_context.py
    records.py
    outlier_stats.py
    workload.py
    exporter.py
```

职责：

```text
metrics.py         → zero/bit/unit/block metrics
runtime_context.py → phase/flow step/attn kind
records.py         → dataclass/counters
outlier_stats.py   → hit/Jaccard/energy
workload.py        → MKN/MAC/BOP/bytes
exporter.py        → CSV/JSON
```

StatManager 保留协调职责。

---

# 84. 第一阶段可以暂时不大重构

为了快速出结果，可以先：

```text
保留 QuantStatManager
+
新增 runtime_context.py
+
新增 workload exporter
+
新增 outlier stability stats
```

等统计定义稳定后再拆目录。

---

# 85. 必须新增的自动测试

建议：

```text
scripts/test_sparsity_phase_context.py
scripts/test_sparsity_identity.py
scripts/test_sparsity_weight_once.py
scripts/test_sparsity_matmul_operands.py
scripts/test_sparsity_export.py
```

---

# 86. `test_sparsity_identity.py`

断言：

```text
所有 physical QuantizedLinear module_id unique
所有 QuantizedMatMul module_id unique
```

当前标准 Transformer 配置的实验记录通常为：

```text
224 Transformer Linear
64 MatMul
288 sites
```

如果 config 把 action head/misc 也纳入 wrapper，则以实际 audit count 为准。

---

# 87. `test_sparsity_phase_context.py`

跑一次：

```text
predict_action_chunk
```

断言：

```text
vlm.*    only prefill
expert.* only denoise
flow steps = 0..9
```

---

# 88. `test_sparsity_weight_once.py`

断言：

```text
每个 weight record exactly once
```

即使 Expert forward 被调用 10 次。

---

# 89. `test_sparsity_matmul_operands.py`

检查：

```text
QK A/B/O
PV A/B/O
```

全部有记录，且 shape 正确。

---

# 90. `test_sparsity_export.py`

用 synthetic tensor 人工构造：

```text
known zeros
known INT bits
known FP codes
```

验证：

```text
per-module counts
phase aggregation
global aggregation
CSV values
```

精确匹配手算。

---

# 91. 近期实验矩阵建议

第一轮只做 5 组：

| ID | Quant | OP | Scale | 目标 |
|---|---|---|---|---|
| S1 | FP8 PoT | no | per_site | clean baseline |
| S2 | FP8 PoT | dynamic | per_site | OP 代价 |
| S3 | FP8 PoT | no | per_component | coarse scale sparsity |
| S4 | A8/W4/O8 | dynamic | per_site | W4 sparsity |
| S5 | A8/W4/O8 | dynamic | per_component | shared W4 candidate |

先使用 offline deployment-faithful profile，不需要一开始跑 full 100-episode SR。

Accuracy 可与已有 D4/Phase F 数据关联。

---

# 92. 每组最少输出

```text
Prefill element sparsity
Prefill bit sparsity

Denoise overall element sparsity
Denoise overall bit sparsity

Denoise step0..9

VLM weight sparsity
Expert weight sparsity

Linear activation/output
QK A/B
PV A/B

active-BOP proxy
```

---

# 93. 不要使用 `outlier_ratio=0` 作为 no-OP 对照

当前项目已经发现历史问题：

```text
get_outlier_mask_channel:
k = max(1, ...)
```

因此 ratio=0 仍可能保护至少一个 channel。

No-OP 必须使用独立方法：

```text
pot_fp8_per_tensor
```

---

# 94. FP8 建议增加 exponent histogram

E4M3 可以统计：

```text
zero
subnormal
exponent bins
max/clipped
```

这能帮助解释：

> 为什么 shared PoT scale 很粗仍然不掉 SR？

---

# 95. INT4 建议保留完整 code histogram

INT4 只有有限 code：

```text
-8..7
```

16-bin histogram 很便宜。

可以直接看到：

```text
scale mismatch 后是否大量集中 0/±1
是否发生 clipping
```

这对解释 W4 collapse 特别有用。

---

# 96. PoT exponent 与 sparsity 联合分析

记录：

\[
k=\log_2(scale)
\]

然后画：

\[
k \rightarrow element\ sparsity
\]

尤其比较：

```text
per_site
per_component
per_layer
global
```

看 scale 抬高一个 PoT 档如何改变 zero/subnormal 分布。

---

# 97. Blockwise PoT 后新增 shared exponent 统计

block32：

```text
每 block 一个 k
```

统计：

```text
k histogram
neighboring-block Δk
block-zero ratio
```

这可以进一步指导 exponent metadata bit width 和硬件 align 单元。

---

# 98. 稀疏“结构性”最终必须研究

总 sparsity 之后，逐步增加：

```text
per-channel zero frequency
per-token sparsity
block sparsity
N:M
flow-step stability
```

因为硬件收益高度依赖规则性。

---

# 99. 第二阶段可研究 token sparsity

对于：

```text
[B,T,H]
```

统计每个 token：

\[
S_t
\]

尤其 Expert action chunk 有多个 action tokens。

可进一步研究：

```text
flow step × action-token index × sparsity
```

但这属于 P3，不建议现在先展开。

---

# 100. Hardware candidate 不应仅按 SR 排序

建议考虑：

\[
Score=f(Accuracy,Regularity,Sparsity,Metadata,ControlCost)
\]

Dynamic OP 即使 SR 高，也因为：

```text
top-k
mask
FP side path
cross terms
```

需要较大 hardware penalty。

---

# 101. 推荐候选路线

```text
C0: FP baseline

C1: FP8 PoT per-tensor no OP

C2: FP8 PoT block32 no OP

C3: FP8 PoT + static channel OP

C4: A8/W4/O8 + group-wise W4

C5: A8/W4/O8 + static OP

C6: sensitivity-guided mixed precision
```

每个 candidate 都输出：

```text
SR
bit sparsity
unit/block sparsity
active BOP
memory bytes
metadata/control cost
```

---

# 102. Hardware simulator 何时开始接入

只要以下字段稳定：

```text
module_id
phase
flow_step
shape
call_count
quant format
sparsity
outlier sidepath fraction
```

就可以开始硬件模拟。

不需要等所有四 suite 量化实验都完成。

---

# 103. 最小可用 workload exporter 完成标准

一次：

```text
predict_action_chunk
```

能够导出：

```text
每个 module
每个 phase
每个 flow step
M/K/N
calls
bytes
quant format
active bits
```

就可以连接你的 CIM / sparse accelerator 模型。

---

# 104. 推荐优先级

## P0 — 必须立即完成

```text
1. 提交当前本地 StatManager sparsity 实现
2. 所有统计 key 改用 module_id
3. 自动 prefill/denoise context
4. flow_step 0..9
5. attention_kind self/cross
6. MatMul A/B/O 分开
7. weight once-only
8. workload.csv
9. correctness audits
```

## P1 — 紧接着做

```text
1. FP8 no-OP baseline sparsity
2. FP8 dynamic OP sidepath cost
3. W4 sparsity
4. scale granularity vs code sparsity
5. outlier mask stability
```

## P2 — Hardware-friendly algorithm

```text
1. static channel OP
2. blockwise PoT FP8
3. group-wise W4
4. KV cache FP8/sparsity
```

## P3 — 深层优化

```text
flow-step mixed precision
token-level sparsity
structured/N:M
sparse scheduling
```

---

# 105. 当前不建议优先做的工作

暂时不要把大量时间放在：

```text
继续堆 Gaussian sensitivity 点
全 suite 大规模 sparsity profile
复杂 pruning
把硬件 latency 代码重新塞入 StatManager
```

目前最缺的是：

```text
正确、结构化、phase-aware、flow-step-aware 的真实 workload trace
```

---

# 106. 推荐实施顺序（工程 checklist）

```text
[ ] A1  将本地 sparsity StatManager push 到主仓库
[ ] A2  增加 module_id-based collector
[ ] A3  新建 runtime_context.py
[ ] A4  wrapper sample_actions
[ ] A5  wrapper denoise_step
[ ] A6  wrapper attention_kind
[ ] A7  Linear input/output structured collector
[ ] A8  MatMul A/B/O structured collector
[ ] A9  weight once-only
[ ] A10 outlier sidepath counters
[ ] A11 outlier mask stability
[ ] A12 workload.csv

[ ] B1  S0 raw-equivalence
[ ] B2  S0 phase audit
[ ] B3  S0 call-count audit
[ ] B4  S0 weight once-only audit

[ ] C1  S1 FP8 PoT no-OP profile
[ ] C2  S2 FP8 dynamic-OP profile
[ ] C3  S3 W4 profile
[ ] C4  S4 flow-step profile
[ ] C5  scale granularity × sparsity

[ ] D1  static OP
[ ] D2  block32 PoT FP8
[ ] D3  group-wise W4
[ ] D4  KV cache quantization

[ ] E1  export to hardware simulator
[ ] E2  roofline / latency / energy
[ ] E3  Accuracy–Cost Pareto
```

---

# 107. 最终架构

```text
                     SmolVLA inference
                           │
                           ▼
                 Runtime Context Layer
              phase / flow step / attn kind
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
 QuantizedLinear                       QuantizedMatMul
        │                                     │
 unique module_id                       unique module_id
        │                                     │
 input/output codes                     A/B/O codes
        └──────────────────┬──────────────────┘
                           │
                           ▼
                   Sparsity Collector
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
 element stats         bit stats          unit/block stats
      │                    │                    │
      └────────────────────┴────────────────────┘
                           │
                           ▼
                    Workload Exporter
                           │
       shapes + calls + bytes + active bits
                           │
                           ▼
                   Hardware Simulator
                           │
        latency / energy / HBM / SRAM / PE util
```

---

# 108. 最核心的六条原则

### 原则 1

\[
\boxed{\text{sparsity identity}=\text{physical module\_id}}
\]

不是 scale_group。

### 原则 2

\[
\boxed{\text{activation sparsity}=\text{runtime phase-dependent property}}
\]

不是静态 layer property。

### 原则 3

\[
\boxed{\text{weight sparsity}=\text{static property}}
\]

每个 physical module 只统计一次。

### 原则 4

\[
\boxed{\text{denoise sparsity}=f(\text{flow step})}
\]

必须验证 10 个 step，而不是默认相同。

### 原则 5

\[
\boxed{\text{bit sparsity}\neq\text{hardware speedup}}
\]

必须通过 workload + mapping 才能得到 latency。

### 原则 6

\[
\boxed{\text{dynamic OP accuracy gain}-\text{sidepath/control cost}}
\]

才是真正 hardware value。

---

# 109. 本阶段完成后的理想结论形式

最终你应该能够写出类似：

> 在部署态 SmolVLA 中，VLM prefill 与 Action Expert iterative denoise 呈现不同的量化稀疏模式。Expert 的 activation/bit sparsity 随 flow step 演化；QK 与 PV 的 A/B operand 具有不同的稀疏结构，且 Expert attention 的一部分 B operand 来自被重复读取的 prefix KV cache。FP8-PoT 在较粗 scale granularity 下仍保持闭环成功率，同时可能改变 code-zero 与 active-bit 分布。W4 显著降低 weight bit workload，但跨 VLM/Expert scale sharing 会破坏有效表示。Dynamic outlier protection 能恢复精度，但高精度 sidepath 与运行时 top-k 降低硬件规则性；static channel protection 或 blockwise PoT FP8 是更值得部署的替代方案。

这才是完整的“量化稀疏 + 硬件部署”研究链条。

---

# 110. 推荐下一步实际动作

建议下一轮代码开发只做以下四项，不要同时扩太多功能：

```text
1. 把当前本地 sparsity StatManager 正式提交
2. 加 runtime_context：prefill / denoise / flow_step / attention_kind
3. 将 Linear / MatMul collector 全部改成 module_id + structured operand API
4. 输出第一版 workload.csv
```

完成后先跑：

```text
FP8 PoT no-OP / per_site / batch1 / 少量 observation
```

得到第一份可信的 phase-aware sparsity + workload baseline，再决定是否优先做 static OP、block32 PoT FP8 或 W4 group-wise。
