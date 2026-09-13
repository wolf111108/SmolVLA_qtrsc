# Phase H：稀疏统计框架代码修改方案

> 项目：SmolVLA_qtrsc  
> 面向版本：远端 `main`，核查日期 2026-09-13  
> 目标：在**不改变既有量化数值路径和闭环行为**的前提下，得到可用于论文分析与后续硬件建模的可信 sparsity 数据。  
> 适用配置：G1-A FP8-all（Goal SR 88%）与 G1-D Expert-W4（Goal SR 84%）；后续可扩展 G5 的 accuracy-preserving winner。  
> 基本原则：**统计器只能观察，不允许改变量化结果、scale、mask、forward 路径或随机数状态。**

---

## 1. 当前代码状态与本轮修改目标

当前仓库已经具备以下基础：

- `QuantStatManager` 支持 element zero ratio、bit sparsity、amplitude-zero-bit、unit/block sparsity；
- `model_wrapper.install_runtime_hooks()` 已通过 `ContextVar` 自动标记：
  - `sample_actions`：`phase="prefill", flow_step=-1`
  - `denoise_step`：`phase="denoise", flow_step=0..9`
- Linear runtime 已能采集 `activation / output`；
- MatMul runtime 已能采集 `A / B / O`；
- `main.py` 已能导出 sparsity CSV；
- `QuantStatManager.collect_model_weight_sparsity()` 已实现 static weight 一次性统计；
- outlier 路径已经有 `outlier_sidepath` 容器，但当前没有形成完整的可解释统计。

本轮不要重写统计框架，而是做四个 P0 修复与两个 P1 增强：

| 优先级 | 修改 | 原因 |
|---|---|---|
| P0-1 | 真正触发 static weight sparsity collection | 当前 `main.py` 导出 `weight_sparsity.csv`，但没有调用 `collect_model_weight_sparsity()` |
| P0-2 | 修正 outlier mask 制造的“假零值/假 sparse bits” | 当前 normal quant path 会先把 FP side-path 位置置零，直接统计会高估可利用稀疏度 |
| P0-3 | 将 `flow_step` 真正纳入 sparsity record key | 现在 `flow_step_sparsity` 只记录调用次数，真实 ratio 仍按 `(layer, phase, role)` 聚合 |
| P0-4 | unit sparsity 与 `phase/flow_step/tensor_role` 对齐 | 当前 structured record 有 role，但 unit 容器粒度不足，不能用于 flow-step/operator 分析 |
| P1-1 | 输出 quantization/outlier coverage manifest | 防止把“Expert W4 很 sparse”误写成“全模型很 sparse” |
| P1-2 | 将 `ideal_speed_up` 明确改名为 upper bound | 当前 `1/(1-sparsity)` 不能直接称为硬件 speedup |

---

# 2. P0-1：补上 static weight sparsity 的真正采集

## 2.1 当前问题

`src/vla_tcs2/quant/stat_manager.py` 已经有：

```python
collect_model_weight_sparsity(model)
```

其语义正确：

1. 遍历所有 `QuantizedLinear`；
2. 使用该层已经加载/校准好的 `w_interval + w_spec`；
3. `quant_awo()` 得到量化 code；
4. 写入 `per_layer_weight_sparsity`；
5. MatMul 不统计 static weight，因为 MatMul 的 B 是 runtime K/V，不是静态参数。

但 `main.py` 当前流程是：

```text
evaluate()
    ↓
直接 export weight_sparsity.csv
```

没有调用 collector，因此 CSV 可能只有表头。

## 2.2 建议修改位置

文件：

```text
main.py
```

位置：

```text
results = evaluate(...)
之后
sparsity CSV export 之前
```

建议：

```python
if sp_cfg.get("enabled", False) and wrapper.stat_manager is not None:
    sm = wrapper.stat_manager

    n_weight_layers = sm.collect_model_weight_sparsity(model)

    print(
        "[sparsity] static weight collection finished: "
        f"{n_weight_layers} QuantizedLinear layers"
    )

    if n_weight_layers <= 0:
        raise RuntimeError(
            "sparsity.enabled=true but no QuantizedLinear weight "
            "sparsity was collected; check scale loading / wrapping."
        )
```

### 为什么放在 evaluation 之后

本轮正式实验会优先：

```bash
python main.py --config ... --skip-calibration
```

部分 scale/interval 的加载可能依赖实际量化 forward。放在 rollout 之后最稳妥，能保证真正被执行的量化层 interval 已准备好。

如果后续确认所有 scale 都在 `set_mode("quant_forward")` 时同步加载，则可提前到 evaluation 之前；但**不要为了这次实验改 scale loading 时序**。

## 2.3 需要额外注意：static weight ≠ outlier normal-path weight

当前 outlier forward 中，runtime weight mask 会同时受到：

- activation channel mask；
- weight own outlier mask；

影响。因此：

```text
collect_model_weight_sparsity()
```

得到的是：

> “如果整块 weight 都使用该层量化 spec/scale 时的 intrinsic quantized-weight sparsity”

而不是：

> “本次 forward 实际 normal quant path 的 dynamic weight sparsity”。

因此建议把导出文件改名或在 metadata 中明确：

```text
weight_sparsity_static.csv
```

至少新增列：

```text
weight_stat_semantics = full_weight_quantized_no_dynamic_outlier_mask
```

为了向后兼容，可以保留旧的 `weight_sparsity.csv`，但文档必须声明语义。

---

# 3. P0-2：修正 outlier protection 引入的“假稀疏”

## 3.1 当前数据路径

当前 `quant_forward_with_outlier()` 的逻辑本质上是：

```python
x_fp        = x * x_mask
x_normal_fp = x * (~x_mask)

w_fp        = w * w_mask
w_normal_fp = w * (~w_mask)

x_sim = quant_awo(x_normal_fp, ...)
w_sim = quant_awo(w_normal_fp, ...)
```

output 也类似：

```python
out_outlier = out * out_mask
out_normal  = out * (~out_mask)
out_normal_quant = quant_awo(out_normal, ...)
```

因此 FP side-path 所在位置在 normal quant path 中会被**人工写成 0**。

如果直接对 `x_sim / w_sim / out_normal_quant` 统计：

```text
zero_rate
sparse_bit_rate
amplitude_zero_bit_rate
```

这些 masked-out 的位置会全部贡献“零元素/零 bit”，从而系统性高估量化路径可利用稀疏度。

## 3.2 本轮必须同时保留两种定义

建议每个 runtime tensor 形成两个统计视角。

### A. reported/path-coded sparsity

即当前代码直接看到的 code tensor：

\[
Z_{\mathrm{reported}}
=
\frac{N_{\mathrm{zero,code}}}{N_{\mathrm{all}}}
\]

它描述的是“带 mask 的 normal-path code tensor 长什么样”。

### B. native quant-path sparsity

排除 protected FP side-path 后：

\[
N_{\mathrm{native}}
=
N_{\mathrm{all}} - N_{\mathrm{protected}}
\]

因为 protected 位置在 normal code tensor 中被强制写成 0：

\[
N_{\mathrm{zero,native}}
=
N_{\mathrm{zero,reported}} - N_{\mathrm{protected}}
\]

于是：

\[
Z_{\mathrm{native}}
=
\frac{
N_{\mathrm{zero,reported}}-N_{\mathrm{protected}}
}{
N_{\mathrm{all}}-N_{\mathrm{protected}}
}
\]

对于 bit sparsity，设 quant spec bitwidth 为 \(b\)：

\[
B_{\mathrm{native}}
=
B_{\mathrm{all}}-N_{\mathrm{protected}}b
\]

\[
B_{\mathrm{sparse,native}}
=
B_{\mathrm{sparse,reported}}-N_{\mathrm{protected}}b
\]

\[
S_{\mathrm{bit,native}}
=
\frac{B_{\mathrm{sparse,native}}}{B_{\mathrm{native}}}
\]

FP8 的 zero code 全 bit 为 0，因此上述扣除成立；INT zero code 同样成立。

## 3.3 新增 side-path collector API

建议在：

```text
src/vla_tcs2/quant/stat_manager.py
```

新增：

```python
def collect_outlier_partition(
    self,
    *,
    module_id: str,
    tensor_role: str,
    total_elements: int,
    protected_elements: int,
    spec: QuantSpec,
    layer_idx: int | None = None,
    runtime_context: dict | None = None,
    attention_kind: str | None = None,
) -> None:
    ...
```

统一 key：

```python
(
    module_id,
    phase,
    flow_step,
    tensor_role,
    attention_kind,
)
```

累计字段：

```python
{
    "calls": 0,
    "total_elements": 0,
    "protected_elements": 0,
    "normal_elements": 0,
    "protected_bits": 0,
}
```

派生：

```python
fp_sidepath_ratio = protected_elements / total_elements
normal_path_ratio = 1.0 - fp_sidepath_ratio
```

### 不要直接存平均 ratio

错误：

```python
mean(call_ratio)
```

正确：

```python
sum(protected_elements) / sum(total_elements)
```

必须始终先累加 numerator / denominator。

## 3.4 quant_methods.py 中的调用位置

在 `quant_forward_with_outlier()` 中，mask 尚未 delete 前立即记录。

### activation

`x_channel_mask` 通常只有最后一维，需要计算 broadcast 后的真实 element 数：

```python
protected_channels = int(x_channel_mask.sum().item())
repeat = x.numel() // x.shape[-1]
protected_x_elements = protected_channels * repeat
```

更稳健的写法是显式检查最后一维：

```python
assert x_channel_mask.shape[-1] == x.shape[-1]
```

然后：

```python
stat_collector.collect_outlier_partition(
    module_id=module_id,
    tensor_role="activation",
    total_elements=x.numel(),
    protected_elements=protected_x_elements,
    spec=layer.a_spec,
    layer_idx=layer.layer_idx,
)
```

### weight

`w_channel_mask` 已与 weight layout 对齐，可以：

```python
protected_w_elements = int(w_channel_mask.sum().item())
```

但注意：它是 activation-channel mask 与 weight own outlier mask 的 union，因此通常**不会严格等于 1%**。

建议 role 用：

```text
weight_runtime_mask
```

不要和 static `weight_sparsity_static.csv` 混在同一含义里。

### output

`out_with_outlier_mask` 形状与 output 一致：

```python
protected_o_elements = int(out_with_outlier_mask.sum().item())
```

role：

```text
output
```

## 3.5 outlier 路径还应记录四分支计算语义

当前 outlier Linear 实际计算包含：

```text
qA × qW
fpA × fpW
fpA × qW
qA × fpW
```

所以后续硬件模型不能只知道“1% outlier”。

P1 可再输出：

```text
normal_normal_fraction
fpA_fpW_fraction
fpA_qW_fraction
qA_fpW_fraction
```

Phase H 第一轮不要求马上把它映射成 latency，但必须保留 side-path coverage，否则以后无法做真实硬件开销估算。

---

# 4. P0-3：真正按 flow step 统计 sparsity

## 4.1 当前问题

runtime hook 已经正确产生：

```text
prefill: flow_step = -1
denoise: flow_step = 0..9
```

但是当前 structured sparsity key 仍是：

```python
(layer_key, phase, tensor_role)
```

因此 10 个 denoise step 的 sparsity numerator/denominator 最后会混到一起。

当前：

```python
flow_step_sparsity
```

只记录：

```text
flow_step -> layer -> call_count
```

它能证明 0..9 被执行，却不能回答：

```text
step 0 sparsity = ?
step 1 sparsity = ?
...
step 9 sparsity = ?
```

## 4.2 建议统一 structured key

改成：

```python
record_key = (
    module_id,
    phase,
    int(flow_step),
    tensor_role,
    attention_kind or "unknown",
)
```

不要使用 `scale_group` 作为 identity。

推荐 record：

```python
{
    "module_id": module_id,
    "component": "vlm" | "expert" | "matmul",
    "layer_idx": layer_idx,
    "operator": "q_proj" | "down_proj" | "qk" | "pv" | ...,
    "phase": phase,
    "flow_step": flow_step,
    "tensor_role": tensor_role,
    "attention_kind": attention_kind,
    ...
}
```

### Canonical phase 名称

建议正式统一成：

```text
prefill
denoise
full_forward
```

`decode` 只作为 backward-compatible alias。

当前 `_accumulate_phase_sparsity()` 会把 `denoise` 归一化到 legacy `decode`。建议新 structured CSV 一律输出 `denoise`，旧 summary 若需兼容可继续保留 `decode`。

## 4.3 修改 `_collect_one_tensor_sparsity()`

当前：

```python
role_key = (layer_key, phase, tensor_role)
```

改成：

```python
role_key = (
    layer_name,          # physical module_id
    phase,
    int(flow_step),
    tensor_role,
    attention_kind,
)
```

entry 增加：

```python
"flow_step": int(flow_step),
"attention_kind": attention_kind,
```

### 重要

prefill 的所有记录使用：

```text
flow_step = -1
```

不要填 `None`，CSV/聚合脚本处理会简单很多。

---

# 5. P0-4：unit/block sparsity 与 structured key 对齐

## 5.1 当前风险

当前 unit sparsity 容器设计主要是：

```text
phase -> layer -> zero_units / total_units
```

而 `_collect_one_tensor_sparsity()` 调 unit collector 时没有把：

```text
tensor_role
flow_step
attention_kind
```

显式作为结构维度。

因此即使 element/bit sparsity 修好了，现有 unit CSV 仍不适合回答：

- Linear activation 与 output 谁更 sparse？
- MatMul A/B/O 哪个更 sparse？
- denoise step 0 与 step 9 的 block sparsity 是否不同？

## 5.2 推荐新增结构，不破坏 legacy

保留：

```python
self.unit_sparsity
```

用于旧代码。

新增：

```python
self.per_role_unit_sparsity: Dict[tuple, Dict[str, int]]
```

key 与主 structured record 完全一致：

```python
(module_id, phase, flow_step, tensor_role, attention_kind)
```

字段：

```python
{
    "zero_units": int,
    "total_units": int,
    "unit_bit_group_size": int,
    "unit_dim_group_size": int,
}
```

派生：

```python
unit_zero_rate = zero_units / total_units
```

## 5.3 collector 调用

将：

```python
collect_unit_sparsity(layer_name, layer_idx, activation, spec)
```

升级为类似：

```python
collect_unit_sparsity_structured(
    module_id=layer_name,
    layer_idx=layer_idx,
    tensor=activation,
    spec=spec,
    phase=phase,
    flow_step=flow_step,
    tensor_role=tensor_role,
    attention_kind=attention_kind,
)
```

在这个修改完成前：

> `unit_sparsity.csv` 可以用于粗略 aggregate sanity check，但不能作为 Phase H 的 flow-step/operator 论文结论。

---

# 6. P1-1：增加 quantization coverage / deployment manifest

## 6.1 为什么必须有 coverage

G1-D 的性质是：

```text
VLM Linear      = raw FP
Expert Linear   = W4 + FP8 A/O
QK/PV MatMul    = FP8
```

如果只报告：

```text
Expert W4 bit sparsity = 85%
```

不能推导：

```text
whole-model sparsity = 85%
```

更不能推导：

```text
whole-model speedup = 6.7x
```

因此 Phase H 应同时输出“被量化的 workload 到底覆盖多少”。

## 6.2 新增 manifest

建议由 `ModelWrapper` 或独立 helper 扫描一次，输出：

```text
quantization_manifest.csv
```

字段建议：

```text
module_id
component
layer_idx
operator
op_type
quantized
method
a_kind
a_bits_or_fmt
w_kind
w_bits_or_fmt
o_kind
o_bits_or_fmt
weight_quant_granularity
outlier_ratio
in_features
out_features
```

对于 MatMul：

```text
A_spec
B_spec
O_spec
attention_kind
```

## 6.3 coverage 定义分两层

### Parameter coverage（静态）

适合 Linear：

\[
C_{\mathrm{param}}
=
\frac{\sum N_{\mathrm{param,quantized}}}
{\sum N_{\mathrm{param,target}}}
\]

### Runtime MAC coverage（动态，优先）

对于某次 rollout：

Linear/GEMM：

\[
MAC=MKN
\]

\[
C_{\mathrm{MAC}}
=
\frac{\sum MAC_{\mathrm{quantized}}}
{\sum MAC_{\mathrm{target}}}
\]

注意：若 raw operator 没有 runtime hook，当前阶段无法严谨得到 whole-target denominator，就必须在结果中写：

```text
quantized-workload sparsity
```

而不是：

```text
whole-model sparsity
```

不要用推测补 denominator。

---

# 7. P1-2：重命名 `ideal_speed_up`

当前：

\[
ideal\_speed\_up = \frac{1}{1-S_{\mathrm{bit}}}
\]

只描述在“所有 sparse bit 都能零成本跳过、无 memory/outlier/load-balance/reduction 开销”时的数学上界。

建议导出字段改为：

```text
ideal_sparse_upper_bound
```

为了兼容历史脚本可同时保留：

```text
ideal_speed_up_legacy
```

结果文档统一表述：

> ideal sparsity upper bound，不是 measured hardware speedup。

---

# 8. 推荐的新 CSV 结构

## 8.1 `module_sparsity.csv`

建议一行一个：

```text
module_id × phase × flow_step × tensor_role × attention_kind
```

最低字段：

```text
config
module_id
component
layer_idx
operator
phase
flow_step
tensor_role
attention_kind

calls

total_elements_reported
zero_elements_reported
zero_rate_reported

protected_elements
fp_sidepath_ratio

total_elements_native
zero_elements_native
zero_rate_native

total_bits_reported
sparse_bits_reported
sparse_bit_rate_reported

protected_bits
total_bits_native
sparse_bits_native
sparse_bit_rate_native

amplitude_zero_bits_native
amplitude_zero_bit_rate_native

ideal_sparse_upper_bound
```

## 8.2 `weight_sparsity_static.csv`

```text
module_id
component
layer_idx
operator
weight_spec
total_elements
zero_elements
zero_rate
total_bits
sparse_bits
sparse_bit_rate
amplitude_zero_bits
amplitude_zero_bit_rate
stat_semantics
```

其中：

```text
stat_semantics = full_weight_quantized_no_dynamic_outlier_mask
```

## 8.3 `outlier_sidepath.csv`

```text
module_id
phase
flow_step
tensor_role
calls
total_elements
protected_elements
fp_sidepath_ratio
```

## 8.4 `unit_sparsity.csv`

```text
module_id
phase
flow_step
tensor_role
attention_kind
unit_bit_group_size
unit_dim_group_size
zero_units
total_units
unit_zero_rate
```

---

# 9. 代码级 invariant / assert

正式实验前建议增加以下 assert，能避免跑数小时后才发现数据无效。

## 9.1 native counter invariant

```python
assert 0 <= protected_elements <= total_elements
assert native_elements == total_elements - protected_elements
assert reported_zero_elements >= protected_elements
assert native_zero_elements >= 0
```

bit：

```python
assert protected_bits == protected_elements * bitwidth
assert native_bits >= 0
assert reported_sparse_bits >= protected_bits
assert native_sparse_bits >= 0
```

## 9.2 flow-step invariant

当：

```text
num_steps = 10
```

正式 smoke 后必须看到：

```text
denoise flow_step = {0,1,2,3,4,5,6,7,8,9}
```

缺任何 step 都 fail。

## 9.3 role invariant

至少：

```text
Linear -> activation, output
MatMul -> A, B, O
```

如果 MatMul O 缺失，不能做完整 attention sparsity 结论。

## 9.4 static weight invariant

G1-A/G1-D 的具体数量不要永久 hard-code，但 smoke 时必须与实际 wrap summary 一致：

```text
n_weight_sparsity_rows == n_wrapped_quantized_linear
```

若历史 wrap summary仍是：

```text
G1-A: 224 QuantizedLinear
G1-D: 112 QuantizedLinear
```

则可以作为当前版本验收参考；一旦 wrapper 范围变化，以当前 wrap manifest 为准。

---

# 10. 测试计划

建议新增：

```text
tests/test_sparsity_accounting.py
```

最低覆盖 6 个测试。

## T1：无 outlier 的 backward compatibility

构造一个小 tensor，`protected=0`：

```text
reported == native
```

必须完全一致。

## T2：人工 outlier mask 修正

例如 8 个 INT4/FP8 code，其中 2 个位置人为 mask 为 zero。

验收：

```text
reported_zero = real_zero + 2
native_zero   = real_zero
native_total  = 6
```

bit denominator 同理。

## T3：weighted aggregation

两个 call：

```text
call1: 10 elem, 1 zero
call2: 90 elem, 45 zero
```

正确：

```text
46 / 100 = 46%
```

而不是：

```text
(10% + 50%) / 2 = 30%
```

## T4：flow-step separation

相同 module/role：

```text
step0 sparse=10%
step1 sparse=90%
```

导出必须存在两行，不能合成 50%。

## T5：unit role separation

同一个 MatMul：

```text
A / B / O
```

必须各有独立 unit row。

## T6：collector 不影响 forward

同一固定输入、同一 model、相同量化 scale：

```text
sparsity.enabled=false
sparsity.enabled=true
```

比较最终 output：

```python
torch.testing.assert_close(out0, out1, rtol=0, atol=0)
```

若由于实现细节无法 bit-exact，最多允许原 forward dtype 的严格 machine-level tolerance，并记录原因。统计代码原则上不应改变数值，所以优先要求 exact。

---

# 11. 性能与内存注意

bit sparsity 本身可能比 forward 慢很多。本轮允许统计带来 runtime overhead，但必须：

1. 不保存整 tensor；
2. 所有计数使用 chunk；
3. 不把 per-call tensor dump 到磁盘；
4. 不为每个 element 建 Python object；
5. 只保存累计 integer counter。

正式运行前记录：

```text
collector overhead ratio = runtime_with_stats / runtime_without_stats
```

该指标只用于工程评估，不影响 sparsity 数值。

如果 100 ep 代价过高，先证明 10ep/30ep 的统计收敛，再决定是否需要全部 100ep。

---

# 12. 建议代码修改顺序

```text
Step C0
  建分支 phaseH-sparsity-accounting
      ↓
Step C1
  flow_step structured key
      ↓
Step C2
  outlier sidepath counters + native corrected metrics
      ↓
Step C3
  structured unit sparsity
      ↓
Step C4
  main.py static weight collection
      ↓
Step C5
  CSV schema + upper-bound rename
      ↓
Step C6
  tests
      ↓
Step C7
  task0×1 smoke
      ↓
Step C8
  正式 Phase H
```

不要先跑正式 100ep 再回来修 schema。

---

# 13. 建议提交拆分

不要一个 commit 混完所有修改。

建议：

```text
commit 1:
  stats: key sparsity records by flow step and tensor role

commit 2:
  stats: account for outlier FP sidepath in sparsity metrics

commit 3:
  stats: add structured unit sparsity export

commit 4:
  pipeline: collect static quantized weight sparsity before export

commit 5:
  stats: add sparsity accounting tests and invariants

commit 6:
  phaseH: add accuracy-preserving sparsity experiment configs/scripts/docs
```

这样若 SR 或输出异常，可以快速 bisect。

---

# 14. 最终验收 Gate

只有全部满足才允许跑正式 H1/H2/H3。

### Gate C-A：数值透明

- [ ] stats ON/OFF fixed-input forward 等价
- [ ] 原 G1-A/G1-D scale 文件不被覆盖
- [ ] 不重新 calibration
- [ ] 不改 outlier mask 生成算法
- [ ] 不改 quant/dequant formula

### Gate C-B：数据完整

- [ ] static weight CSV 非空
- [ ] Linear activation/output 都存在
- [ ] MatMul A/B/O 都存在
- [ ] prefill 存在
- [ ] denoise 0..9 全存在
- [ ] FP sidepath rows 非空
- [ ] native denominator > 0
- [ ] 无负 counter
- [ ] unit structured rows 非空

### Gate C-C：语义正确

- [ ] reported sparsity 与 native sparsity 均输出
- [ ] protected mask 不再被当成 native zero
- [ ] `ideal_speed_up` 只标记为 upper bound
- [ ] G1-D 结果明确写“quantized workload”，不冒充 whole-model sparsity

---

# 15. 后续硬件模型接口

Phase H 最终应给硬件模拟器提供的不是单一 sparsity 数，而是：

```text
[module_id]
component
operator
layer_idx
phase
flow_step
tensor_role
precision
M/K/N
zero_rate_native
sparse_bit_rate_native
unit_zero_rate
fp_sidepath_ratio
call_count
```

这样后续才能计算：

\[
T_i
=
f(
M_i,K_i,N_i,
precision_i,
S_{bit,i},
S_{unit,i},
R_{FP,i}
)
\]

最后再聚合：

\[
T_{\mathrm{total}}=\sum_i N_{\mathrm{call},i}T_i
\]

而不是把所有层先平均成一个 sparsity 再做 speedup。

---

## 参考的当前仓库位置

- `main.py`
- `src/vla_tcs2/quant/stat_manager.py`
- `src/vla_tcs2/quant/quant_methods.py`
- `src/vla_tcs2/model_wrapper.py`
- `src/vla_tcs2/eval.py`
- `experiments/README.md`
- `experiments/2026-09-10_phaseG_w4-root-cause/tasks/component-localization/`
- `G2.md`

