# SmolVLA Quantization Sparsity Ratio Quick Scan

> 日期：2026-09-15  
> 目标：使用当前 `SmolVLA_qtrsc` 稀疏统计框架，在尽量 1 小时内完成 INT8、INT16、FP8W4 PoT、FP8 PoT 四组量化配置的**元素级 sparsity**与**bit 级 sparsity**统计。  
> 快速协议：`LIBERO Goal task0 × 1 episode/config`，`n_action_steps=10`，`num_steps=10`。本实验只做 workload characterization，不做正式 SR claim。

## 0. H3 正在运行时的版本纪律

本 quick scan **不要修改** `main.py`、`stat_manager.py`、`quant_methods.py`、`model_wrapper.py`。当前框架已经足够。只新增新的 experiment configs/scripts/docs。

原因：H3 runner 可能逐 task 启动新 Python 进程；若 H3 中途修改 core code，后续 task 可能载入不同代码版本，破坏 H3 一致性。

## 1. 统计范围

一次 SmolVLA action generation 按时间拆成：

```text
sample_actions
├── VLM prefill ×1
└── Action Expert denoise ×10
```

本实验固定只汇总两组：

```text
PREFILL_VLM:
  phase == "prefill"
  component == "vlm"

DENOISE_EXPERT:
  phase == "denoise"
  component == "expert"
  flow_step == 0..9
```

不要把 prefill/denoise 或 VLM/Expert 混成一个 headline ratio。

## 2. 指标定义

### 2.1 Runtime element sparsity

使用 `module_sparsity.csv` 的 native counters：

\[
S_{elem}=\frac{\sum zero\_elements\_native}{\sum total\_elements\_native}
\]

PoT/outlier 配置必须用 `native`，不能用 `reported`，因为 protected FP sidepath 会在 normal quant path 中制造人工 0。

### 2.2 Runtime bit sparsity

\[
S_{bit}=\frac{\sum sparse\_bits\_native}{\sum total\_bits\_native}
\]

### 2.3 Static Linear weight sparsity

使用 `weight_sparsity_static.csv`：

\[
S_{elem,W}=\frac{\sum zero\_elements}{\sum total\_elements}
\]

\[
S_{bit,W}=\frac{\sum sparse\_bits}{\sum total\_bits}
\]

Static weight 按 component 归属：`vlm -> VLM prefill`，`expert -> Expert denoise`。这只是 deployment/workload 归属，不代表 weight 随时间变化。

## 3. Bit metric 语义

当前框架的 bit sparsity 不是四种格式完全统一的存储编码指标：

```text
INT8 / INT16 / INT4:
  two's-complement sign-aware sparse-bit metric
  positive: 0 bit 可跳过
  negative: sign-extension 1 bit 可跳过

FP8 E4M3:
  4-bit significand (1MMM / 0MMM) zero-bit ratio
  不包含 sign/exponent
```

因此：

- 元素级 zero ratio 可以直接横向比较；
- bit sparsity 应解释成“各自格式的 compute-bit skipping opportunity”；
- 不要把 FP8 40% 与 INT8/INT4 70% 直接写成同一指标的 +30pp。

## 4. 四个配置

统一：

```text
model              = lerobot/smolvla_libero
n_action_steps     = 10
num_steps          = 10
suite              = libero_goal
task_id            = 0
n_episodes         = 1
batch_size         = 1
seed               = 1000
quantize_matmul    = true
Linear include     = "*"
outlier_ratio      = 0.01
```

### Q0 INT8

```text
Linear A/W/O = INT8
QK/PV A/B/O  = INT8
method       = outlier
```

Fresh quick calibration required.

### Q1 INT16

```text
Linear A/W/O = INT16
QK/PV A/B/O  = INT16
method       = outlier
```

Fresh quick calibration required.

### Q2 FP8W4 PoT

复用 Phase-G G1-B：

```text
Linear A/O   = E4M3
Linear W     = INT4
Linear method= pot_ao_outlier
QK/PV A/B/O  = E4M3
MatMul method= pot_fp8_outlier
```

复用 scale：

```text
scales/2026-09-10_phaseG_w4-root-cause/component-localization/g1b_w4_all
```

严格语义：`pot_ao_outlier` 只保证 A/O scale 为 PoT；W4 weight scale 保持 continuous calibrated scale。因此论文/文档中更准确的名字是 **FP8(A/O)-W4 + PoT(A/O)**。

### Q3 FP8 PoT

复用 Phase-G G1-A：

```text
Linear A/W/O = E4M3
QK/PV A/B/O  = E4M3
method       = pot_fp8_outlier
all scales   = PoT
```

复用 scale：

```text
scales/2026-09-10_phaseG_w4-root-cause/component-localization/g1a_fp8_all
```

## 5. 一小时 quick calibration

仅 Q0/Q1 新校准：

```yaml
calibration:
  dataset_repo_id: HuggingFaceVLA/libero
  dataset_revision: v3.0
  episodes: 1
  batch_size: 8
  frame_stride: 16
  seed: 42
```

这是 sparsity quick scan 的 scale，不用于正式 accuracy PTQ 结论。

## 6. 减少统计开销

本实验只需要 element + bit，因此：

```yaml
sparsity:
  enabled: true
  chunk_size: 4194304
  unit:
    enabled: false
  export:
    dir: null
```

关闭/不启用：

```text
fp_code_audit
unit/block sparsity
tensor dump
video rendering
```

若显存压力大，把 `chunk_size` 降回 `1048576`。

## 7. 目录

```text
experiments/
└── 2026-09-15_sparsity-ratio-quickscan/
    ├── configs/
    │   ├── q0_int8.yaml
    │   ├── q1_int16.yaml
    │   ├── q2_fp8w4_po2.yaml
    │   └── q3_fp8_po2.yaml
    ├── scripts/
    │   ├── run_quickscan.sh
    │   └── summarize_quick_sparsity.py
    └── docs/
        └── results.md
```

输出：

```text
outputs/2026-09-15_sparsity-ratio-quickscan/
├── q0_int8/
├── q1_int16/
├── q2_fp8w4_po2/
├── q3_fp8_po2/
├── quick_sparsity_summary.csv
└── quick_sparsity_by_role.csv
```

## 8. 运行前检查

```bash
conda activate smolvla_eval
cd ~/VLA_tcs2

git rev-parse HEAD
git status --short

test -d scales/2026-09-10_phaseG_w4-root-cause/component-localization/g1a_fp8_all
test -d scales/2026-09-10_phaseG_w4-root-cause/component-localization/g1b_w4_all
```

## 9. 运行

```bash
GPU=1 bash experiments/2026-09-15_sparsity-ratio-quickscan/scripts/run_quickscan.sh
```

`GPU` 请选择**未被 H3 占用**的 GPU。

多 GPU 时可并行四个 config；单 GPU 顺序执行即可。

## 10. 时间预算

```text
Q0 INT8    calibration + 1ep   ~10–15 min
Q1 INT16   calibration + 1ep   ~10–15 min
Q2 FP8W4   reuse scale + 1ep   ~5–10 min
Q3 FP8     reuse scale + 1ep   ~5–10 min
summary                         <1 min
----------------------------------------
sequential                     ~30–50 min
```

实际时间取决于 episode 长度与 GPU contention。

## 11. 输出 Gate

每个 config 至少需要：

```text
sparsity/module_sparsity.csv
sparsity/weight_sparsity_static.csv
sparsity/quantization_manifest.csv
sparsity/outlier_sidepath.csv
```

并验证：

```text
phase: prefill + denoise
denoise flow_step: 0..9
component: vlm + expert
Linear roles: activation/output
MatMul roles: A/B/O
native counters 无负值
```

## 12. 最终 Primary 表

| Config | Stage | Runtime element sparsity | Runtime bit sparsity | Weight element sparsity | Weight bit sparsity |
|---|---|---:|---:|---:|---:|
| INT8 | VLM prefill | | | | |
| INT8 | Expert denoise | | | | |
| INT16 | VLM prefill | | | | |
| INT16 | Expert denoise | | | | |
| FP8W4 PoT | VLM prefill | | | | |
| FP8W4 PoT | Expert denoise | | | | |
| FP8 PoT | VLM prefill | | | | |
| FP8 PoT | Expert denoise | | | | |

Secondary 表按 role 输出：

```text
Linear activation
Linear output
QK A/B/O
PV A/B/O
Weight q/k/v/o/gate/up/down
```

其中：

```text
PV A = softmax attention probability P
PV B = V
```

## 13. 聚合规范

禁止平均百分比：

```python
mean(row["zero_rate_native"])          # 错
mean(row["sparse_bit_rate_native"])    # 错
```

必须 numerator/denominator 先求和：

\[
S_{elem}=\frac{\sum zero}{\sum total}
\]

\[
S_{bit}=\frac{\sum sparse\ bits}{\sum total\ bits}
\]

随附 `summarize_quick_sparsity.py` 已按该规则实现。

## 14. 结果解读

INT8 vs INT16：重点看 bit width 增大后 quantization-induced element zeros 是否减少，以及 sign-aware sparse-bit ratio 如何变化。

FP8 PoT vs FP8W4 PoT：runtime A/O/QK/PV 主要仍为 FP8，因此 runtime sparsity 预计接近；差异预计主要在 static Linear weight（FP8 W vs INT4 W）。

Prefill vs denoise：必须分开，因为 VLM prefill 每次 generation 1 次，而 Expert denoise 每次 generation 有 10 个 flow step；相同 sparsity ratio 不意味着相同总硬件节省。

## 15. 本 quick scan 不做

```text
正式 Success Rate 对比
100ep rollout
unit/block sparsity
FP raw-code audit
BOP/hardware speedup
whole-model FLOP-weighted speedup
```

`1/(1-S)` 不能作为实际硬件 speedup。
