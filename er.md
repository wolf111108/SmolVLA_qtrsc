建议把这件事单独做成一个很小的 **H1-Audit**，不要马上改量化算法，也不要重跑整个 H1。目标只有一个：

$$
\boxed{\text{判断 output/O 的 }0.3\%-0.6\%\text{ bit sparsity 到底是真的，还是统计错误}}
$$

最合适的办法是：**直接对真正送进现有 sparsity collector 的 post-quant FP8 code 做独立 bit-pattern histogram**。当前 Linear 的 `output_code` 和 MatMul 的 `O_code` 本来就统一经过 `collect_quant_tensor()`，所以把 audit 放在那里，不需要改 forward 数值路径。

另外有一点很重要：我当前重新读取到的远端 `stat_manager.py` 里，`_extract_sm_from_raw()` 仍然是旧的 `sign<<mant_bits | mant`，并用 `(exp != 0) | (mant != 0)` 判断 hidden bit 的版本。 所以这次 audit **不要调用现有 `_extract_sm_from_raw()`**，而是独立解析 E4M3 raw code。这样 audit 才能真正作为第二套独立验证器。

---

# 1. 先定义这个 Audit 要回答什么

对于每个：

```text
module_id
phase
flow_step
tensor_role
attention_kind
```

统计 E4M3 code 的：

```text
zero code ratio
mantissa 000~111 histogram
exponent 0000~1111 histogram
significand 0000~1111 histogram
non-zero significand bit sparsity
mantissa=111 ratio
saturation ratio
NaN ratio
```

最关键的是这两个：

$$
S_{\text{sig,nonzero}}
$$

和

$$
P(\text{mantissa}=111\mid q\neq0)
$$

如果你真的得到：

```text
output:
nonzero_significand_zero_rate ≈ 0.5%
mantissa_111_nonzero_ratio ≈ 98~99%
```

那说明这个现象是真实的。

如果 histogram 很正常，例如：

```text
mantissa 000~111 都有大量分布
```

但现有 `module_sparsity.csv` 仍然说 0.5%，那就是 sparsity collector 有 bug。

---

# 2. 不改 `quant_methods.py`，先只改 `stat_manager.py`

这是第一轮 audit 最重要的原则。

当前 `collect_quant_tensor()` 已经拿到了真正的：

```python
tensor_code
spec
module_id
phase
flow_step
tensor_role
attention_kind
```

。

而 `quant_awo()` 的 FP8 路径本身就是：

```python
q = x / scale
q = clamp(q)
q = q.to(torch.float8_e4m3fn)
```

scalar path 最后只是把这个已量化值存进指定的 output dtype，因此把 `tensor_code` 再 cast 回 `float8_e4m3fn` 可以恢复实际 FP8 code pattern。

所以第一轮完全没有必要碰 forward。

---

# 3. 在 `QuantStatManager.__init__()` 增加 Audit 状态

在：

```python
src/vla_tcs2/quant/stat_manager.py
```

`__init__()` 里加入：

```python
# ------------------------------------------------------------------
# Phase H FP-code audit
# Debug-only: inspect the actual post-quant FP8 code distribution.
# Must never modify model tensors or RNG state.
# ------------------------------------------------------------------
self.fp_code_audit_enabled = False

# Deterministic maximum number of elements sampled from each
# collect_quant_tensor() call. 0/None means all elements.
self.fp_code_audit_max_elements_per_call = 262_144

# Roles to inspect. Start with both operands and outputs so we have
# a healthy input-side reference.
self.fp_code_audit_roles = {
    "activation",
    "output",
    "A",
    "B",
    "O",
}

# key:
# (module_id, phase, flow_step, tensor_role, attention_kind)
self.fp_code_audit: Dict[tuple, Dict[str, Any]] = {}
```

在：

```python
reset_sparsity()
```

也加入：

```python
self.fp_code_audit = {}
```

---

# 4. 增加配置函数

在 `QuantStatManager` 中新增：

```python
def configure_fp_code_audit(
    self,
    enable: bool = False,
    max_elements_per_call: int = 262_144,
    roles=None,
):
    self.fp_code_audit_enabled = bool(enable)

    if max_elements_per_call is None:
        self.fp_code_audit_max_elements_per_call = 0
    else:
        self.fp_code_audit_max_elements_per_call = int(
            max_elements_per_call
        )

    if roles is not None:
        self.fp_code_audit_roles = set(roles)
```

这个 audit 默认关闭，不影响 H1/H2 正常实验。

---

# 5. 增加独立的 E4M3 code collector

这是核心。

直接放进 `QuantStatManager`：

```python
def _collect_fp_code_audit(
    self,
    *,
    module_id: str,
    tensor_role: str,
    tensor_code: torch.Tensor,
    spec: QuantSpec,
    phase: str,
    flow_step: int,
    attention_kind: str,
):
    """
    Debug-only independent E4M3 raw-code audit.

    IMPORTANT:
    - Does NOT reuse _extract_sm_from_raw().
    - Does NOT modify tensor_code.
    - Does NOT use RNG.
    - Deterministically subsamples large tensors.
    """

    if not self.fp_code_audit_enabled:
        return

    if tensor_role not in self.fp_code_audit_roles:
        return

    if tensor_code is None or spec is None:
        return

    if spec.kind != "fp":
        return

    fmt = (spec.fmt or "").lower().strip()

    if fmt not in {
        "e4m3",
        "e4m3fn",
        "fp8_e4m3",
        "fp8_e4m3fn",
    }:
        return

    x = tensor_code.detach().reshape(-1)

    if x.numel() == 0:
        return

    # --------------------------------------------------------------
    # Deterministic sampling.
    # No torch.rand / RNG, so rollout behavior is unaffected.
    # --------------------------------------------------------------
    max_n = self.fp_code_audit_max_elements_per_call

    if max_n and max_n > 0 and x.numel() > max_n:
        stride = (
            x.numel() + max_n - 1
        ) // max_n

        x = x[::stride][:max_n]

    # tensor_code contains FP8-representable numerical values,
    # generally stored as float32 for the simulator.
    q8 = x.to(torch.float8_e4m3fn)

    raw = (
        q8.view(torch.uint8)
        .reshape(-1)
        .to(torch.int64)
    )

    q32 = q8.to(torch.float32)

    # --------------------------------------------------------------
    # Independent E4M3 parsing.
    #
    # raw:
    #   bit7     = sign
    #   bit6..3  = exponent
    #   bit2..0  = mantissa
    # --------------------------------------------------------------
    sign = (raw >> 7) & 0x1
    exp = (raw >> 3) & 0xF
    mant = raw & 0x7

    nan_mask = torch.isnan(q32)

    # ±0 both count as zero.
    zero_mask = (exp == 0) & (mant == 0)

    # E4M3 subnormal: exponent zero, mantissa non-zero.
    subnormal_mask = (exp == 0) & (mant != 0)

    valid_mask = ~nan_mask

    # --------------------------------------------------------------
    # Magnitude significand for the metric used by Phase H:
    #
    # normal:    1MMM
    # subnormal: 0MMM
    # zero:      0000
    #
    # Sign is intentionally NOT part of these 4 bits.
    # --------------------------------------------------------------
    hidden = (exp != 0).to(torch.int64) << 3
    sig = mant | hidden

    # Force zero to exactly 0000.
    sig = torch.where(
        zero_mask,
        torch.zeros_like(sig),
        sig,
    )

    # Ignore NaN when computing numerical sparsity audit.
    sig_valid = sig[valid_mask]

    nonzero_valid_mask = valid_mask & (~zero_mask)
    sig_nonzero = sig[nonzero_valid_mask]

    mant_valid = mant[valid_mask]
    mant_nonzero = mant[nonzero_valid_mask]
    exp_valid = exp[valid_mask]

    # --------------------------------------------------------------
    # Histograms
    # --------------------------------------------------------------
    mant_hist = torch.bincount(
        mant_valid,
        minlength=8,
    )

    mant_nonzero_hist = torch.bincount(
        mant_nonzero,
        minlength=8,
    )

    exp_hist = torch.bincount(
        exp_valid,
        minlength=16,
    )

    sig_hist = torch.bincount(
        sig_valid,
        minlength=16,
    )

    # --------------------------------------------------------------
    # Independent significand zero-bit count.
    # --------------------------------------------------------------
    sig_zero_bits = 0

    for bit_idx in range(4):
        bit = (sig_valid >> bit_idx) & 1
        sig_zero_bits += int(
            (1 - bit).sum().item()
        )

    sig_nonzero_zero_bits = 0

    for bit_idx in range(4):
        bit = (sig_nonzero >> bit_idx) & 1
        sig_nonzero_zero_bits += int(
            (1 - bit).sum().item()
        )

    max_val = float(
        torch.finfo(torch.float8_e4m3fn).max
    )

    saturation_count = int(
        (
            torch.abs(q32[valid_mask])
            >= max_val
        ).sum().item()
    )

    key = (
        module_id,
        phase,
        int(flow_step),
        tensor_role,
        attention_kind or "unknown",
    )

    entry = self.fp_code_audit.setdefault(
        key,
        {
            "module_id": module_id,
            "phase": phase,
            "flow_step": int(flow_step),
            "tensor_role": tensor_role,
            "attention_kind": attention_kind or "unknown",

            "calls": 0,
            "sampled_elements": 0,
            "valid_elements": 0,
            "nonzero_elements": 0,

            "zero_code_count": 0,
            "subnormal_count": 0,
            "nan_count": 0,
            "negative_count": 0,
            "saturation_count": 0,

            "sig_zero_bits": 0,
            "sig_total_bits": 0,

            "sig_nonzero_zero_bits": 0,
            "sig_nonzero_total_bits": 0,

            "mant_hist": [0] * 8,
            "mant_nonzero_hist": [0] * 8,
            "exp_hist": [0] * 16,
            "sig_hist": [0] * 16,
        },
    )

    n_sampled = int(raw.numel())
    n_valid = int(valid_mask.sum().item())
    n_nonzero = int(nonzero_valid_mask.sum().item())

    entry["calls"] += 1
    entry["sampled_elements"] += n_sampled
    entry["valid_elements"] += n_valid
    entry["nonzero_elements"] += n_nonzero

    entry["zero_code_count"] += int(
        zero_mask.sum().item()
    )

    entry["subnormal_count"] += int(
        subnormal_mask.sum().item()
    )

    entry["nan_count"] += int(
        nan_mask.sum().item()
    )

    entry["negative_count"] += int(
        (sign[valid_mask] != 0).sum().item()
    )

    entry["saturation_count"] += saturation_count

    entry["sig_zero_bits"] += sig_zero_bits
    entry["sig_total_bits"] += 4 * n_valid

    entry[
        "sig_nonzero_zero_bits"
    ] += sig_nonzero_zero_bits

    entry[
        "sig_nonzero_total_bits"
    ] += 4 * n_nonzero

    for i in range(8):
        entry["mant_hist"][i] += int(
            mant_hist[i].item()
        )

        entry["mant_nonzero_hist"][i] += int(
            mant_nonzero_hist[i].item()
        )

    for i in range(16):
        entry["exp_hist"][i] += int(
            exp_hist[i].item()
        )

        entry["sig_hist"][i] += int(
            sig_hist[i].item()
        )
```

这里最重要的是：

```python
hidden = (exp != 0) << 3
sig = mant | hidden
```

而不是调用现有：

```python
_extract_sm_from_raw()
```

这样两个统计器真正独立。

---

# 6. 在 `collect_quant_tensor()` 中只加一个调用

当前 `collect_quant_tensor()` 大约在读取：

```python
phase
flow_step
attention_kind
```

之后调用：

```python
_collect_one_tensor_sparsity(...)
```

。

改成：

```python
phase = ctx.get("phase", "unknown")

if (
    phase == "unknown"
    and self.current_phase != "full_forward"
):
    phase = self.current_phase

flow_step = int(
    ctx.get("flow_step", -1)
)

resolved_attention_kind = (
    attention_kind
    or ctx.get("attention_kind", "unknown")
)

# Existing Phase-H collector.
self._collect_one_tensor_sparsity(
    module_id,
    layer_idx,
    tensor_code,
    spec,
    tensor_role=tensor_role,
    phase=phase,
    flow_step=flow_step,
    attention_kind=resolved_attention_kind,
)

# H1-Audit: independent raw FP8-code inspection.
self._collect_fp_code_audit(
    module_id=module_id,
    tensor_role=tensor_role,
    tensor_code=tensor_code,
    spec=spec,
    phase=phase,
    flow_step=flow_step,
    attention_kind=resolved_attention_kind,
)
```

到这里为止：

$$
\boxed{\text{quant\_methods.py 完全不用改}}
$$

因此它不可能改变：

```text
scale
mask
quantization
outlier path
model output
SR
```

---

# 7. 增加 CSV exporter

在 `stat_manager.py` 加：

```python
def export_fp_code_audit_csv(
    self,
    path: str,
):
    import csv
    import os

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    fieldnames = [
        "module_id",
        "phase",
        "flow_step",
        "tensor_role",
        "attention_kind",

        "calls",
        "sampled_elements",
        "valid_elements",
        "nonzero_elements",

        "zero_code_count",
        "zero_code_rate",

        "subnormal_count",
        "subnormal_rate",

        "nan_count",
        "nan_rate",

        "negative_count",
        "negative_rate",

        "saturation_count",
        "saturation_rate",

        "sig_zero_bits",
        "sig_total_bits",
        "sig_zero_rate",

        "sig_nonzero_zero_bits",
        "sig_nonzero_total_bits",
        "sig_nonzero_zero_rate",

        "mant_111_nonzero_rate",
    ]

    fieldnames += [
        f"mant_{i:03b}"
        for i in range(8)
    ]

    fieldnames += [
        f"mant_nonzero_{i:03b}"
        for i in range(8)
    ]

    fieldnames += [
        f"exp_{i:04b}"
        for i in range(16)
    ]

    fieldnames += [
        f"sig_{i:04b}"
        for i in range(16)
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for key, e in sorted(
            self.fp_code_audit.items()
        ):
            sampled = e["sampled_elements"]
            valid = e["valid_elements"]
            nonzero = e["nonzero_elements"]

            sig_total = e["sig_total_bits"]
            sig_nz_total = e[
                "sig_nonzero_total_bits"
            ]

            mant_nz_total = sum(
                e["mant_nonzero_hist"]
            )

            row = {
                "module_id": e["module_id"],
                "phase": e["phase"],
                "flow_step": e["flow_step"],
                "tensor_role": e["tensor_role"],
                "attention_kind": e[
                    "attention_kind"
                ],

                "calls": e["calls"],
                "sampled_elements": sampled,
                "valid_elements": valid,
                "nonzero_elements": nonzero,

                "zero_code_count":
                    e["zero_code_count"],

                "zero_code_rate":
                    e["zero_code_count"] / valid
                    if valid else 0.0,

                "subnormal_count":
                    e["subnormal_count"],

                "subnormal_rate":
                    e["subnormal_count"] / valid
                    if valid else 0.0,

                "nan_count":
                    e["nan_count"],

                "nan_rate":
                    e["nan_count"] / sampled
                    if sampled else 0.0,

                "negative_count":
                    e["negative_count"],

                "negative_rate":
                    e["negative_count"] / valid
                    if valid else 0.0,

                "saturation_count":
                    e["saturation_count"],

                "saturation_rate":
                    e["saturation_count"] / valid
                    if valid else 0.0,

                "sig_zero_bits":
                    e["sig_zero_bits"],

                "sig_total_bits":
                    sig_total,

                "sig_zero_rate":
                    e["sig_zero_bits"] / sig_total
                    if sig_total else 0.0,

                "sig_nonzero_zero_bits":
                    e["sig_nonzero_zero_bits"],

                "sig_nonzero_total_bits":
                    sig_nz_total,

                "sig_nonzero_zero_rate":
                    e["sig_nonzero_zero_bits"]
                    / sig_nz_total
                    if sig_nz_total else 0.0,

                "mant_111_nonzero_rate":
                    (
                        e["mant_nonzero_hist"][7]
                        / mant_nz_total
                    )
                    if mant_nz_total else 0.0,
            }

            for i in range(8):
                row[f"mant_{i:03b}"] = (
                    e["mant_hist"][i]
                )

                row[
                    f"mant_nonzero_{i:03b}"
                ] = e["mant_nonzero_hist"][i]

            for i in range(16):
                row[f"exp_{i:04b}"] = (
                    e["exp_hist"][i]
                )

                row[f"sig_{i:04b}"] = (
                    e["sig_hist"][i]
                )

            writer.writerow(row)
```

---

# 8. `main.py` 增加两个很小的接口

在当前 sparsity setup 中，你现在已经有：

```python
sm.enable_sparsity(...)
sm.configure_unit_sparsity(...)
```

。

后面加：

```python
audit_cfg = sp_cfg.get(
    "fp_code_audit",
    {},
)

if audit_cfg.get("enabled", False):
    sm.configure_fp_code_audit(
        enable=True,
        max_elements_per_call=audit_cfg.get(
            "max_elements_per_call",
            262_144,
        ),
        roles=audit_cfg.get(
            "roles",
            [
                "activation",
                "output",
                "A",
                "B",
                "O",
            ],
        ),
    )

    print(
        "[sparsity] FP-code audit enabled"
    )
```

在现有 CSV export 后面加：

```python
if sm.fp_code_audit_enabled:
    sm.export_fp_code_audit_csv(
        os.path.join(
            sparsity_dir,
            "fp_code_audit.csv",
        )
    )
```

---

# 9. 建一个专门的 H1-Audit config

不要污染原 H1。

新建：

```text
experiments/
2026-09-13_phaseH_accuracy-preserving-sparsity/
configs/
s0_fp8_all_h1_audit_task0.yaml
```

最保险的方法：

> **直接复制 H1 S0 task00 已实际运行的 YAML。**

只改：

```yaml
output_dir: outputs/2026-09-13_phaseH_accuracy-preserving-sparsity/h1_audit/s0/task00
```

以及：

```yaml
evaluation:
  env:
    task: libero_goal
    task_ids: [0]

  n_episodes: 1
  batch_size: 1
```

在已有：

```yaml
sparsity:
```

下面增加：

```yaml
  fp_code_audit:
    enabled: true

    # 第一轮已经足够大。
    # 使用 deterministic strided sampling，不使用随机数。
    max_elements_per_call: 262144

    roles:
      - activation
      - output
      - A
      - B
      - O
```

其他所有字段：

```text
model
scale_dir
method
outlier_ratio
A/W/O formats
n_action_steps
num_steps
seed
```

一个都不要改。

尤其仍然：

```text
n_action_steps = 10
num_steps = 10
seed = 1000
```

以及：

```bash
--skip-calibration
```

---

# 10. 运行

先跑 tests：

```bash
cd ~/VLA_tcs2

python -m pytest \
    tests/test_sparsity_accounting.py \
    -v
```

然后：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=2

python main.py \
    --config experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/configs/s0_fp8_all_h1_audit_task0.yaml \
    --skip-calibration
```

最后应该多得到：

```text
outputs/.../h1_audit/s0/task00/sparsity/
    module_sparsity.csv
    weight_sparsity_static.csv
    outlier_sidepath.csv
    unit_sparsity.csv
    fp_code_audit.csv        <-- 新增
```

---

# 11. 先不用写复杂分析程序，直接跑这个汇总

新建：

```text
scripts/analyze_fp_code_audit.py
```

内容：

```python
#!/usr/bin/env python

import csv
import sys
from collections import defaultdict


path = sys.argv[1]

rows = []

with open(path, newline="") as f:
    rows = list(csv.DictReader(f))


def add(a, k, v):
    a[k] += int(float(v))


groups = defaultdict(
    lambda: defaultdict(int)
)

for r in rows:
    key = (
        r["tensor_role"],
        r["phase"],
    )

    g = groups[key]

    for name in [
        "valid_elements",
        "nonzero_elements",
        "zero_code_count",
        "subnormal_count",
        "nan_count",
        "saturation_count",
        "sig_zero_bits",
        "sig_total_bits",
        "sig_nonzero_zero_bits",
        "sig_nonzero_total_bits",
    ]:
        add(g, name, r[name])

    for i in range(8):
        add(
            g,
            f"mant_nonzero_{i:03b}",
            r[f"mant_nonzero_{i:03b}"],
        )


print(
    f"{'role':12s} "
    f"{'phase':10s} "
    f"{'zero':>10s} "
    f"{'sig-all':>10s} "
    f"{'sig-nz':>10s} "
    f"{'mant111':>10s} "
    f"{'sat':>10s}"
)

for (role, phase), g in sorted(
    groups.items()
):
    valid = g["valid_elements"]
    nonzero = g["nonzero_elements"]

    sig_total = g["sig_total_bits"]
    sig_nz_total = g[
        "sig_nonzero_total_bits"
    ]

    mant_nz_total = sum(
        g[f"mant_nonzero_{i:03b}"]
        for i in range(8)
    )

    zero_rate = (
        g["zero_code_count"] / valid
        if valid else 0
    )

    sig_rate = (
        g["sig_zero_bits"] / sig_total
        if sig_total else 0
    )

    sig_nz_rate = (
        g["sig_nonzero_zero_bits"]
        / sig_nz_total
        if sig_nz_total else 0
    )

    mant111 = (
        g["mant_nonzero_111"]
        / mant_nz_total
        if mant_nz_total else 0
    )

    sat = (
        g["saturation_count"] / valid
        if valid else 0
    )

    print(
        f"{role:12s} "
        f"{phase:10s} "
        f"{zero_rate:10.3%} "
        f"{sig_rate:10.3%} "
        f"{sig_nz_rate:10.3%} "
        f"{mant111:10.3%} "
        f"{sat:10.3%}"
    )
```

执行：

```bash
python \
experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/analyze_fp_code_audit.py \
outputs/2026-09-13_phaseH_accuracy-preserving-sparsity/h1_audit/s0/task00/sparsity/fp_code_audit.csv
```

---

# 12. 最关键的是怎么读结果

你最后大概会得到这种形式：

```text
role        phase      zero    sig-all     sig-nz    mant111        sat
activation prefill    1.xx%     41.xx%     40.xx%      xx.xx%      0.xx%
activation denoise    1.xx%     41.xx%     40.xx%      xx.xx%      0.xx%
A          denoise    1.xx%     55.xx%     54.xx%      xx.xx%      0.xx%
B          denoise    1.xx%     40.xx%     39.xx%      xx.xx%      0.xx%
output     denoise    1.xx%      1.xx%      0.xx%      99.xx%      ?.??%
O          denoise    1.xx%      1.xx%      0.xx%      99.xx%      ?.??%
```

然后按下面判断。

### 情况 1：确认是真实 code distribution

如果：

```text
output sig_nonzero_zero_rate ≈ 0~1%
O      sig_nonzero_zero_rate ≈ 0~1%

同时：

mant111_nonzero ≈ 95~100%
```

那么：

$$
\boxed{
0.5\%\text{ 是真实 post-quant E4M3 code 特征}
}
$$

这时不要修 sparsity collector。

下一步才进入 **Scale/Prequant Audit**。

---

### 情况 2：独立 audit 是正常的 30–40%，原 CSV 却是 0.5%

例如：

```text
fp_code_audit:
output sig-nz = 37%

module_sparsity:
output native = 0.5%
```

那么：

$$
\boxed{
\text{stat\_manager 主 bit accounting 仍有 bug}
}
$$

这时重点检查：

```python
compute_sparse_stats_fp()
_extract_sm_from_raw()
native subtraction
```

尤其注意：我当前读取到的远端 `_extract_sm_from_raw()` 仍然是旧实现。

---

### 情况 3：`mant111≈99%`，但 saturation 也非常高

例如：

```text
mant111 = 98%
saturation = 40%
```

说明 output quantization 很可能处在异常 scale/clipping 区域。

这时进入第二阶段：

$$
\boxed{\text{prequant/scale audit}}
$$

---

### 情况 4：`mant111≈99%`，但 saturation≈0%

这是最有意思的情况：

```text
mantissa 111 主导
不是 clipping
不是 native subtraction
不是 statistic bug
```

说明 `out_normal_quant` 的数值本身大量落在每个 exponent bin 的高端，例如：

$$
1.875\times2^e
$$

附近。

这时候才值得深入分析 output datapath。

---

# 13. 如果确认是真实分布，再做第二阶段：Prequant/Scale Audit

**不要现在就做。**

只有第一阶段确认：

```text
mantissa_111_nonzero >> 90%
```

才去改 `quant_methods.py`。

因为当前 Linear output 进入量化前明确是：

```python
out_normal
```

然后：

```python
out_normal_quant = quant_awo(
    out_normal,
    layer.o_interval,
    ...
)
```

。

MatMul 同理是：

```python
out_normal
→ quant_awo(... layer.O_interval ...)
→ out_normal_quant
```

。

第二阶段只需统计：

$$
z=\frac{out_{\rm normal}}{s_O}
$$

以及：

```text
|z| / 448
clip ratio
log2 magnitude histogram
```

从而判断到底是：

```text
scale 问题
```

还是：

```text
GEMM output 数值本身的 significand 分布
```

。

---

## 我建议现在严格按这个顺序

1. **先在已有 H1 CSV 上保留当前结果，不重跑 H1。**
2. 加 `fp_code_audit`，只修改 `stat_manager.py + main.py`。
3. 不改 `quant_methods.py`。
4. 跑 `S0 Goal task0 × 1 episode`。
5. 看：

   * `sig_nonzero_zero_rate`
   * `mant_111_nonzero_rate`
   * `saturation_rate`
6. 再决定是否需要第二阶段 scale audit。

这样只需要一个很短的 rollout，就能明确回答现在最关键的问题：

$$
\boxed{
\text{output/O 的 }0.5\%\text{ 到底是统计 bug，还是 SmolVLA 的真实 FP8 code 分布}
}
$$

并且这套 audit 不改变 forward、scale、outlier mask、RNG 或 SR，因此不会污染你已经完成的 H1。
