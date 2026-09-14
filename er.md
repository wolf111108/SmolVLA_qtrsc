我重新读了仓库当前 `main` 上的最新 `model_wrapper.py`、G6 四个配置和运行脚本。结论是：**component-aware precision routing 的核心实现方向是对的，G6 配置本身也基本正确，但现在还不建议直接开始 4×100 episodes。至少有 2 个阻塞问题、2 个框架安全问题需要先修。**

## 审计结论

| 项目                                   | 状态                       | 严重度         |
| ------------------------------------ | ------------------------ | ----------- |
| `resolve_linear_quant_config()`      | ✅ 正确                     | —           |
| override → physical `module_id`      | ✅ 正确                     | —           |
| VLM/Expert 同名 operator 可不同 precision | ✅ 已实现                    | —           |
| `per_site` mixed-precision guard     | ✅ 正确                     | —           |
| G6-A/B/C/D YAML precision routing    | ✅ 正确                     | —           |
| 224 Linear / 64 MatMul 理论计数          | ✅ 正确                     | —           |
| `run_smoke.sh` repo root             | ❌ 路径少一层 `..`             | **BLOCKER** |
| `run_goal.sh` repo root              | ❌ 同上                     | **BLOCKER** |
| `audit_routing.py` 本地 `src` 路径       | ❌ 少一层 `..`               | **HIGH**    |
| override `target` typo safety        | ⚠️ 可能静默匹配全部 Linear       | **HIGH**    |
| Gate 3 raw equivalence               | ⏳ 尚未完成                   | **必须先做**    |
| Gate 5 task0×1                       | ⏳ 尚未完成                   | **必须先做**    |
| audit count 自动 assert                | ⚠️ 目前只打印                 | MEDIUM      |
| `linear.enabled`                     | ⚠️ `_should_wrap()` 没使用它 | LOW/旧问题     |

下面详细解释。

---

# 1. 核心 precision routing 实现是正确的

现在 `create_quantized_linear()` 已经增加：

```python
module_id: str | None = None
```

并在真正构造 `QuantizedLinear` 前执行：

```python
layer_config, matched_overrides = resolve_linear_quant_config(
    quant_config=quant_config,
    layer_type=layer_type,
    module_id=physical_id,
)
```

因此 precision 不再只由：

```text
q_proj
k_proj
...
```

决定，而是可以进一步根据 physical module identity 覆盖。并且最终 effective config 真正用于 `a_bit/w_bit/o_bit/method` 构造，不只是 metadata。

同时你已经把：

```python
module_id=module_id
```

传入 VLM Attention、VLM MLP、Expert Attention、Expert MLP 的所有 `create_quantized_linear()` 调用。也就是说：

```text
vlm.layers.3.self_attn.q_proj
expert.layers.3.self_attn.q_proj
```

现在确实可以获得不同的 effective precision。

这一部分我认为 **PASS**。

---

# 2. Resolver 的 override precedence 也是正确的

现在：

```python
base = dict(quant_config.get(layer_type, {}) or {})
```

先取 operator base config，然后按 YAML 顺序：

```python
if _matches_target(module_id, target):
    base.update(patch)
```

因此实现的是：

$$
\text{operator default}
\rightarrow
\text{override 0}
\rightarrow
\text{override 1}
\rightarrow\cdots
$$

后匹配项覆盖前匹配项，符合之前设计。你还限制了 override 可修改：

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

未知 **config field** 会直接报错，这是好的 fail-fast。

---

# 3. `per_site` guard 实现正确，而且非常必要

你现在明确做了：

```python
if overrides and granularity != "per_site":
    raise ValueError(...)
```

这是正确的。

因为现在完全可能出现：

```text
VLM q_proj    = W4
Expert q_proj = FP8
```

如果两者还共用一个 calibration scale，就会产生语义错误。

而当前：

```python
resolve_linear_scale_group(..., "per_site")
```

产生：

```text
vlm_q_proj, layer_idx
expert_q_proj, layer_idx
```

两套独立 scale identity。

因此：

$$
\boxed{\text{precision routing 与 scale persistence 当前是相容的}}
$$

这一项也是 **PASS**。

---

# 4. G6 四个配置本身是正确的

G6-A：

```text
VLM Linear    = FP8
Expert Linear = FP8
```

因为 include 同时包含：

```yaml
- vlm.*
- expert.*
```

且没有 override。

G6-B 默认所有 Linear FP8，然后：

```yaml
target:
  module_id: vlm.layers.*.self_attn.*_proj
config:
  w_bit: 4
  method: pot_ao_outlier
```

所以得到：

$$
\boxed{
VLM\ Attention=A8W4O8,\quad
VLM\ MLP=A8W8O8,\quad
Expert=A8W8O8
}
$$

完全正确。

G6-C 对：

```text
vlm.layers.*.mlp.*_proj
```

做 W4，因此理论上：

$$
16\times3=48
$$

个 W4，其余 176 个 Linear FP8。

G6-D：

```text
vlm.layers.*.*.*_proj
```

会覆盖 VLM 的 Attention + MLP：

$$
16\times7=112
$$

所以：

```text
112 VLM W4
112 Expert FP8
```

也是正确的。

而实验文档记录的 Gate 2：

```text
G6-A 224/0
G6-B 160/64
G6-C 176/48
G6-D 112/112
MatMul = 64
```

与理论完全一致。

---

# 5. BLOCKER 1：`run_smoke.sh` 的 `REPO_ROOT` 算错了

现在代码：

```bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
```



但脚本实际位置是：

```text
repo/
└── experiments/
    └── 2026-09-10_phaseG_w4-root-cause/
        └── tasks/
            └── vlm-selective-expert-fp8/
                └── scripts/
                    └── run_smoke.sh
```

从 `scripts/` 回 repo：

```text
..          task
../..       tasks
../../..    experiment
../../../.. experiments
../../../../.. repo
```

需要 **5 个 `..`**。

你现在只有 4 个，因此：

```text
REPO_ROOT = ~/VLA_tcs2/experiments
```

而不是：

```text
~/VLA_tcs2
```

后面：

```bash
TASK="$REPO_ROOT/experiments/..."
```

会变成：

```text
~/VLA_tcs2/experiments/experiments/...
```

而且：

```bash
cd "$REPO_ROOT"
python main.py
```

会在：

```text
~/VLA_tcs2/experiments
```

寻找 `main.py`。

### 应改成

```bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
```

这不是 cosmetic issue，是 **会直接导致 smoke 脚本无法按预期执行的 blocker**。

---

# 6. BLOCKER 2：`run_goal.sh` 有完全相同的问题

当前也是：

```bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
```



必须同样改成：

```bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
```

否则正式 G6 运行脚本的：

```text
TASK
out
main.py
```

路径都会错。

所以目前 **不要直接执行 `run_goal.sh`**。

---

# 7. HIGH：`audit_routing.py` 的 `sys.path` 也少了一层

现在：

```python
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "src"
    )
)
```



从：

```text
.../task/scripts
```

四个 `..` 只能到：

```text
repo/experiments
```

所以实际插入的是：

```text
repo/experiments/src
```

而不是：

```text
repo/src
```

正确应该是五层：

```python
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..", "..", "src"
        )
    )
)
```

更推荐彻底避免数 `..`：

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "src"))
```

这里尤其需要修，因为现在 Gate 2 虽然显示已经跑通，但错误 `sys.path` 意味着：

> audit 脚本可能依赖 conda 环境里已有的 editable install，而不是明确保证加载当前 checkout 的 `src/vla_tcs2`。

换句话说，**Gate 2 的结果目前可信，但它的 reproducibility 不够强**。

你脚本后面计算 output repo root 的“dirname 六次”反而是正确的；错的是最前面的 import path。

---

# 8. HIGH：override 的 `target` typo 可能造成“全模型误匹配”

这是我这次审计里发现的最重要的 framework safety 问题。

当前 `_matches_target()` 开头：

```python
if not target:
    return False
```

然后只识别：

```text
module_id
module_ids
component
layer
operator
```

最后直接：

```python
return True
```



这意味着如果有人误写：

```yaml
target:
  componet: vlm
```

注意把 `component` 写成了 `componet`。

`target` 并不为空，所以不会在开头返回 False。

之后：

```python
target.get("module_id")  -> None
target.get("component")  -> None
target.get("layer")      -> None
target.get("operator")   -> None
```

最后：

```python
return True
```

结果是：

$$
\boxed{\text{这个 typo 可能匹配所有 Linear}}
$$

然后：

```yaml
w_bit: 4
```

可能把整个 wrapped model 全改成 W4。

这是危险的。

### 当前 G6 不受影响

因为你的配置用的是合法：

```yaml
target:
  module_id: vlm.layers.*.self_attn.*_proj
```

所以当前四组不会因此路由错误。

但是既然你已经把它做成 framework feature，我建议正式 rollout 前就修。

最小修法是在 `resolve_linear_quant_config()` 里验证 target：

```python
_ALLOWED_LINEAR_TARGET_KEYS = {
    "module_id",
    "module_ids",
    "component",
    "layer",
    "operator",
}

unknown_target = set(target) - _ALLOWED_LINEAR_TARGET_KEYS
if unknown_target:
    raise ValueError(
        f"Unsupported linear override target fields at index {idx}: "
        f"{sorted(unknown_target)}"
    )

if not any(k in target for k in _ALLOWED_LINEAR_TARGET_KEYS):
    raise ValueError(
        f"linear.overrides[{idx}].target has no supported selector"
    )
```

这样：

```yaml
componet: vlm
```

立即 fail-fast。

---

# 9. Audit 脚本目前“能看结果”，但还不是严格 Gate

当前：

```python
is_w4 = str(w_bit) == "4"

if is_w4:
    n_w4 += 1
else:
    n_fp8 += 1
```



所以任何不是 `"4"` 的精度都会被算成 FP8。

例如：

```text
w_bit = 8
w_bit = 16
w_bit = e5m2
```

理论上都会落进：

```text
n_fp8 += 1
```

G6 当前 YAML 确实只有：

```text
e4m3
4
```

因此当前统计数字没问题，但作为 audit gate 建议改成严格判断：

```python
if str(w_bit) == "4":
    n_w4 += 1
elif str(w_bit).lower() == "e4m3":
    n_fp8 += 1
else:
    raise AssertionError(
        f"Unexpected w_bit={w_bit!r} at {mid}"
    )
```

还应该同时验证 method：

```python
if str(w_bit) == "4":
    assert module.method == "pot_ao_outlier"
else:
    assert module.method == "pot_fp8_outlier"
```

现在 CSV 已经把 method 输出出来了，但脚本没有 assert。

---

# 10. 建议让 routing Gate 自动判断 count

当前 audit 只是：

```text
print FP8
print W4
print Total
print MatMul
```

不会因为 count 错误返回非零状态。

建议增加：

```python
EXPECTED = {
    "g6a_all_fp8_control": (224, 0, 64),
    "g6b_vlm_attn_w4_expert_fp8": (160, 64, 64),
    "g6c_vlm_mlp_w4_expert_fp8": (176, 48, 64),
    "g6d_vlm_all_w4_expert_fp8": (112, 112, 64),
}

expected_fp8, expected_w4, expected_mm = EXPECTED[cfg_name]

assert n_fp8 == expected_fp8
assert n_w4 == expected_w4
assert n_matmul == expected_mm
```

那么 Gate 2 才真正是：

```text
exit 0 = PASS
non-zero = FAIL
```

而不是人工看日志。

---

# 11. `linear.enabled` 目前其实没有生效

这是一个旧问题，不是本次新增的。

配置里有：

```yaml
linear:
  enabled: true
```

但是 `_should_wrap()` 只读取：

```python
include_patterns
exclude_patterns
```

完全不读取：

```python
linear_cfg["enabled"]
```



所以理论上：

```yaml
linear:
  enabled: false
```

只要 quantization 顶层：

```yaml
enabled: true
```

Linear 仍然会被 wrap。

当前 G6 全部是：

```yaml
enabled: true
```

因此不影响当前实验。

但顺手可以修成：

```python
def _should_wrap(module_name, quant_config):
    linear_cfg = quant_config.get("linear", {})

    if not linear_cfg.get("enabled", True):
        return False

    ...
```

我把它定为 LOW，是因为它不影响当前 G6，但属于 framework semantics bug。

---

# 12. G6 calibration 的 scale isolation 没问题

这一点我专门核查了。

当前 G6：

```yaml
linear_scale_granularity: per_site
```



对应：

```python
if granularity == "per_site":
    return f"{component}_{name}", layer_idx
```



因此比如：

```text
VLM layer3 q_proj
→ vlm_q_proj_w_scale_3.p

Expert layer3 q_proj
→ expert_q_proj_w_scale_3.p
```

不会发生：

```text
VLM W4 scale
```

覆盖：

```text
Expert FP8 scale
```

的情况。

这部分可以放心。

---

# 13. Calibration policy 对当前 G6 也没问题

现在所有：

```text
q/k/v/o/gate/up/down
qk/pv
```

都明确：

```yaml
recalibrate
```



虽然 `resolve_calibration_policy()` 目前并没有 component-aware policy，只看：

```text
layer_type
layer_idx
```



但因为 G6 所有 physical site 都重新校准，所以当前不产生错误。

只是以后如果想做：

```text
VLM q_proj recalibrate
Expert q_proj reuse
```

现有 `calibration_policy` 还表达不了。

这不是 G6 blocker。

---

# 14. Gate 3 现在确实还缺

你最新 `experiment_setup.md` 已经诚实记录：

```text
[x] Gate 0
[x] Gate 1
[x] Gate 2
[ ] Gate 3 raw Linear equivalence
[ ] Gate 4 calibration smoke
[ ] Gate 5 task0×1
[ ] Gate 6 Goal×100
```



这个顺序是对的。

尤其现在 `create_quantized_linear()` signature 和 routing 都改过，我不建议跳过 Gate 3。

需要验证：

$$
QLinear_{\text{mode=raw}}(x)
=
F.linear(x,W,b)
$$

至少抽：

```text
VLM q_proj
VLM down_proj
Expert q_proj
Expert down_proj
```

并包含 G6-B/C/D 下实际被 override 的 module。

实际上 raw path 仍然就是：

```python
if self.mode == "raw":
    return F.linear(x, self.weight, self.bias)
```



所以理论上不会有问题，但 **Gate 的价值是确认 wrap/copy/routing 没有破坏参数**。

---

# 15. Gate 5 目前没有对应执行脚本

现在 repo 里的 `run_smoke.sh` 实际只做：

```bash
python main.py ... --skip-evaluation
```

也就是 **Gate 4 calibration-only smoke**。

它没有做：

```text
task0 × 1 episode
```

而实验文档又单独定义了 Gate 5。

因此建议补：

```text
run_task0_smoke.sh
```

而不是拿完整 10-task config 直接试。

---

# 我建议现在修改这几处

按优先级：

```text
P0  修 run_smoke.sh REPO_ROOT
P0  修 run_goal.sh REPO_ROOT

P1  修 audit_routing.py local src path
P1  override target unknown-key fail-fast
P1  完成 Gate 3 raw equivalence

P2  audit_routing 增加 exact bit/method/count assert
P2  增加 task0×1 smoke
P2  修 linear.enabled semantics
```

### Bash 路径统一改成

```bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
```

### Python 脚本统一用

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "src"))
```

这样以后目录层级一眼就能审计。

---

## 最终判断

**Core routing：PASS。**

现在确实已经实现了你之前缺失的：

$$
\boxed{
\text{VLM operator precision}
\neq
\text{Expert operator precision}
}
$$

而且 G6 四个 YAML 的数学语义与我们的设计一致：

$$
\boxed{
G6A: VLM8 + Expert8
}
$$

$$
\boxed{
G6B: VLM\ Attn4,\ VLM\ MLP8,\ Expert8
}
$$

$$
\boxed{
G6C: VLM\ Attn8,\ VLM\ MLP4,\ Expert8
}
$$

$$
\boxed{
G6D: VLM4 + Expert8
}
$$

但**执行基础设施暂时不是 PASS**：两个 shell runner 的根目录计算存在确定性错误，routing audit 的 local source path 也有错误。另外 override selector 还有一个值得马上封死的“target typo → potentially match all”的安全漏洞。

所以我建议当前状态定义成：

$$
\boxed{\text{Framework core PASS, experiment execution NOT YET READY}}
$$

把上面 P0/P1 修完，再跑 Gate 3 → Gate 4 → Gate 5；这三个全部 PASS 后，就可以放心开始 G6 Goal×100。
