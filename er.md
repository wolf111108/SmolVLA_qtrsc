我检查了远端 `main` 的最新内容。结论比较明确：

> **Phase H 的实验目录、S0/S1 配置、task-by-task 生成器和 H0/H1 执行脚本已经搭好了，但统计框架的代码修改目前只完成了一部分。现在不建议直接开始正式 H0；至少还有 4 个会影响实验正确性的 P0 问题，其中一个会让 rollout 跑完后直接在导出阶段报错。**

GitHub 当前显示 `main`、27 commits，Phase H 相关文件已经推上远端。([GitHub][1])

## 当前审计结果

| 项目                                      | 当前状态   |            严重度 | 判定               |
| --------------------------------------- | ------ | -------------: | ---------------- |
| Phase H 实验目录与 S0/S1 config              | ✅      |              — | 基本正确             |
| H0/H1 task-by-task 脚本                   | ✅      |              — | 设计正确             |
| 复用 Phase G scale / `--skip-calibration` | ✅      |              — | 正确               |
| `export.dir: null` fallback             | ❌      | **P0 blocker** | 会在导出时报错          |
| static weight collector 真正调用            | ❌      | **P0 blocker** | weight CSV 为空    |
| flow-step 真正拆分 sparsity                 | ⚠️ 半完成 |         **P0** | context 有，但统计仍混合 |
| outlier native sparsity accounting      | ❌      |         **P0** | 当前仍会高估 sparsity  |
| structured unit sparsity                | ❌      |          P0/P1 | 仍只按 phase/layer  |
| YAML `chunk_size` 接入                    | ❌      |             P1 | 当前被忽略            |
| YAML unit key 接入                        | ❌      |             P1 | 配置名与代码不一致        |
| `ideal_speed_up` 改 upper bound          | ❌      |             P1 | 语义仍不规范           |
| task generator deep copy                | ⚠️     |             P2 | 建议修              |

下面逐项说。

---

## 1. 最先修：`export.dir: null` 当前一定有问题

你现在 S0 和 S1 都写了：

```yaml
sparsity:
  ...
  export:
    dir: null
```

([GitHub][2])

但是 `main.py` 现在是：

```python
sparsity_dir = sp_cfg.get("export", {}).get(
    "dir", os.path.join(output_dir, "sparsity")
)
os.makedirs(sparsity_dir, exist_ok=True)
```

([GitHub][3])

Python 的：

```python
dict.get("dir", default)
```

只有在 `dir` **不存在**时才返回 default。

你现在 YAML 显式写：

```yaml
dir: null
```

解析后就是：

```python
{"dir": None}
```

所以：

```python
sparsity_dir = None
os.makedirs(None, exist_ok=True)
```

会直接报错。

更麻烦的是，它发生在：

```text
evaluate()
↓
rollout 全部跑完
↓
sparsity export
↓
TypeError
```

也就是说，你可能白跑一个 episode 甚至更多 episode 后才炸。

### 应立即改成

```python
export_dir = sp_cfg.get("export", {}).get("dir")

sparsity_dir = (
    export_dir
    if export_dir
    else os.path.join(output_dir, "sparsity")
)

os.makedirs(sparsity_dir, exist_ok=True)
```

或者更简洁：

```python
sparsity_dir = (
    sp_cfg.get("export", {}).get("dir")
    or os.path.join(output_dir, "sparsity")
)
```

这是我认为当前的**第一优先级 blocker**。

---

# 2. static weight collector 仍然没有接进 `main.py`

你已经在 `stat_manager.py` 实现好了：

```python
collect_model_weight_sparsity(model)
```

而且实现本身语义是对的：它遍历 `QuantizedLinear`，用实际 `w_interval + w_spec` 重新生成 code 并统计；`QuantizedMatMul` 被正确跳过，因为 attention 的 B 是 runtime K/V，不是 static weight。代码注释也明确要求它在 calibration/scale loading 后调用。([GitHub][4])

但是我重新查了当前远端 `main.py`：

```text
collect_model_weight_sparsity
```

仍然 **0 个调用**。([GitHub][3])

当前 export 直接：

```python
sm.export_per_layer_weight_sparsity_csv(
    os.path.join(sparsity_dir, "weight_sparsity.csv"),
)
```

([GitHub][3])

因此：

```text
per_layer_weight_sparsity = {}
```

大概率一直是空的，最后：

```text
weight_sparsity.csv
```

只有 header。

### 建议在 evaluation 后、export 前加入

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
            "sparsity enabled, but no static quantized weights "
            "were collected"
        )

    export_dir = sp_cfg.get("export", {}).get("dir")
    sparsity_dir = export_dir or os.path.join(
        output_dir, "sparsity"
    )

    ...
```

这个必须在 H0 前补。

---

# 3. `flow_step` 看起来已经改了，但实际上统计仍然没有按 step 拆

这是这次远端修改里最容易被误判为“已经完成”的地方。

你现在确实已经把：

```python
flow_step
attention_kind
```

传进 `_collect_one_tensor_sparsity()` 了：

```python
self._collect_one_tensor_sparsity(
    ...
    tensor_role=tensor_role,
    phase=phase,
    flow_step=flow_step,
    attention_kind=...
)
```

而且函数 docstring 也已经写成：

> `tensor_role/flow_step/attention_kind extend the record key`

([GitHub][4])

**但是实际 key 还没改。**

当前仍然是：

```python
role_key = (
    layer_key,
    phase,
    tensor_role,
)
```

([GitHub][4])

所以：

```text
denoise step 0
denoise step 1
...
denoise step 9
```

对于同一个 module + role，最终仍然全都进入同一个 accumulator。

现在的：

```python
self.flow_step_sparsity
```

只是：

```python
flow_step -> layer -> call_count
```

不是 sparsity numerator/denominator。初始化处甚至明确这么注释。([GitHub][4])

因此当前最多能回答：

> flow step 0~9 都调用过这个层多少次。

不能回答：

> flow step 0 的 bit sparsity 是多少？

### 应改成

```python
flow_step = int(
    -1 if flow_step is None else flow_step
)
attention_kind = attention_kind or "unknown"

role_key = (
    layer_key,
    phase,
    flow_step,
    tensor_role,
    attention_kind,
)
```

同时 entry：

```python
role_entry = self.per_role_sparsity.setdefault(
    role_key,
    {
        "module_id": layer_name,
        "layer_idx": layer_idx,
        "phase": phase,
        "flow_step": flow_step,
        "tensor_role": tensor_role,
        "attention_kind": attention_kind,
    },
)
```

然后 export CSV 必须增加：

```text
flow_step
attention_kind
```

两列。

### 还有一个小错误

现在代码里：

```python
flow_step = ctx.get("flow_step", -1)
flow_step = ctx.get("flow_step", -1)
```

写了两遍。([GitHub][4])

没有数值影响，但说明这块修改还没做完，顺手删掉一行。

---

# 4. outlier accounting 目前基本还只是 YAML，核心代码尚未实现

你的 config 已经很好地加了：

```yaml
outlier_accounting:
  enabled: true
  export_reported: true
  export_native: true
```

S0/S1 都有。([GitHub][2])

`stat_manager.py` 也已经建立：

```python
self.outlier_sidepath = {}
```

并写了漂亮的注释：

```text
Outlier side-path accounting
protected element fraction
```

([GitHub][4])

但是我继续往下查，目前没有：

```python
collect_outlier_partition(...)
```

也没有：

```text
protected_elements
native_zero_rate
native_sparse_bit_rate
```

相关实现。([GitHub][4])

更关键的是，`quant_methods.py` 当前仍然明确写：

```python
# outlier side-path fraction accounting is future work
```

([GitHub][5])

所以现在：

```yaml
outlier_accounting:
  enabled: true
```

**不会真正产生你实验文档里定义的 native sparsity。**

---

## 5. 而当前 outlier datapath 确实仍然会制造人工 0

这一点也重新确认了。

当前代码：

```python
x_fp = x * x_channel_mask
x_normal_fp = x * (~x_channel_mask)

w_fp = layer.weight * w_channel_mask
w_normal_fp = layer.weight * (~w_channel_mask)

x_sim = quant_awo(x_normal_fp, ...)
w_sim = quant_awo(w_normal_fp, ...)
```

([GitHub][5])

也就是说 protected 部分：

```text
original:
[x0 x1 x2 x3 ...]

mask:
[ 0  1  0  0 ...]

normal path:
[x0  0 x2 x3 ...]
```

第二个元素根本不是因为量化变成 0，它是因为已经被送去 FP side path。

现在统计的又恰好是：

```python
x_sim
```

所以如果不加 protected counter：

$$
S_{\text{current}}
=
S_{\text{native quant}}
+
S_{\text{artificial mask}}
$$

会系统性高估硬件能够利用的 sparsity。

因此这个不是“以后优化一下”的项目，而是 **Phase H 论文级 sparsity 实验的 P0 correctness 问题**。

---

# 6. unit sparsity 仍然没有 structured 化

现在 unit 数据结构仍然是：

```python
# phase -> layer_key -> ...
self.unit_sparsity
```

([GitHub][4])

并且 `_collect_one_tensor_sparsity()` 最后还是：

```python
if self.enable_unit_sparsity:
    self.collect_unit_sparsity(
        layer_name,
        layer_idx,
        activation,
        spec,
    )
```

([GitHub][4])

这里没有：

```text
tensor_role
flow_step
attention_kind
```

所以现在 unit sparsity 会把：

```text
Linear activation
Linear output
MatMul A
MatMul B
MatMul O
```

以及所有 denoise step 混在同一个 layer/phase 桶里。

因此如果现在运行，**不要用当前 `unit_sparsity.csv` 去画 flow-step 或 A/B/O 的 unit sparsity 图**。

建议新增：

```python
self.per_role_unit_sparsity = {}
```

key 和 element/bit 完全一致：

```python
(
    layer_key,
    phase,
    flow_step,
    tensor_role,
    attention_kind,
)
```

当前远端里还没有这个结构。([GitHub][4])

---

# 7. 配置和 `main.py` 还有两个 schema 没接上

这两个不会立刻破坏本轮默认结果，但现在修掉最合适。

你 YAML 写：

```yaml
sparsity:
  chunk_size: 1048576

  unit:
    enabled: true
    bit_group_size: 2
    dim_group_size: 2
```

([GitHub][2])

但是 `main.py` 当前调用：

```python
sm.enable_sparsity()
```

根本没有把：

```yaml
chunk_size
```

传进去。([GitHub][3])

而 `QuantStatManager.enable_sparsity()` 本身明明已经支持：

```python
enable_sparsity(
    enable=True,
    chunk_size=None,
)
```

([GitHub][4])

应改成：

```python
sm.enable_sparsity(
    enable=True,
    chunk_size=sp_cfg.get("chunk_size"),
)
```

---

另一处是 unit key。

YAML：

```yaml
bit_group_size: 2
dim_group_size: 2
```

但 `main.py` 读的是：

```python
bit_group_size=unit_cfg.get("rows", 2),
dim_group_size=unit_cfg.get("cols", 2),
```

([GitHub][3])

本轮由于两边默认值刚好都是 `2,2`，所以**数值恰好没错**。

但例如以后你改：

```yaml
bit_group_size: 4
dim_group_size: 8
```

代码仍然悄悄使用：

```text
2 × 2
```

这非常危险。

建议兼容旧格式：

```python
bit_group_size = unit_cfg.get(
    "bit_group_size",
    unit_cfg.get("rows", 2),
)

dim_group_size = unit_cfg.get(
    "dim_group_size",
    unit_cfg.get("cols", 2),
)
```

---

# 8. `ideal_speed_up` 也还没有改

当前仍然：

```python
entry["ideal_speed_up"] = (
    1 / (1 - entry["sparse_bit_rate"])
    ...
)
```

CSV header 也继续叫：

```text
ideal_speed_up
```

([GitHub][4])

建议至少同时输出：

```python
entry["ideal_sparse_upper_bound"] = ...
```

旧字段可以保留一段时间兼容历史脚本：

```python
entry["ideal_speed_up_legacy"] = ...
```

Phase H 的图表和结果只读新字段。

---

# 9. 实验配置本身是对的

这部分修改总体不错。

S0 保持了：

```text
lerobot/smolvla_libero
n_action_steps=10
num_steps=10
pot_fp8_outlier
G1-A 原 scale_dir
QK/PV quantized
Goal
seed=1000
```

([GitHub][2])

S1 也正确复用了 G1-D 的 scale，Linear 只：

```yaml
include:
  - expert.*
```

并把 Expert q/k/v/o/gate/up/down 的 W 设成 4 bit，而 A/O 保持 E4M3，QK/PV 仍为 E4M3。([GitHub][6])

因此 **S0/S1 没必要重配**。问题主要在 collector wiring。

H0 脚本也正确：

```bash
python main.py --config "$cfg" --skip-calibration
```

并且只跑 S0/S1 task0×1，符合之前设计。([GitHub][7])

---

# 10. `make_task_configs.py` 有一个低优先级问题

现在：

```python
cfg = dict(base)
```

是 shallow copy。([GitHub][8])

随后：

```python
cfg["evaluation"]["env"]["task_ids"] = [tid]
cfg["evaluation"]["n_episodes"] = ep_per_task
```

会修改嵌套 dict。

按你当前的单进程生成方式不一定马上造成明显错误，但这是非常容易以后踩坑的写法。

建议：

```python
import copy

...

cfg = copy.deepcopy(base)
```

这是 P2，不阻塞 H0，但顺手修掉最好。

---

# 11. 我建议你现在按这个最小 patch 顺序修改

1. **先修 `main.py`**

   * `export.dir: null` fallback；
   * 接入 `chunk_size`；
   * 兼容 `bit_group_size/dim_group_size`；
   * evaluation 后调用 `collect_model_weight_sparsity()`。

2. **再修 `stat_manager.py`**

   * `role_key` 加 `flow_step + attention_kind`；
   * entry/export 加这两列；
   * 去掉重复 `flow_step = ...`；
   * 新增 structured unit accumulator。

3. **再修 outlier accounting**

   * activation protected elements；
   * runtime weight protected elements；
   * output protected elements；
   * `reported` 与 `native` numerator/denominator 同时保存；
   * 导出 `fp_sidepath_ratio`。

4. **最后修实验小问题**

   * `deepcopy()`；
   * `ideal_sparse_upper_bound`；
   * config/document status。

---

## 建议直接采用的 `main.py` 核心修改

```python
sp_cfg = config.get("sparsity", {})

if sp_cfg.get("enabled", False):
    sm = wrapper.stat_manager

    if sm is None:
        raise RuntimeError(
            "sparsity.enabled=true requires a stat manager"
        )

    sm.enable_sparsity(
        enable=True,
        chunk_size=sp_cfg.get("chunk_size"),
    )

    unit_cfg = sp_cfg.get("unit", {})

    if unit_cfg:
        sm.configure_unit_sparsity(
            enable=unit_cfg.get("enabled", True),
            bit_group_size=unit_cfg.get(
                "bit_group_size",
                unit_cfg.get("rows", 2),
            ),
            dim_group_size=unit_cfg.get(
                "dim_group_size",
                unit_cfg.get("cols", 2),
            ),
        )
```

evaluation 后：

```python
if sp_cfg.get("enabled", False) and wrapper.stat_manager is not None:
    sm = wrapper.stat_manager

    n_weight_layers = sm.collect_model_weight_sparsity(model)

    if n_weight_layers <= 0:
        raise RuntimeError(
            "No static QuantizedLinear weight sparsity collected."
        )

    export_dir = sp_cfg.get("export", {}).get("dir")
    sparsity_dir = (
        export_dir
        or os.path.join(output_dir, "sparsity")
    )

    os.makedirs(sparsity_dir, exist_ok=True)

    ...
```

---

# 当前 Gate 判定

我会把当前远端状态定为：

```text
Phase H experiment setup       PASS
S0/S1 quant config             PASS
task-by-task orchestration     PASS
scale reuse protocol           PASS

static weight wiring           FAIL
flow-step sparsity split       FAIL / partial implementation
outlier native accounting      FAIL
structured unit sparsity       FAIL
export fallback                FAIL — runtime blocker

--------------------------------------
H0 READY                       NO
```

所以**现在先不要正式启动 H1/H2/H3**。H0 如果已经启动也问题不大，但当前版本即便 rollout 正常，`export.dir: null` 这一个问题就足以让它在统计导出阶段失败；即使把这个 crash 修掉，现阶段输出的 `weight_sparsity`、flow-step sparsity、outlier native sparsity 仍然不满足我们之前定义的实验语义。

等你把上面几个 P0 修改再推一次后，我建议我再做一次远端 code review。下一次主要只需要核查 **`main.py + stat_manager.py + quant_methods.py` 三个文件以及 H0 config**；如果全部通过，就可以直接给出 H0 的正式启动命令和 CSV 验收 checklist。

[1]: https://github.com/wolf111108/SmolVLA_qtrsc "GitHub - wolf111108/SmolVLA_qtrsc · GitHub"
[2]: https://github.com/wolf111108/SmolVLA_qtrsc/blob/main/experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/configs/s0_fp8_all_base.yaml "SmolVLA_qtrsc/experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/configs/s0_fp8_all_base.yaml at main · wolf111108/SmolVLA_qtrsc · GitHub"
[3]: https://github.com/wolf111108/SmolVLA_qtrsc/blob/main/main.py "SmolVLA_qtrsc/main.py at main · wolf111108/SmolVLA_qtrsc · GitHub"
[4]: https://github.com/wolf111108/SmolVLA_qtrsc/blob/main/src/vla_tcs2/quant/stat_manager.py "SmolVLA_qtrsc/src/vla_tcs2/quant/stat_manager.py at main · wolf111108/SmolVLA_qtrsc · GitHub"
[5]: https://github.com/wolf111108/SmolVLA_qtrsc/blob/main/src/vla_tcs2/quant/quant_methods.py "SmolVLA_qtrsc/src/vla_tcs2/quant/quant_methods.py at main · wolf111108/SmolVLA_qtrsc · GitHub"
[6]: https://github.com/wolf111108/SmolVLA_qtrsc/blob/main/experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/configs/s1_expert_w4_base.yaml "SmolVLA_qtrsc/experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/configs/s1_expert_w4_base.yaml at main · wolf111108/SmolVLA_qtrsc · GitHub"
[7]: https://github.com/wolf111108/SmolVLA_qtrsc/blob/main/experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/run_h0_smoke.sh "SmolVLA_qtrsc/experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/run_h0_smoke.sh at main · wolf111108/SmolVLA_qtrsc · GitHub"
[8]: https://github.com/wolf111108/SmolVLA_qtrsc/blob/main/experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/make_task_configs.py "SmolVLA_qtrsc/experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/scripts/make_task_configs.py at main · wolf111108/SmolVLA_qtrsc · GitHub"
