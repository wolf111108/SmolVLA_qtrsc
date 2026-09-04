# Phase D / E 测试指令

本文档列出 Phase D（校准/量化验证）与 Phase E（sensitivity）的全部测试脚本、
config 与执行命令。所有命令均在 `conda activate smolvla_eval` 环境下、仓库根目录
`/home/zyzhao/VLA_tcs2` 下执行。

> 前置：`git log` 确认已包含 `module_id/scale_group 解耦` 提交（`97254f4`）。

---

## 0. 快速自检（不跑 LIBERO，秒级~分钟级）

这些脚本只用随机合成输入 + `checkpoints/smolvla_base` 本地 checkpoint，
验证底层 identity / routing / 量化链路，不依赖真实数据集和 MuJoCo。

```bash
conda activate smolvla_eval

# Phase A/B/C 回归（raw 等价 + 4 档粒度 scale 数）
python scripts/test_phaseB_raw_equiv.py checkpoints/smolvla_base per_site
python scripts/test_phaseB_e2e_forward.py checkpoints/smolvla_base
python scripts/test_phaseC_calibration.py checkpoints/smolvla_base per_site
```

---

## Phase D — 校准 / 量化验证

### D1. Preflight audit（§24）—— 身份不变量检查

```bash
python scripts/test_phaseD_preflight.py checkpoints/smolvla_base
```

**通过标准**（逐粒度打印）：
- 物理对象数：MatMul=64、Linear=224
- 唯一 `module_id`：64 + 224，且 MatMul/Linear 不重叠
- 唯一 scale group 数 = 粒度预期值（下表）

| granularity | MatMul 组 | Linear 组 | scale 文件 |
|---|---:|---:|---:|
| global | 2 | 7 | 27 |
| per_component | 4 | 14 | 54 |
| per_layer | 32 | 112 | 432 |
| per_site | 64 | 224 | 864 |

### D2. Call-count audit（§25）—— 证明 64 MatMul 真的被执行

```bash
python scripts/test_phaseD_call_count.py checkpoints/smolvla_base
```

**通过标准**：打印 `PHASE D CALL-COUNT: ALL 64 MATMUL EXECUTED`，
且无 `NEVER CALLED` 条目（prefix→vlm、denoise→expert 覆盖全部 64 个 site）。

### D3. Quantization sanity（§27）—— 4 步递进量化检查

```bash
python scripts/test_phaseD_quant_sanity.py checkpoints/smolvla_base
```

**通过标准**：Step 1~3 输出全部 finite，Step 4 打印 4 档粒度的
MatMul 组数与 scale 文件数都正确，最后 `ALL STEPS PASSED`。

### D4. 正式 granularity ablation（§28）—— 真实 LIBERO 校准 + 评测

四个 config 已就绪（pot_fp8_outlier + 4 档粒度）：

```bash
# 单档（示例 per_site）
python main.py --config configs/experiments/phaseD_ablation_per_site.yaml

# 一键跑完 4 档（顺序：per_site → per_layer → per_component → global）
bash scripts/run_phaseD_ablation.sh
```

四档 config 文件：
- `configs/experiments/phaseD_ablation_per_site.yaml`
- `configs/experiments/phaseD_ablation_per_layer.yaml`
- `configs/experiments/phaseD_ablation_per_component.yaml`
- `configs/experiments/phaseD_ablation_global.yaml`

每个粒度独立 `scale_dir`（`scales/phaseD_ablation_<g>/`）与 `output_dir`
（`outputs/experiments/phaseD_ablation_<g>/`），**禁止混用 scale 目录**。

汇总后应得到形如下表的结果：

| MatMul scale granularity | scale 文件 | Object SR |
|---|---:|---:|
| per_site | 864 | TBD |
| per_layer | 432 | TBD |
| per_component | 54 | TBD |
| global | 27 | TBD |

> 默认 `task: libero_object`、`n_episodes: 10`。要跑满 4 个 suite 或更多
> episode，改对应 config 的 `evaluation` 块即可。

---

## Phase E — Sensitivity（§19 / §29）

### E1. 前向扰动 sensitivity map（快，免校准）

逐 site 注入 Gaussian RMS 噪声（其余 raw），测 suffix hidden 输出扰动，
按 `max_abs_diff` 降序输出 CSV。用于快速定位敏感层，再对 top 候选跑闭环 SR。

```bash
# 单 site
python scripts/test_phaseE_sensitivity.py checkpoints/smolvla_base \
    --module-id expert.layer.7.qk --out-dir outputs/sensitivity

# 全量 sweep（64 MatMul + 224 Linear）
python scripts/test_phaseE_sensitivity.py checkpoints/smolvla_base \
    --sweep all --out-dir outputs/sensitivity

# 分组 overview（component × operator，共 14 组代表）
python scripts/test_phaseE_sensitivity.py checkpoints/smolvla_base \
    --sweep group --out-dir outputs/sensitivity

# 换噪声方法 / 幅度 / 注入位置
python scripts/test_phaseE_sensitivity.py checkpoints/smolvla_base \
    --sweep group --method gaussian_rms_input --alpha 0.05 --site input
```

**输出**：`outputs/sensitivity/sensitivity.csv`，字段
`module_id, component, layer, operator, max_abs_diff, mean_abs_diff, relative_norm`，
按 `max_abs_diff` 降序。

**可注入方法**（`--method`）：
- `gaussian_rms_{input|weight|output}`（Linear）/ `gaussian_rms_{A|B|output}`（MatMul）
- `quant_residual_{input|weight|output}`（Linear）/ `quant_residual_{A|B|output}`（MatMul）
  —— 需要先校准出 scale（见 E2）

### E2. 闭环 SR sensitivity（慢，需真实数据 + 校准）

对前向 map 的 top 敏感 site 做 closed-loop SR 确认。config 已就绪：
`configs/experiments/phaseE_sensitivity.yaml`（默认 target `expert.layer.7.qk`）。

```bash
# gaussian 方法（免校准，一步到位）
python main.py --config configs/experiments/phaseE_sensitivity.yaml

# quant_residual 方法（先校准 scale，再复用注入）
python main.py --config configs/experiments/phaseE_sensitivity.yaml --skip-evaluation
python main.py --config configs/experiments/phaseE_sensitivity.yaml --skip-calibration
```

换 target 只需改 config 的 `test.target`：

```yaml
test:
  target:
    # 单 site
    module_id: expert.layer.7.qk
    # 或 selector 组合（AND 关系）
    # component: expert
    # operator: qk
    # layer: [0, 1, 2, 3]
```

**支持 selector**（`_matches_target`）：
- `module_id`（精确/glob）、`module_ids`（列表 OR）
- `component`（`vlm`/`expert`）
- `layer`（int 或 list）
- `operator`（`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj`/`qk`/`pv`）

---

## 建议执行顺序

```text
1. D1 preflight        —— 秒级，验证身份不变量
2. D2 call-count       —— 秒级，验证 64 MatMul 全被执行
3. D3 quant sanity     —— 分钟级，验证 4 步量化链路
4. E1 sensitivity map  —— 分钟级（sweep group）/ 数十分钟（sweep all）
5. D4 granularity ablation —— 小时级（真实校准 + 评测）
6. E2 closed-loop SR   —— 小时级（对 E1 的 top 敏感 site 确认）
```

## 结果落盘位置

- `outputs/sensitivity/sensitivity.csv` —— E1 敏感度表
- `outputs/experiments/phaseD_ablation_<g>/result.json` —— D4 各粒度 SR
- `outputs/experiments/phaseE_sensitivity/result.json` —— E2 闭环 SR
- `scales/phaseD_ablation_<g>/`、`scales/phaseE_sensitivity/` —— 校准 scale
