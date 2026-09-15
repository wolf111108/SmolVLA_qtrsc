# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：2026-09-15_sparsity-ratio-quickscan
- **状态**：draft（draft / running / done / aborted）
- **负责人**：
- **创建日期**：2026-09-15
- **协议来源**：`README_EXPERIMENT.md`（仓库根，协议契约）与 `results_template (1).md`（表格模板）。本文档为其在实验规范下的落地版本；两者若有出入，以本实验的 configs 实际取值为准并记录于 `logs.md`。
- **相关前序实验**：
  - `../../../2026-09-13_phaseH_accuracy-preserving-sparsity/`（稀疏统计框架与聚合口径的来源；Q2/Q3 复用其 S0/S1 所用 scale）
  - `../../../2026-09-10_phaseG_w4-root-cause/tasks/component-localization/`（G1-A / G1-B 的量化配置与 scale 来源）

---

## 1. 实验目的

用现有 `SmolVLA_qtrsc` 稀疏统计框架，对 4 种量化配置做**元素级 / bit 级 sparsity** 的 workload characterization，回答：

1. INT8 → INT16 位宽增大后，**量化诱导的 element zeros** 是否减少，sign-aware sparse-bit ratio 如何变化？
2. FP8 PoT 与 FP8(A/O)-W4 + PoT(A/O) 的 runtime sparsity 是否接近？差异是否主要落在 **static Linear weight**（FP8 W vs INT4 W）？
3. **prefill（VLM）与 denoise（Expert）** 的 sparsity 分别是什么？（两者必须分开，因为每次 generation 里 prefill 走 1 次、denoise 走 10 个 flow step）

**本实验只做 workload characterization，不做正式 SR claim，不推导硬件加速比。**

---

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval |
| 仓库 revision / commit | 见 `docs/logs.md` §4（运行时 `git rev-parse HEAD` 登记） |
| LeRobot 路径与 commit | `lerobot_current/` @ `6adf5151` |
| GPU | NVIDIA H100 NVL（单卡；运行前须确认无其它实验占用，见 §8） |
| 关键依赖版本 | Python 3.12, torch 2.7.1+cu118, mujoco 3.3.2, `MUJOCO_GL=egl` |

---

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | `lerobot/smolvla_libero`（legacy_public） |
| 评测基准 | LIBERO goal，**task_id=0**，1 episode/config |
| 校准数据 | Q0/Q1：`HuggingFaceVLA/libero@v3.0`，1 episode / batch 8 / frame_stride 16（quick calibration，本实验专用，不用于正式 accuracy 结论）<br>Q2/Q3：**复用**（不重新校准），本地只读副本见下表 |

### 3.1 scale 来源与纪律

| 组 | scale_dir | 来源 | 说明 |
|---|---|---|---|
| Q0 | `scales/2026-09-15_sparsity-ratio-quickscan/q0_int8/` | 本实验 fresh quick calibration | 独立目录 |
| Q1 | `scales/2026-09-15_sparsity-ratio-quickscan/q1_int16/` | 本实验 fresh quick calibration | 独立目录 |
| Q2 | `scales/2026-09-15_sparsity-ratio-quickscan/q2_fp8w4_po2/` | **复制** Phase-G `g1b_w4_all`（864 文件） | 见下「只读副本」 |
| Q3 | `scales/2026-09-15_sparsity-ratio-quickscan/q3_fp8_po2/` | **复制** Phase-G `g1a_fp8_all`（864 文件） | 见下「只读副本」 |

> **只读副本（安全性改进，偏离 README §4 的字面路径）**
>
> README §4 建议 Q2/Q3 的 `scale_dir` 直接指向 `scales/2026-09-10_phaseG_w4-root-cause/component-localization/{g1b_w4_all,g1a_fp8_all}`。本实验**不这么做**，而是先复制到本实验自有目录再指过去，原因：
>
> - 那两个目录是 **Phase H（H1/H2/H3）正在复用**的共享 scale（Phase H setup §2 明文「Phase H 期间禁止覆盖这些 scale」并留有 hash manifest）。若有人误在不加 `--skip-calibration` 的情况下运行 Q2/Q3，框架会**原地重写**这些 scale，破坏 Phase H 的 provenance；
> - 复制件已用 sha256 逐文件校验与原始 **bit-identical**（864/864 文件），因此统计结果完全等价；
> - `run_quickscan.sh` 对 Q2/Q3 **硬编码 `--skip-calibration`**，使破坏路径在 runner 层不可达；
> - 校验清单存于 `docs/manifest/{q2_fp8w4_po2,q3_fp8_po2}_scale_hashes.txt`。

---

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | 1 task × 1 episode，seed=1000（**非 accuracy 评测**） |
| 采样参数 | `n_action_steps=10`，`num_steps=10`（与 Phase G/H 一致，保证 workload 可比） |
| 指标 | runtime element sparsity / runtime bit sparsity / static weight element sparsity / static weight bit sparsity |
| sparsity 汇总范围 | `PREFILL_VLM`：phase=prefill ∧ component=vlm<br>`DENOISE_EXPERT`：phase=denoise ∧ component=expert ∧ flow_step 0..9 |

### 4.1 指标定义

\[
S_{elem} = \frac{\sum \text{zero\_elements\_native}}{\sum \text{total\_elements\_native}}, \qquad
S_{bit} = \frac{\sum \text{sparse\_bits\_native}}{\sum \text{total\_bits\_native}}
\]

**强制使用 `native` 计数**：PoT / outlier 配置的 protected FP sidepath 会在 normal quant path 制造人工 0，用 `reported` 会低估稀疏度。

聚合规范（README §13）：**禁止对逐行比率取平均**，必须先对分子/分母分别求和再相除。`summarize_quick_sparsity.py` 已按此实现。

### 4.2 bit 指标语义（重要）

当前框架的 bit sparsity **不是四种格式统一的存储编码指标**：

| 家族 | 语义 |
|---|---|
| INT8 / INT16 / INT4 | two's-complement sign-aware sparse-bit（正数可跳过 0 bit，负数可跳过符号扩展 1 bit） |
| FP8 E4M3 | 4-bit significand（1MMM / 0MMM）的 zero-bit ratio，**不含 sign / exponent** |

因此：element 级可以横向比较；bit 级只能解释为「各格式自身的 compute-bit skipping opportunity」，**不得**把 FP8 与 INT 的 bit 比率差值写成同一指标的 pp 差。

---

## 5. 实验变量与分组

核心变量：**量化格式/位宽**。固定项：checkpoint、suite/task、episode 数、采样参数、`quantize_matmul=true`、`linear.include=['*']`、`outlier_ratio=0.01`。

| 组 | config | Linear A/W/O | MatMul A/B/O | method | scale | 说明 |
|---|---|---|---|---|---|---|
| Q0 | `q0_int8.yaml` | INT8 / INT8 / INT8 | INT8 | `outlier` | fresh | 8-bit 整数全量化 |
| Q1 | `q1_int16.yaml` | INT16 / INT16 / INT16 | INT16 | `outlier` | fresh | 16-bit 整数全量化（位宽对照） |
| Q2 | `q2_fp8w4_po2.yaml` | E4M3 / **INT4** / E4M3 | E4M3 | Linear `pot_ao_outlier`、MatMul `pot_fp8_outlier` | 复用 G1-B | FP8(A/O)-W4 + PoT(A/O) |
| Q3 | `q3_fp8_po2.yaml` | E4M3 / E4M3 / E4M3 | E4M3 | `pot_fp8_outlier` | 复用 G1-A | FP8 + 全 scale PoT |

**命名纪律**：Q2 的 `pot_ao_outlier` 只保证 **A/O** scale 为 PoT，W4 weight scale 仍是 continuous calibrated scale。因此文档与论文中应写 **FP8(A/O)-W4 + PoT(A/O)**，不得写成「全 PoT」。

### 5.1 sparsity 采集配置

```yaml
sparsity:
  enabled: true
  chunk_size: 4194304   # 显存紧张时降回 1048576
  unit:
    enabled: false      # 本实验不需要 unit/block sparsity
  export:
    dir: null           # null ⇒ 落到 <output_dir>/sparsity/
```

不启用：`fp_code_audit`（缺省 false）、unit/block sparsity、tensor dump、video rendering。

---

## 6. 运行命令

```bash
conda activate smolvla_eval
cd ~/VLA_tcs2

# 一键 4 组串行（GPU 必填）
GPU=0 bash experiments/2026-09-15_sparsity-ratio-quickscan/scripts/run_quickscan.sh

# 单组（Q0/Q1 需校准；Q2/Q3 必须带 --skip-calibration）
python main.py --config experiments/2026-09-15_sparsity-ratio-quickscan/configs/q0_int8.yaml
python main.py --config experiments/2026-09-15_sparsity-ratio-quickscan/configs/q2_fp8w4_po2.yaml --skip-calibration

# 汇总
python experiments/2026-09-15_sparsity-ratio-quickscan/scripts/summarize_quick_sparsity.py
```

---

## 7. 输出目录映射

| config | 输出目录（outputs/ 下） | 状态 |
|---|---|---|
| `q0_int8.yaml` | `outputs/2026-09-15_sparsity-ratio-quickscan/q0_int8/` | pending |
| `q1_int16.yaml` | `outputs/2026-09-15_sparsity-ratio-quickscan/q1_int16/` | pending |
| `q2_fp8w4_po2.yaml` | `outputs/2026-09-15_sparsity-ratio-quickscan/q2_fp8w4_po2/` | pending |
| `q3_fp8_po2.yaml` | `outputs/2026-09-15_sparsity-ratio-quickscan/q3_fp8_po2/` | pending |
| 汇总（Primary） | `outputs/2026-09-15_sparsity-ratio-quickscan/quick_sparsity_summary.csv` | pending |
| 汇总（Secondary） | `outputs/2026-09-15_sparsity-ratio-quickscan/quick_sparsity_by_role.csv` | pending |

单组产物（README §11 Gate）：`sparsity/module_sparsity.csv`、`sparsity/weight_sparsity_static.csv`、`sparsity/quantization_manifest.csv`、`sparsity/outlier_sidepath.csv`。

---

## 8. 风险与注意事项

- **不修改 core code**。`main.py`、`src/vla_tcs2/quant/stat_manager.py`、`quant_methods.py`、`model_wrapper.py` 一律不动。原因：Phase H 的 H3 runner **逐 task 启动新 Python 进程**，若中途改 core code，后续 task 会载入不同代码版本，破坏 H3 一致性。本实验只新增 configs / scripts / docs。
- **GPU 单卡约束**。本机仅 1 张 H100 NVL（GPU 0），且 H3 正在占用。`run_quickscan.sh` **强制要求显式传 `GPU=`**，避免误抢同一张卡；建议等 H3 结束或接受与 H3 共享（会互相稀释算力）。
- **Q2/Q3 必须 `--skip-calibration`**。指向的本地副本虽不影响 Phase H，但重校准会改变数值、使「复用 G1-A/G1-B」的语义失效。runner 已硬编码该 flag。
- **GPU 显存**。`chunk_size=4194304` 已上调以减少统计开销；若 OOM 降回 `1048576`。
- **`1/(1-S)` 不是硬件加速比**。本实验不输出 speedup 结论。
- **`outlier_accounting` 是无操作键**。Phase H 的 yaml 里有该段，但代码从未读取它；native 列无条件导出。故本实验省略该段，不影响结果。
- **`d_bit: 4` / `p: 4` 不是精度**。二者是统计收集用的硬件建模元数据（digit size / parallelism），不改量化。
- **1 episode 的统计噪声**。单 episode 的 sparsity 不能代表全 suite 分布；本实验定位为 quickscan，如需收敛值应扩 episode（对照 Phase H：H1 1ep → H2 3ep 时 sparsity estimate 几乎不变）。
