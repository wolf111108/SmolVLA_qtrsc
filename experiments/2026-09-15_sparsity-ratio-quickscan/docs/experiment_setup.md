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

**强制使用 `native` 计数**：PoT / outlier 配置的 protected FP sidepath 会在 normal quant path 中制造人工 0（protected 位置的真值非零，但 normal path 存的是 0），因此 `reported` 会**高估**可归属于 native quantized workload 的稀疏度。本实验统一使用 `native` counters，排除 protected positions。

聚合规范（README §13）：**禁止对逐行比率取平均**，必须先对分子/分母分别求和再相除。`summarize_quick_sparsity.py` 已按此实现。

### 4.2 bit 指标语义（重要）

当前框架的 bit sparsity **不是四种格式统一的存储编码指标**：

| 家族 | 语义 |
|---|---|
| INT8 / INT16 / INT4 | two's-complement sign-aware sparse-bit（正数可跳过 0 bit，负数可跳过符号扩展 1 bit） |
| FP8 E4M3 | 4-bit significand（1MMM / 0MMM）的 zero-bit ratio，**不含 sign / exponent** |

因此：element 级可以横向比较；bit 级只能解释为「各格式自身的 compute-bit skipping opportunity」，**不得**把 FP8 与 INT 的 bit 比率差值写成同一指标的 pp 差。

> **位宽伪影（实测，见 `results.md` §3.2）**：即使 INT 家族内部，sign-aware 指标也带有**位宽偏置**——每个负元素固定贡献 1 个可跳过的符号位，而这 1 bit 在分母里的权重是 $1/8 = 12.5\%$（INT8）vs $1/16 = 6.25\%$（INT16）。因此 **INT8 与 INT16 的 bit sparsity 差值中约 6.25pp 纯粹来自位宽**，不反映真实数据稀疏度差异。**跨位宽比较 bit sparsity 前必须先扣除该项**，否则会系统性高估低位宽收益。

### 4.3 统计范围（Scope）—— 明确不是「整个 SmolVLA」

当前量化框架显式包装的 target 只有：

```text
VLM text_model 16 层：  q/k/v/o + gate/up/down
Action Expert 16 层：   q/k/v/o + gate/up/down
VLM / Expert：           QK + PV
```

因此本实验的 **`VLM prefill` 严格含义是「VLM text-Transformer 被量化包装部分的 sparsity」**，**不包括**：

- SigLIP vision encoder
- vision connector
- 其它未被包装的非 Transformer 算子

> **Scope：** 当前量化框架覆盖的 VLM text Transformer + Action Expert Transformer 的 Linear/QK/PV；vision encoder、connector 及未包装的非 Transformer 算子不计入本 quick scan。
>
> 不可外推描述为「整个 SmolVLA prefill 的所有计算 sparsity」——vision encoder 的 FLOPs 占比很大，这类措辞会严重误导。

### 4.4 Primary headline 的定义

Primary 表的 `runtime element/bit sparsity` 是把某 stage 下所有已 instrumentation 的 quantized tensor observations（Linear activation/output + QK A/B/O + PV A/B/O）按 element/bit denominator 加权求和后的结果。

它**不等于**：

- MAC/FLOP-weighted sparsity
- unique activation-memory sparsity
- hardware speedup

原因：同一份语义数据可能先作为某个 operator 的 output、再作为下一个 operator 的 input 被重复计入不同 role。相关脚注需随结果表一并给出。

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

# 一键 4 组串行（不做设备隔离；启动时打印 GPU 占用供人工确认）
# 自动执行「阶段 1 校准（仅 Q0/Q1）→ 阶段 2 eval + 采集」
bash experiments/2026-09-15_sparsity-ratio-quickscan/scripts/run_quickscan.sh

# 单组（Q0/Q1 需先校准，再 eval；见下方两阶段说明）
#   阶段 1（只生成 scale，不导出 sparsity）
python main.py --config experiments/2026-09-15_sparsity-ratio-quickscan/configs/q0_int8.yaml --skip-evaluation
#   阶段 2（只 eval；必须跳过校准）
python main.py --config experiments/2026-09-15_sparsity-ratio-quickscan/configs/q0_int8.yaml --skip-calibration

# Q2/Q3 只有阶段 2（scale 为本地只读副本）
python main.py --config experiments/2026-09-15_sparsity-ratio-quickscan/configs/q2_fp8w4_po2.yaml --skip-calibration

# 汇总（Primary + Secondary CSV）
python experiments/2026-09-15_sparsity-ratio-quickscan/scripts/summarize_quick_sparsity.py

# 绘图（四配置 × runtime/weights × prefill/denoise = 4 张图）
python experiments/2026-09-15_sparsity-ratio-quickscan/scripts/plot_bit_sparsity.py
```

> **为何 Q0/Q1 必须拆两阶段**：`src/vla_tcs2/calibration.py:102` 的 `calibrate()` 会无条件 `QuantStatManager(scale_dir)` 新建实例并覆写到每个 quantized module，而新实例的 `sparsity_enabled` 默认 `False`（`main.py` 只对 `wrapper.stat_manager` 调过 `enable_sparsity(True)`）⇒「同一次 run 既校准又统计」时 **runtime sparsity 静默全空**（`module_sparsity.csv` 只有表头、无任何告警、`[DONE]` 照常打印）。详见 `logs.md` §4.9。本实验不改 core code（§8），故以两阶段绕开。

---

## 7. 输出目录映射

| config | 输出目录（outputs/ 下） | 状态 |
|---|---|---|
| `q0_int8.yaml` | `outputs/2026-09-15_sparsity-ratio-quickscan/q0_int8/` | ✅ done |
| `q1_int16.yaml` | `outputs/2026-09-15_sparsity-ratio-quickscan/q1_int16/` | ✅ done |
| `q2_fp8w4_po2.yaml` | `outputs/2026-09-15_sparsity-ratio-quickscan/q2_fp8w4_po2/` | ✅ done |
| `q3_fp8_po2.yaml` | `outputs/2026-09-15_sparsity-ratio-quickscan/q3_fp8_po2/` | ✅ done |
| 汇总（Primary） | `outputs/2026-09-15_sparsity-ratio-quickscan/quick_sparsity_summary.csv` | ✅ 8 行 |
| 汇总（Secondary） | `outputs/2026-09-15_sparsity-ratio-quickscan/quick_sparsity_by_role.csv` | ✅ 120 行 |
| 图（4 张） | `experiments/2026-09-15_sparsity-ratio-quickscan/docs/figures/` | ✅ done |

单组产物（README §11 Gate）：`sparsity/module_sparsity.csv`（3520 行）、`sparsity/weight_sparsity_static.csv`（224 行）、`sparsity/quantization_manifest.csv`、`sparsity/outlier_sidepath.csv`。

> `outputs/` 与 `scales/` 均被 `.gitignore` 排除，故上述 CSV/scale 不入库；入库的是生成它们的脚本与文档。

---

## 8. 风险与注意事项

- **不修改 core code**。`main.py`、`src/vla_tcs2/quant/stat_manager.py`、`quant_methods.py`、`model_wrapper.py` 一律不动。原因：Phase H 的 H3 runner **逐 task 启动新 Python 进程**，若中途改 core code，后续 task 会载入不同代码版本，破坏 H3 一致性。本实验只新增 configs / scripts / docs。
- **GPU 单卡约束，不做设备隔离**。本机仅 1 张 H100 NVL（GPU 0）。runner **不设** `CUDA_VISIBLE_DEVICES`，也不改 `MUJOCO_EGL_DEVICE_ID`，完全沿用 Phase F/G/H 的约定（依赖 conda env 的 `MUJOCO_EGL_DEVICE_ID=2`）。
  - 原因：robosuite 断言 `MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`（子串判断），而 `smolvla_eval` 的 conda env config vars 会把 `MUJOCO_EGL_DEVICE_ID=2` 写回子进程、覆盖 shell export。一旦限制 `CUDA_VISIBLE_DEVICES=0` 就会 `AssertionError`。
  - 详见 `logs.md` §4.8。启动时会打印 GPU 占用与同时刻其它任务，供人工确认。
  - **注意**：交互式 bash 会话的环境变量跨命令持久，调试时若曾 `export CUDA_VISIBLE_DEVICES`，务必先 `unset` 再启动，否则会污染 runner。
- **Q2/Q3 必须 `--skip-calibration`**。指向的本地副本虽不影响 Phase H，但重校准会改变数值、使「复用 G1-A/G1-B」的语义失效。runner 已硬编码该 flag。
- **GPU 显存**。`chunk_size=4194304` 已上调以减少统计开销；若 OOM 降回 `1048576`。
- **`1/(1-S)` 不是硬件加速比**。本实验不输出 speedup 结论。
- **`outlier_accounting` 是无操作键**。Phase H 的 yaml 里有该段，但代码从未读取它；native 列无条件导出。故本实验省略该段，不影响结果。
- **`d_bit: 4` / `p: 4` 不是精度**。二者是统计收集用的硬件建模元数据（digit size / parallelism），不改量化。
- **1 episode 的统计噪声**。单 episode 的 sparsity 不能代表全 suite 分布；本实验定位为 quickscan，如需收敛值应扩 episode（对照 Phase H：H1 1ep → H2 3ep 时 sparsity estimate 几乎不变）。
