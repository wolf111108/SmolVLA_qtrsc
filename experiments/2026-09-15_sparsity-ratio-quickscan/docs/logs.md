# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**（区别于 `results.md` 的结果记录）。
> 每次启动/完成一次运行、发现或修复异常时追加一条；章节为固定结构，不可删除；无内容写「无」。
> 运行日志按时间**正序**追加，最新状态反映在「当前状态」与头部字段。

- **实验名称**：2026-09-15_sparsity-ratio-quickscan
- **状态**：done（draft / running / done / aborted）
- **最后更新**：2026-09-19

---

## 1. 当前状态

**实验已完成（done）。** Q0–Q3 四组 sparsity 数据全部有效（Q0/Q1 经两阶段重跑，2026-09-16 完成），两份汇总 CSV 已生成，结果回填 results.md，四配置柱状图已出（runtime/weights × prefill/denoise 共 4 张，均已过越界校验）。

**静态校验已完成**：`bash -n` 通过、`py_compile` 通过、4 个 YAML 可解析、汇总脚本用 Phase H 真实产物验证过聚合正确性（见 §4.3）；Gate 退出码三场景已实测（§4.5）；runner 幂等逻辑已实测（§4.6）。2026-09-15 已按 `er.md` 审计报告修复 3 个 P1（§4.7）。

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| 骨架 + 文档 | ✅ done | 2026-09-15 | configs / scripts / docs |
| scale 准备 | ✅ done | 2026-09-15 | Q2/Q3 本地只读副本 + sha256 存证（864/864 bit-identical） |
| 静态校验 | ✅ done | 2026-09-15 | bash -n / py_compile / YAML / 汇总脚本真值验证 |
| er.md 审计修复 | ✅ done | 2026-09-15 | 3 个 P1 + 3 项规范补充（见 §4.7） |
| Q0 INT8 | ✅ done | 2026-09-15 22:09 | 两阶段重跑后有效 |
| Q1 INT16 | ✅ done | 2026-09-16 00:16 | 两阶段重跑后有效 |
| Q2 FP8(A/O)-W4 | ✅ done | 2026-09-15 18:43 | 有效 |
| Q3 FP8 PoT | ✅ done | 2026-09-15 18:50 | 有效 |
| 汇总 | ✅ done | 2026-09-16 | 4/4 组有效 + 2 份 CSV |
| 绘图（四配置） | ✅ done | 2026-09-19 | 4 张（runtime/weights × prefill/denoise）；越界检测 0 |

## 2. 运行日志

| 日期 | 阶段 | 命令 / 配置 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-09-15 | 建骨架 | `mkdir` + `_template/docs/*.md` 替换 | ✅ | — | 目录名按 `README_EXPERIMENT.md` §7 取 `2026-09-15_sparsity-ratio-quickscan` |
| 2026-09-15 | 写 config | `configs/q0..q3` | ✅ | — | Q2/Q3 量化块逐字复用 G1-B/G1-A |
| 2026-09-15 | scale 准备 | `cp` G1-A/G1-B → 本实验目录 + sha256 校验 | ✅ | `scales/2026-09-15_sparsity-ratio-quickscan/` | 各 864 文件，bit-identical |
| 2026-09-15 | 写脚本 | `run_quickscan.sh` / `summarize_quick_sparsity.py` | ✅ | — | 语法与空跑校验通过 |
| 2026-09-15 | 校验 | 用 Phase H H2-S0 task00 产物验证聚合 | ✅ | `/tmp/qs_validate`（临时） | Expert weight_bit=42.58%，与 Phase H 记录一致 |
| 2026-09-15 | 审计修复 | 按 `er.md` 自查并修复 3 P1 | ✅ | — | 见 §4.7 |
| 2026-09-15 | 验证 | Gate 退出码三场景 + 幂等逻辑实测 | ✅ | `/tmp/qs{2,3,4}`（临时） | 见 §4.5 / §4.6 |

## 3. 后台任务

| PID | 阶段 | 启动时间 | 状态 | 备注 |
|---|---|---|---|---|
| — | Q0–Q3 全部 | 2026-09-15 18:43 → 2026-09-16 00:16 | ✅ 已结束 | Q2 18:43 / Q3 18:50 / Q0 22:09 / Q1 00:16（Q0/Q1 为两阶段重跑后的有效时刻） |

### 3.1 GPU 占用情况（2026-09-15 18:0x，启动时的历史快照）

| GPU | 型号 | 占用者 | 说明 |
|---|---|---|---|
| 0 | NVIDIA H100 NVL | Phase H `run_h3_100ep.sh`（s0 → s1，各 10 task × 10ep） | **本实验唯一可用卡** |

> `nvidia-smi -L` 仅列出 GPU 0。README §9 要求「GPU 请选择未被 H3 占用的 GPU」，但本机无第二张卡可选。两种选择：
> 1. **等 H3 跑完再跑本实验**（最干净；H3 预计 2026-09-15 晚间至 09-16 凌晨结束）；
> 2. 与 H3 **共享 GPU 0**（显存充裕：8949 MiB / 95830 MiB，但算力互稀释，单 episode 与校准都会变慢）。
>
> 本实验是 quickscan（预算 30–50 min），共享代价可接受，但需意识到 H3 也会被拖慢。**建议优先方案 1。**

## 4. 异常与处理

### 4.1 脚手架目录名不符合 README 要求

- **现象**：`new_experiment.sh 2026-09-15 sparsity-ratio quickscan` 生成 `2026-09-15_sparsity-ratio_quickscan`；换成 `sparsity ratio-quickscan` 又得到 `2026-09-15_sparsity_ratio-quickscan`。
- **根因**：脚手架固定用 `${DATE}_${PHASE}_${TOPIC}` 拼接，段间分隔符恒为 `_`，无法产出 `README_EXPERIMENT.md` §7 要求的 `2026-09-15_sparsity-ratio-quickscan`（后缀全用连字符）。
- **处理**：按 README 的字面路径**手工建骨架**（`configs/ scripts/ docs/figures/ tasks/`），并用 `sed` 从 `experiments/_template/docs/` 生成三份文档，保证章节结构仍与规范一致。
- **待办**：规范与 README 的命名冲突需要一次性裁决（是放宽脚手架以支持连字符 topic，还是统一改用下划线）。本实验按 README 执行。

### 4.2 Q2/Q3 复用 scale 的覆盖风险

- **风险**：README §4 让 Q2/Q3 的 `scale_dir` 直指 `scales/2026-09-10_phaseG_w4-root-cause/component-localization/{g1a_fp8_all,g1b_w4_all}`。这两个目录**正被 Phase H 的 H1/H2/H3 复用**（Phase H setup §2 明文禁止覆盖，且维护 hash manifest）。若有人不加 `--skip-calibration` 跑 Q2/Q3，框架会原地重写这些 scale。
- **处理**（三重防护）：
  1. 把 scale **复制**到 `scales/2026-09-15_sparsity-ratio-quickscan/{q2_fp8w4_po2,q3_fp8_po2}`，config 的 `scale_dir` 指向副本；
  2. 副本与原始逐文件 sha256 校验 **864/864 bit-identical**，统计等价性成立；
  3. `run_quickscan.sh` 对 Q2/Q3 **硬编码 `--skip-calibration`**，破坏路径不可达。
- **校验记录**：`docs/manifest/{q2_fp8w4_po2,q3_fp8_po2}_scale_hashes.txt`。
- **附注**：原始目录 mtime 保持 2026-09-11（未被触碰）。

### 4.3 汇总脚本的真值验证

为避免「先跑完才发现聚合写错」，用 **Phase H H2-S0 task00** 的既有产物（其量化配置与 Q3 相同：FP8 PoT + G1-A scale）做了端到端验证：

```text
[FP8 PoT] q3_fp8_po2
  [OK] phase 覆盖 prefill+denoise
  [OK] denoise flow_step 覆盖 0..9
  [OK] component 覆盖 vlm+expert
  [OK] tensor_role 覆盖 Linear activation/output + MatMul A/B/O
  VLM prefill      elem=  1.92%  bit= 41.69%  | w_elem=  0.00%  w_bit= 40.86%
  Expert denoise   elem=  3.03%  bit= 42.15%  | w_elem=  0.00%  w_bit= 42.58%
```

**`Expert denoise weight_bit = 42.58%` 与 Phase H `results.md` 记录的「S0 Expert FP8 42.58%」完全一致**，说明本脚本的 sum-then-divide 聚合与 Phase H 独立实现（`build_results_tables.py`）口径相同。Gate 校验四项亦全部通过。

### 4.4 已知的生成期瑕疵（已修）

- `q1_int16.yaml` 的 `batch_size` 曾被批量替换误改为 16（正则 `": 8\n" -> ": 16\n"` 连带命中），已恢复为 README §5 要求的 8。
- `q2/q3` 的 `calibration` 段原为 Phase G 的 `episodes: 8 / frame_stride: 4`，已统一改为 quick 值 `1 / 8 / 16`，使误触发的重校准代价最小。

### 4.5 Gate 退出码实测（修复后）

原先只有「缺 CSV」会置 `all_fail`，因此屏幕上出现 `[FAIL] native counters 出现负值` 时进程仍会 `exit 0`，runner 会误报 `Quick scan finished.`。修复为「任一 FAIL 即整体失败」，并把 flow_step / tensor_role 两项由 WARN 提升为 FAIL（本实验 contract 要求必须齐备）。

| 场景 | 结果 |
|---|---|
| A：仅 1/4 组有产物 | `exit 1`，3 条 FAIL ✅ |
| B：4/4 组齐备 | `exit 0`，0 条 FAIL ✅ |
| C：注入 `zero_elements_native=-5` | `exit 1`，`[FAIL] native counters 出现负值（1 行）` ✅（修复前为 exit 0） |

### 4.6 runner 幂等逻辑实测（修复后）

原先只检查 `module_sparsity.csv`。但 `main.py` 的 export 顺序是 `module_sparsity.csv → … → weight_sparsity_static.csv`，「module 已写、后续 export 崩溃」会被误判为已完成而永久跳过。改为**同时要求两个 primary 产物**：

```text
仅有 module_sparsity.csv            → 不 SKIP ✅
module + weight_sparsity_static 齐备 → SKIP   ✅
```

### 4.7 er.md 审计自查结果（2026-09-15）

`er.md` 审计结论：四组配置本身**无 P0 级数值错误，可以开跑**；期望 3 个 P1 修复。逐项核实与处理：

| 审计项 | 审计判定 | 自查结论 | 处理 |
|---|---|---|---|
| Q0/Q1 INT8/INT16 config | ✅ | 核对无误（Q1 位宽 16，`parse_quant_spec` 走 `kind="int"`） | 无需改 |
| Q2 FP8(A/O)-W4 + PoT(A/O) | ✅ | `pot_ao_outlier` 仅 A/O 为 PoT，W4 scale continuous | 无需改 |
| Q3 FP8 PoT | ✅ | 全 E4M3 + `pot_fp8_outlier`（A/W/O 均校验 PoT） | 无需改 |
| Q2/Q3 scale 副本策略 | ✅ 很好 | 864 = 224×3 + 64×3，副本 bit-identical | 无需改 |
| Linear/MatMul output 旧 `mul_` bug | ✅ | 远端已是 `.mul()`，无原地覆盖 | 无需改 |
| INT bit / FP8 significand 统计 | ✅ | — | 无需改 |
| **`reported` 方向写反** | ❌ P1 | **确认写反**。protected 位置真值非零而 normal path 存 0 ⇒ `S_reported > S_native`，是**高估** | ✅ 已改为「高估」（`experiment_setup.md` §4.1 + 脚本 docstring 两处） |
| **Secondary QK/PV 分不开** | ❌ P1 | **确认属实**。原按 `(stage, tensor_role)` 聚合 ⇒ QK A 与 PV A 合并 | ✅ 改为 `(stage, operator, tensor_role)` + `_role_label()`，输出 `Linear:*` / `QK:*` / `PV:*`；实测表行数 24 → **30**，两 stage 均出现独立 `QK:A/B/O` 与 `PV:A/B/O` |
| **Gate 退出语义** | ⚠️ P1 | **确认属实** | ✅ 已修（§4.5）+ flow_step/tensor_role 提为 FAIL |
| runner 部分运行误判 | ⚠️ P1 | **确认属实** | ✅ 已修（§4.6） |
| 统计范围命名 | ⚠️ P1 | 确认需明示；框架只包装 VLM text Transformer + Action Expert | ✅ `experiment_setup.md` 新增 §4.3（含 Scope 句）+ `results.md` §2 表头加 Scope 注 |
| Primary headline 定义 | ⚠️ | 需说明是 observed-tensor-element-weighted | ✅ `experiment_setup.md` §4.4 + `results.md` §2 脚注 |
| `README_EXPERIMENT.md` 404 | ⚠️ P2 | **文件实际在仓库根**（`/home/zyzhao/VLA_tcs2/README_EXPERIMENT.md`，已随 `addaed0` 推送）；审计方查的是 `experiments/.../README_EXPERIMENT.md` | ✅ 所有引用补「（仓库根）」消除路径歧义 |

**未采纳的建议**：无。审计提出的 3 个 P1 与 3 项规范补充全部采纳。

**Primary 表未受影响**：修复前后 Primary 数值完全一致（1.92/41.69/40.86、3.03/42.15/42.58），确认 QK/PV 分组 bug 只影响 secondary 表。

### 4.8 首次启动即挂：EGL 设备索引与 CUDA 可见性的耦合（2026-09-15 18:19——18:28）

**现象**：Q0 校准成功后，在创建 LIBERO env 时报：

```text
AssertionError: MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id
specified in CUDA_VISIBLE_DEVICES
```

**根因（两层）**：

1. robosuite 有硬断言（`site-packages/robosuite/utils/binding_utils.py:35`）：
   ```python
   CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
   if CUDA_VISIBLE_DEVICES != "":
       MUJOCO_EGL_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
       if MUJOCO_EGL_DEVICE_ID is not None:
           assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
               MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES   # ← **子串**判断
           ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of ..."
   ```
   Phase F/G/H 的脚本只设 `MUJOCO_EGL_DEVICE_ID=2` 而**不设** `CUDA_VISIBLE_DEVICES`，
   所以这句断言被整段跳过，从未暴露。
2. `smolvla_eval` 的 **conda env config vars** 里固定了：
   ```text
   MUJOCO_GL = egl
   PYOPENGL_PLATFORM = egl
   MUJOCO_EGL_DEVICE_ID = 2
   ```
   `conda run` 会把这些值**写回**子进程环境，覆盖 shell 里的 `export`。实测：
   | 方式 | 子进程看到的 EGL |
   |---|---|
   | shell `export MUJOCO_EGL_DEVICE_ID=0` | **2**（被 conda 覆盖） |
   | `conda run ... env MUJOCO_EGL_DEVICE_ID=0 python` | 0（`env` 前缀在 conda 之后生效） |

   两者叠加：我最初加的 `GPU=` 守卫生成了 `CUDA_VISIBLE_DEVICES=0`，而 conda 仍注入 `EGL=2`，
   `"2" in "0"` 为假 ⇒ 断言失败。

**曾尝试但未采用的修法**：用 `env MUJOCO_EGL_DEVICE_ID=$CUDA_FIRST` 前缀强制两者一致。断言确实能过，
但有个**更隐蔽的问题**：`CUDA_VISIBLE_DEVICES` 只重映射 CUDA runtime 的可见设备，
**不影响 MuJoCo/EGL 自己的设备枚举**。因此「CUDA 索引 0」与「EGL 设备索引 0」未必是同一张物理卡；
在一个只有 1 张卡的环境里看不出问题，但它会为未来多卡场景埋下「静默渲染到别的卡」的隐患。
故舍弃。

**采纳的修法**：**彻底不做设备隔离**——runner 不再设 `CUDA_VISIBLE_DEVICES`，也不改
`MUJOCO_EGL_DEVICE_ID`，完全沿用 Phase F/G/H 的既定约定（靠 conda env 的 `EGL=2`）。
改为在启动时打印 GPU 占用与同时刻在跑的本仓库任务，供人工确认。
本机只有 1 张 H100 NVL，`GPU=` 参数本身不提供任何真实隔离能力。

**附带坑（同一次排查中踩到）**：`run_in_terminal` 的 bash 会话**跨命令持久**。
我前面调试时在会话里 `export CUDA_VISIBLE_DEVICES=0`，后续 `nohup` 启动的 runner 就**继承了这个污染值**，
导致「已经改好的脚本」仍然报同一个断言。排查此值时务必要 `unset` 后再重试，
或明确以 `env -u CUDA_VISIBLE_DEVICES` 启动。

**验证**：修复后 Q0 顺利走到 `Built vec env | suite=libero_goal | task_id=0` 与 rollout 进度条。

### 4.9 【框架 bug】calibration 会静默清空 runtime sparsity 收集（2026-09-15 首轮结果作废）

**这是本次实验最重要的发现，属于框架级问题，非本实验 config 的问题。**

**现象**：首轮 4 组全部 `[DONE]`，但汇总脚本对 Q0/Q1 报「缺少 module_sparsity.csv」——而文件其实存在。核查发现：

| 组 | module_sparsity.csv | workload rows | outlier_sidepath.csv | 判定 |
|---|---|---|---|---|
| Q0 INT8 | **494 B（仅表头）** | **0** | 130 B（空） | ❌ runtime 全空 |
| Q1 INT16 | **494 B（仅表头）** | **0** | 130 B（空） | ❌ runtime 全空 |
| Q2 FP8W4 | ~1.0 MB（3520 行） | 3520 | 434 KB | ✅ 正常 |
| Q3 FP8 | ~1.0 MB（3520 行） | 3520 | 434 KB | ✅ 正常 |

注意 weight 的静态统计（`weight_sparsity_static.csv`）在四组都正常（~49 KB），**只有 runtime 收集失效**。

**根因**：`src/vla_tcs2/calibration.py:102`

```python
# -------------------------------------------------------------------------
# 1. Bind a stat manager to every quantized module
# -------------------------------------------------------------------------
stat_manager = QuantStatManager(str(scale_dir))     # ← 无条件 new 一个新实例

for module in model.modules():
    if isinstance(module, (QuantizedLinear, QuantizedMatMul)):
        module._stat_manager = stat_manager          # ← 覆盖掉 wrapper 注入的实例
```

而 `main.py` 的 STEP 2.5 只对 **`wrapper.stat_manager`** 调用了 `enable_sparsity(True)`：

```python
sp_cfg = config.get("sparsity", {})
if sp_cfg.get("enabled", False):
    sm = wrapper.stat_manager          # ← 被 enable 的是这个
    sm.enable_sparsity(enable=True, ...)
```

`calibrate()` **创建的新 `QuantStatManager` 的 `sparsity_enabled` 仍是默认 `False`**。于是校准之后，每个 module 的 forward 都走新实例，而 `collect_quant_tensor` 的第一道 gate：

```python
if not self.sparsity_enabled:
    return
```

把**所有** runtime 记录静默丢弃。因为 `main.py` 的 calibration 是 if/else 分支：

```python
if not args.skip_calibration:
    calibrate(...)          # ← 替换 _stat_manager
else:
    print("Calibration skipped. Existing scale files will be reused.")
```

所以 Q2/Q3（`--skip-calibration`）**根本不调用 `calibrate()`**，`_stat_manager` 保持为被 enable 过的 wrapper 实例，收集正常。

**为什么 Phase H 没暴露**：H1/H2/H3 全部用 `--skip-calibration`（复用 Phase G 的 scale），从未在「同一次 run 里既校准又统计」。

**危险等级：高**。失败形态完全静默——无异常、无警告、无日志提示，`[DONE]` 照常打印，CSV 正常生成（只有表头），只有把 CSV 打开看行数才会发现。

**本实验的绕过（不改 core code，因 H3 正在运行，项目规范禁止此时改动 core）**：
把「生成 scale」与「采集 sparsity」拆成两个 run：

```text
阶段 1  python main.py --config <Q0|Q1> --skip-evaluation
        → 只校准、生成 scale；不进入 STEP 3、不导出 sparsity
阶段 2  python main.py --config <Q0|Q1> --skip-calibration
        → 只 eval；module._stat_manager 保持 wrapper 实例 ⇒ 收集正常
```

`run_quickscan.sh` 已据此重构：新增 `NEEDS_CALIBRATION` 阶段（幂等，scale ≥ 864 时跳过），`RUNS` 四组统一 `--skip-calibration`；并加了「`module_sparsity.csv` 行数 > 1」的产物有效性判据，避免 header-only CSV 被幂等逻辑误判为完成。

**建议的框架层修复（待 H3 结束后实施，需评审）**：`calibrate()` 不应 new 新实例，至少应满足其一 ——
1. 复用调用方传入的 stat manager（新增可选参数），保留其 `sparsity_enabled` 状态；
2. 或在校准结束后把新实例的配置（含 `sparsity_enabled`）与 wrapper 的实例同步；
3. 或最保守：在 `calibrate()` 里检测 `stat_manager.sparsity_enabled` 并把该状态带到新实例。

**验证**：修复 runner 后重跑 Q0/Q1（scale 已存在 ⇒ 阶段 1 被正确跳过），Q0 于 20:24:36 以 `--skip-calibration` 启动。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-15 | 创建文档；记录骨架/config/scale/脚本就绪与静态校验结果 | |
| 2026-09-15 | 记录 §4.1 命名冲突、§4.2 scale 覆盖风险与三重防护、§4.3 用 Phase H 真值验证聚合、§4.4 生成期瑕疵修正 | |
| 2026-09-15 | 按 `er.md` 审计修复 3 P1（reported 措辞 / QK-PV 分组 / Gate 退出语义）+ runner 幂等 + scope/headline 定义；新增 §4.5-§4.7 | |
| 2026-09-15 | Q0-Q3 首次启动即撞 EGL 断言；定位并记录 §4.8（conda env 覆盖 EGL / CUDA 与 EGL 索引耦合 / 会话变量污染），改为不做设备隔离后成功启动 | |
| 2026-09-16 | 回填 Q0/Q1 两阶段重跑完成时刻（Q0 22:09 / Q1 00:16）；§4.9 记录首轮作废与框架 bug | |
| 2026-09-19 | **状态改 done**；补充 §1 状态表与 §3 后台任务收尾；绘图扩展为四配置（4 张）并记录越界校验为 0；results.md / experiment_setup.md 同步回填 | |
