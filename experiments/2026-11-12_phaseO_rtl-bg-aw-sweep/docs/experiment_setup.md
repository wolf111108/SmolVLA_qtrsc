# Phase O — RTL 对齐的 activation / 8-bit 补齐 weight BG 稀疏度

## 目的

新增独立 Phase O，不覆盖 Phase N 的原码幅值统计或历史输出。在同一次 SmolVLA Goal task 0 × 1 episode 中采集 activation，并只扫描一次静态 Linear weight。配对 pa×pw=2048：

| pa（token） | pw（输出通道） |
|---:|---:|
| 2 | 1024 |
| 4 | 512 |
| 8 | 256 |
| 16 | 128 |
| 32 | 64 |
| 64 | 32 |

activation 保持 2-bit BG。weight 同时分析 2-bit/4-bit BG，VLM 的 INT8 与 Expert 的 INT4 均按 8-bit 表示计数；INT4 的扩展高位参与分子和分母。

## 环境与版本

- 软件基线：`feat/hardware-timeloop-mapper@2267fceb15e59fb19c10d060b419a9e761539c5b`。
- 检测规则依据 `chipds` 分支 `docs/bsc-dsp-v2-spec` 的 `rtl_v2/bsc/bsc_zero_mask.v` 与 `rtl/dsp/bg_dsp_pack_2x2.sv`：负数取反，负数最低组加一；同一 K 上所有有效 lane 都为零才跳过。
- 统计入口：`src/vla_tcs2/stats/rtl_bit_group.py`。共享运行入口：`src/vla_tcs2/stats/bg_experiment.py`。共享 scale/coverage 合同：`src/vla_tcs2/stats/goal_protocol.py`。
- Phase N 保留原配置和 `BitGroupCollector` 的幅值语义；其 runner 改为薄入口，共用模型构造、校准和评测。Phase L 的 helper 名称保持可导入，抽取到 `goal_protocol`，量化前向源码未改动。
- 正式运行记录代码 SHA256、Git commit、配置哈希、checkpoint revision、864 个尺度文件哈希及统计协议。

## 模型与数据

沿用 Phase L `a8w4_expert`：`lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`，VLM Linear A8W8O8，Expert Linear A8W4O8，VLM/Expert QK/PV A8B8O8。Vision、connector、action/state/time heads 仍为原浮点；`outlier_ratio=0.01` 浮点保护路径保留。

校准沿用 `HuggingFaceVLA/libero@v3.0` 的 8 demonstrations、frame stride=4、seed=42、batch=1。默认复用完成且协议/源码/尺度哈希均匹配的 Phase L 校准；无匹配缓存则独立校准。复用检查失败时明确报错，不静默接受不匹配尺度。

## 评测协议

只跑 `libero_goal` task 0、1 episode、seed=1000、batch=1、单 worker，不录视频。`n_action_steps=10`、`num_steps=10` 保持不变。六组 pa/pw 同时统计，不重复跑六遍任务。一个 episode 的成功与否仅检查运行，不作为可靠任务成功率结论。

要求 288 个站点（224 Linear + 64 QK/PV），VLM prefill 与 Expert 的全部十个 denoise step 有实际输入记录。静态权重恰有 224 个物理 Linear 站点，每个只统计一次；不随 Expert 十次 denoise 重复累计。完整权重编码由生产 `quant_awo` 及实际校准的 `w_interval` 计算。

## 实验变量与分组

### 1. 精确符号规则

令 `q8` 为符号扩展后的 8-bit 补码，`s=(q<0)`，BG 宽度为 b，最低组 g=0：

```text
raw_g = (q8 >> (b*g)) & (2**b - 1)
normalized_g = raw_g XOR (s ? 2**b - 1 : 0)
D_g = normalized_g + (s AND g==0)
nonzero_g = (normalized_g != 0) OR (s AND g==0)
skip_pack = NOT OR_valid_lanes(nonzero_g)
```

只对负数 BG0 加一；BG2 的 D0 可为 4，BG4 的 D0 可为 16，不把溢出进位传播到相邻 BG。不调用 `abs(q)` 代替该分解。原始 q 可由 `sign_factor * sum(D_g*2**(b*g))` 精确恢复。

### 2. INT4 符号扩展到 8 位

INT4 输入按真实 signed 编码 `[-8,7]`，不是 `[0,15]` 的 nibble。负数高四位补 1，非负数高四位补 0，再执行上述符号归一化。例如：

```text
q=-8: INT4 1000 → INT8 11111000
取反:                  00000111
BG2 有效 D（低→高）: [4, 1, 0, 0]
BG4 有效 D（低→高）: [8, 0]
```

因此每个 INT4 weight 的 BG2 level 2/3 与 BG4 level 1 恒为稀疏。每个 weight 的分母为 4 个 BG2 或 2 个 BG4，包括扩展高位；绝不按原 4-bit 分母重新归一化。`sign_extended_groups` 单独标明扩展位组数量。尾部不存在的输出通道不属于“位宽补齐”，不纳入 native 分母。

### 3. pa / pw 的分组方向

- Linear activation `[batch,tokens,K]` 沿 tokens 以 pa 分组；固定 K 和 BG level。
- QK/PV 左输入 A 沿 query tokens 以 pa 分组。`bsc_left` 汇总 Linear activation + QK/PV A。
- QK B 为 Kᵀ，额外 `all_inputs` 统计沿其最后一维 token；PV B 沿倒数第二维 token。它们属于 runtime activation，不混入静态 weight。
- Linear weight 的 PyTorch 布局为 `[out_features,in_features]`，沿 out_features（N）以 pw 分组，固定输入 K。等价于 GEMM 的 W[K,N] 沿 N 分组。不得沿 K 打包。
- 不跨 batch、head、layer、调用或 denoise step 拼接。

### 4. 分母与指标

所有跨层/组件汇总先累加整数计数再相除，不平均百分比。VLM 和 Expert 分开汇总；activation 的运行期访问次数与 weight 的一次性静态计数保持独立。

| 字段 | 含义 |
|---|---|
| `native_groups` | 真实普通路径元素对应的位组数；weight 包含位宽扩展 BG |
| `native_zero_groups` | 单元素位组为零，尚未考虑 pa/pw 共享 |
| `native_skipped_groups` | 整个 pack 可跳过时覆盖的真实普通路径 BG 数 |
| `native_skip_rate` | 主表指标，native_skipped_groups/native_groups |
| `individual_bg_zero_rate` | 单元素 BG 零率，用于解释共享粒度损失 |
| `normal_zero_packs / total_packs` | 普通路径 pack 零率，含尾包、受 FP mask 置零影响 |
| `normal_issued_packs` | 普通路径非零包数；不是整机周期 |
| `full_native_pack_zero_rate` | 只对恰含 p 个真实且全部未保护 lane 的包计数；分母为零输出 N/A/CSV 空值 |
| `tail_pack_fraction` | 不足 pa/pw 的尾包占比 |
| `sign_extended_group_fraction` | 补到 8 位所引入的 BG 占比；INT4 为 1/2 |

activation 复用实际前向的量化编码及 outlier mask，排除被浮点保护的 lane，不将其人工置零解释为天然稀疏。静态 weight 采用完整权重量化编码，不施加与 activation 相关的动态 outlier mask；对应 `full_weight_quantized_no_dynamic_outlier_mask`。该表描述静态权重的编码机会，不代表逐次前向实际 normal-path weight 流量。

`calls` 等辅助计数在每个 p/level 上重复保存；跨 level 的 all 行是 BG 加权计数，不能据此重新推算去重后的物理层数或元素数。物理权重元素数以 `weight_manifest.json` 为准。

## 运行命令

仓库根目录、既有 `smolvla_eval` 环境：

```bash
conda activate smolvla_eval
GPU=0 bash experiments/2026-11-12_phaseO_rtl-bg-aw-sweep/scripts/run.sh
```

CPU 配置检查、依赖检查及回归：

```bash
bash experiments/2026-11-12_phaseO_rtl-bg-aw-sweep/scripts/run.sh --preflight
bash experiments/2026-11-12_phaseO_rtl-bg-aw-sweep/scripts/run.sh --check-env
python tests/test_rtl_bit_group_sparsity.py
python tests/test_bit_group_sparsity.py
python experiments/2026-11-09_phaseL_int8-goal/scripts/test_experiment.py
```

指定已完成的 Phase L 根目录，或强制 fresh calibration：

```bash
bash experiments/2026-11-12_phaseO_rtl-bg-aw-sweep/scripts/run.sh \
  --reuse-from /absolute/path/to/outputs/2026-11-09_phaseL_int8-goal
bash experiments/2026-11-12_phaseO_rtl-bg-aw-sweep/scripts/run.sh \
  --fresh-calibration --output-dir outputs/2026-11-12_phaseO_rtl-bg-aw-sweep-fresh
```

已有输出目录拒绝覆盖。可用 `PYTHON=/path/to/python` 指定解释器。

## 输出目录映射

默认 `outputs/2026-11-12_phaseO_rtl-bg-aw-sweep/`：

| 文件 | 内容 |
|---|---|
| `bit_groups/summary.md` | 六组 pa/pw × VLM/Expert A-BG2/W-BG2/W-BG4 汇总表 |
| `bit_groups/paired_summary.csv` | 六行配对总表，保留原始计数、尾包率和严格 full-pack 比率 |
| `bit_groups/summary.csv` | 组件 × scope × BG 宽度 × p × level（含 all） |
| `bit_groups/activation_by_site.csv` | module/phase/flow_step/role/pa/level 明细 |
| `bit_groups/weight_by_site.csv` | module/source_bits/BG 宽度/pw/level 明细 |
| `bit_groups/weight_manifest.json` | 每层静态 shape、原精度、8-bit 表示、编码 SHA256、scale |
| `coverage.json` | 实际 activation 站点/step/shape、224 权重站点覆盖 |
| `config.yaml`, `execution.json`, `scale_audit.json` | 实际配置、源码与尺度可追溯信息 |
| `completed.json` | 评测、覆盖与全部输出校验通过后才写 PASS |

## 风险与注意

当前 RTL 实现 activation BG2 的共享 K 跳过；weight BG2/BG4 和超过当前 RTL profile 的 pa/pw 是同规则下的分析扩展，不代表硬件已经支持这些配置。不能将 activation 与 weight 的两个边际稀疏率相乘当作联合跳过率，也不能把 issued packs 当作端到端运行周期。

一次 task 只用于快速探索；小于 pa/pw 的真实维度产生尾包，其全长包指标应为 N/A。位宽扩展的高位则始终计入稀疏度分母，两种“补齐”必须区分。
