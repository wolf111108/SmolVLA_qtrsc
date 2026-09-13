# 2026-09-13 Phase H — Accuracy-Preserving Quantization Sparsity Characterization

> 状态：draft / pre-implementation  
> 实验名称：`2026-09-13_phaseH_accuracy-preserving-sparsity`  
> 研究目标：在已经通过闭环 accuracy 验证的量化配置上，系统统计 element / bit / unit sparsity、outlier FP side-path 与 phase/flow-step 分布，为后续 VLA 硬件稀疏加速建模提供可信 workload。  
> 本文档遵循仓库 `experiments/README.md` 固定实验文档结构。

---

# 1. 目的

本实验不再搜索新的量化配置，而是固定已有 accuracy-preserving 配置，回答：

\[
\boxed{
\text{闭环精度可接受的量化配置，到底产生多少真正可利用的稀疏性？}
}
\]

核心问题：

1. FP8-all 与 Expert-W4 的 element zero ratio / bit sparsity 分别是多少？
2. sparsity 主要来自 VLM、Action Expert、QK 还是 PV？
3. prefill 与 denoise 的分布是否显著不同？
4. denoise flow step 0→9 是否存在系统性 sparsity 演化？
5. activation / weight / output 哪一侧最 sparse？
6. outlier protection 实际占多少 FP side-path？
7. W4 带来的 sparse-bit opportunity 能否在**不损失闭环 SR**的条件下成立？
8. unit/block sparsity 是否足够高，值得在 bit-serial/CIM 数据通路中做 coarse-grained skip？
9. 稀疏度是否在 10ep→30ep 后已经收敛，从而避免不必要的 100ep 统计成本？
10. 后续硬件模型应该使用哪一种 sparsity 定义，而不是简单的平均 zero ratio？

本实验不直接声称硬件 speedup。任何：

\[
1/(1-S)
\]

仅作为 ideal sparsity upper bound。

---

# 2. 环境与版本

正式运行前把以下 manifest 落盘：

```bash
git rev-parse HEAD
git status --short
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name())"
python -c "import mujoco; print(mujoco.__version__)"
python -c "import lerobot, inspect; print(getattr(lerobot, '__version__', 'unknown'))"
```

要求：

```text
conda env        = smolvla_eval
MuJoCo           = 3.3.2
EGL              = current validated setup
model            = lerobot/smolvla_libero
suite main       = libero_goal
seed             = 1000
```

额外保存：

```text
manifest/
  git_head.txt
  git_status.txt
  pip_freeze.txt
  nvidia_smi.txt
  env.txt
  scale_hashes.txt
```

### scale hash

对 S0/S1 复用的 calibration scale 文件做：

```bash
find <scale_dir> -type f -print0 | sort -z | xargs -0 sha256sum \
  > manifest/scale_hashes.txt
```

Phase H 期间禁止覆盖这些 scale。

---

# 3. 模型与数据

## 3.1 模型

```text
checkpoint = lerobot/smolvla_libero
```

保持与 Phase G 一致，不换 checkpoint。

## 3.2 数据

calibration 不重新执行。

如需仅检查配置：

```text
dataset = HuggingFaceVLA/libero@v3.0
```

但 Phase H 正式 run 应复用 Phase G 已存在 scale：

```bash
--skip-calibration
```

## 3.3 当前 accuracy-preserving 配置

### S0：G1-A FP8-all anchor

来源：

```text
experiments/2026-09-10_phaseG_w4-root-cause/
tasks/component-localization/configs/g1a_fp8_all.yaml
```

关键条件：

```text
Linear:
  VLM + Expert = FP8 E4M3
MatMul:
  QK/PV = FP8 E4M3
outlier_ratio = 0.01
n_action_steps = 10
num_steps = 10
Goal SR = 88%
```

角色：

> 高精度全量化 anchor。

### S1：G1-D Expert-W4

来源：

```text
.../g1d_w4_expert_only.yaml
```

关键条件：

```text
VLM Linear      = raw FP
Expert Linear W = INT4/W4
Expert Linear A/O = FP8
QK/PV MatMul    = FP8
outlier_ratio   = 0.01
n_action_steps  = 10
num_steps       = 10
Goal SR         = 84%
```

相对 S0：

```text
ΔSR = -4pp
```

在 Phase G 使用的经验噪声带内，因此本轮暂称：

> Goal 条件下 accuracy-preserving / within-noise configuration。

不能写成：

> 四 suite 全局无损。

### S2：G5 winner（可选）

只有当 G5-A/B/C 正式结果产生并满足预定义 accuracy gate 后，才加入 Phase H。

不要在结果未知时预先指定 S2。

---

# 4. 评测协议

## 4.1 为什么保持 n_action_steps=10

本实验要测的是：

> “已经验证保持精度的当前量化配置”的 sparsity。

因此必须复用 Phase G 的协议：

```text
n_action_steps = 10
num_steps      = 10
suite          = libero_goal
seed           = 1000
batch_size     = 1
max_parallel_tasks = 1
```

不要为了贴论文 Table 2 临时改回 `n_action_steps=1`，否则 accuracy anchor 与 sparsity run 不再是同一个系统。

如果以后要研究 Table-2 protocol 下 sparsity，应另建独立实验。

## 4.2 每 task 单独运行

为了保留 task-level sparsity，正式运行采用：

```text
task0
task1
...
task9
```

分别落盘。

优点：

- 不需要在核心 stats manager 中引入 task_id；
- task 成功率可直接与 task sparsity 对齐；
- 某个 task crash 不会污染全部统计；
- 聚合脚本可以严格从 numerator/denominator 汇总。

如果当前 `evaluation.env.task_ids: [i]` 可直接传入 LeRobot env config，则使用该方式；若当前版本不接受该字段，再在 `eval.py` 做最小显式透传，不修改 rollout 语义。

---

# 5. 实验变量与分组

整个实验只允许变化两个维度：

```text
config: S0 / S1 / optional S2
sample budget: 1ep/task / 3ep/task / 10ep/task
```

其他全部锁死。

## 5.1 H0 — correctness smoke

```text
S0 Goal task0 × 1 episode
S1 Goal task0 × 1 episode
```

目的：

- 验证统计代码正确；
- 不关注最终 sparsity 数值；
- Gate 全通过后才能进入 H1。

## 5.2 H1 — 10 episode pilot

```text
10 tasks × 1 episode / config
= 10 ep/config
```

目的：

- 得到第一版 task/layer/operator/flow-step sparsity 分布；
- 检查统计字段覆盖；
- 估计 collector overhead。

## 5.3 H2 — 30 episode convergence

```text
10 tasks × 3 episodes / config
= 30 ep/config
```

目的：

- 与 H1 比较统计收敛；
- 决定是否必须为了 sparsity 本身跑满 100ep。

## 5.4 H3 — Goal formal run

```text
10 tasks × 10 episodes / config
= 100 ep/config
```

目的：

- 同时重新确认 stats-on 条件下 SR；
- 形成最终 Goal 表与论文/组会主图。

### H3 accuracy gate

S0：

```text
预期围绕既有 88% 波动
```

S1：

```text
预期围绕既有 84% 波动
```

正式判定不要把 ±1~2pp 当成真实量化变化，应结合 Phase G 已使用的噪声带。

最重要的是：

> stats ON 不应改变 forward，因此若出现大幅 SR 漂移，先按代码 bug 处理，而不是解释为随机波动。

## 5.5 H4 — 四 suite 扩展（可选但推荐）

只有在需要对外声称：

> “该量化配置在 LIBERO overall 上保持精度”

时才做。

至少 S1：

```text
Spatial
Object
Goal
Long
```

统一使用与当前量化研究一致的 rollout protocol。

如果目标只是 sparsity characterization，H4 可后置。

---

# 6. 统计对象

## 6.1 Linear

VLM：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Expert：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

role：

```text
activation
weight_static
output
```

## 6.2 MatMul

```text
QK
PV
```

role：

```text
A
B
O
```

必须区分：

```text
self-attention
cross-attention
```

如果当前 module metadata 能识别。

## 6.3 phase

正式 canonical：

```text
prefill
denoise
```

prefill：

```text
flow_step = -1
```

denoise：

```text
flow_step = 0..9
```

---

# 7. 指标定义

每个 structured record 至少输出：

## 7.1 Element zero ratio

reported：

\[
Z_r=\frac{N_{zero,r}}{N_r}
\]

native quant path：

\[
Z_q=
\frac{N_{zero,r}-N_{protected}}
{N_r-N_{protected}}
\]

论文/硬件分析优先用：

```text
zero_rate_native
```

## 7.2 Bit sparsity

\[
S_{bit}
=
\frac{N_{sparse\ bit}}{N_{total\ bit}}
\]

同时保留：

```text
sparse_bit_rate_reported
sparse_bit_rate_native
```

## 7.3 Amplitude zero bit rate

保留当前项目 sign-magnitude/amplitude 口径：

\[
S_{amp}
=
\frac{N_{amplitude-zero-bit}}
{N_{total-bit}}
\]

不要与 two's-complement sparse-bit 混写。

## 7.4 Unit/block sparsity

定义：

```text
unit_bit_group_size
unit_dim_group_size
```

必须在 config/CSV 中显式记录。

\[
S_{unit}
=
\frac{N_{zero\ units}}
{N_{all\ units}}
\]

Phase H 第一轮建议保持当前默认单元，只要文档固定；后续若硬件 macro 有明确 unit，再单独扫 granularity。

## 7.5 FP side-path ratio

\[
R_{FP}
=
\frac{N_{protected}}
{N_{all}}
\]

分别报告：

```text
activation FP sidepath
weight-mask fraction
output FP sidepath
```

## 7.6 quantized coverage

如果 denominator 已被可靠记录：

\[
C_{MAC}
=
\frac{\sum MAC_{quantized}}
{\sum MAC_{target}}
\]

如果 raw operator 没有完整 runtime denominator，则只报告：

```text
quantized-workload sparsity
```

禁止擅自写 whole-model effective sparsity。

## 7.7 ideal upper bound

\[
U=\frac{1}{1-S_{bit}}
\]

字段：

```text
ideal_sparse_upper_bound
```

它不是 measured speedup。

---

# 8. 聚合规则

任何指标都不允许直接平均 percentage。

错误：

\[
(S_1+S_2+\dots+S_n)/n
\]

正确：

\[
S=
\frac{\sum_i numerator_i}
{\sum_i denominator_i}
\]

例如 zero ratio：

\[
Z=
\frac{\sum N_{zero}}
{\sum N_{elem}}
\]

bit sparsity：

\[
S_{bit}=
\frac{\sum N_{sparse-bit}}
{\sum N_{bit}}
\]

unit：

\[
S_{unit}=
\frac{\sum N_{zero-unit}}
{\sum N_{unit}}
\]

task-level 平均 SR 可以按 episode binary result 汇总，但 sparsity 一律按计数加总。

---

# 9. 建议目录结构

```text
experiments/
└── 2026-09-13_phaseH_accuracy-preserving-sparsity/
    ├── configs/
    │   ├── s0_fp8_all_base.yaml
    │   └── s1_expert_w4_base.yaml
    ├── scripts/
    │   ├── make_task_configs.py
    │   ├── run_h0_smoke.sh
    │   ├── run_h1_10ep.sh
    │   ├── run_h2_30ep.sh
    │   ├── run_h3_100ep.sh
    │   ├── summarize_sparsity.py
    │   └── plot_sparsity.py
    ├── tasks/
    └── docs/
        ├── experiment_setup.md
        ├── results.md
        └── figures/
```

原始输出：

```text
outputs/
└── 2026-09-13_phaseH_accuracy-preserving-sparsity/
    ├── manifest/
    ├── h0_smoke/
    │   ├── s0/task00/
    │   └── s1/task00/
    ├── h1_10ep/
    │   ├── s0/task00 ... task09
    │   └── s1/task00 ... task09
    ├── h2_30ep/
    └── h3_100ep/
```

---

# 10. Base config 规范

## 10.1 S0

从 G1-A 复制，不重新手写所有量化字段。

只允许修改：

```text
output_dir
evaluation.env.task_ids
evaluation.n_episodes
sparsity.*
```

保持：

```yaml
model:
  path: lerobot/smolvla_libero
  overrides:
    n_action_steps: 10
    num_steps: 10
```

保持原：

```text
method
A/W/O spec
MatMul spec
outlier_ratio
scale_dir
include/exclude
```

## 10.2 S1

同理从 G1-D 复制。

尤其不能误把：

```text
VLM raw FP
```

改成 FP8，否则就不再是原 accuracy-preserving point。

## 10.3 sparsity section

建议：

```yaml
sparsity:
  enabled: true

  chunk_size: 1048576

  unit:
    enabled: true
    bit_group_size: 2
    dim_group_size: 2

  outlier_accounting:
    enabled: true
    export_reported: true
    export_native: true

  structured:
    split_phase: true
    split_flow_step: true
    split_tensor_role: true
    split_attention_kind: true

  export:
    dir: null
```

`dir: null` 时建议由代码自动落到：

```text
<output_dir>/sparsity/
```

避免 config 内写死 task path。

---

# 11. 运行命令

## 11.1 创建实验

```bash
cd ~/VLA_tcs2

bash experiments/new_experiment.sh \
  2026-09-13 phaseH accuracy-preserving-sparsity
```

## 11.2 H0

每个 config 必须复用已有 scale：

```bash
python main.py \
  --config experiments/2026-09-13_phaseH_accuracy-preserving-sparsity/configs/<generated>.yaml \
  --skip-calibration
```

禁止：

```text
自动 recalibrate
覆盖旧 scale
```

## 11.3 task config 生成器

建议 `make_task_configs.py` 输入：

```text
--base s0_fp8_all_base.yaml
--stage h1_10ep
--episodes-per-task 1
```

自动生成临时 config：

```text
generated/h1_10ep/s0/task00.yaml
...
task09.yaml
```

每个只改：

```yaml
output_dir: outputs/.../h1_10ep/s0/task00

evaluation:
  env:
    task: libero_goal
    task_ids: [0]
  n_episodes: 1
```

task01 对应 `[1]`，以此类推。

不要人工复制 20 份 YAML，容易配置漂移。

---

# 12. H0 Smoke 验收

H0 不看“sparsity 高不高”，只看系统正确。

必须全部 PASS：

- [ ] S0/S1 能加载已有 scale
- [ ] 没有 calibration
- [ ] eval 可完整结束
- [ ] `eval_info.json` 存在
- [ ] `module_sparsity.csv` 非空
- [ ] `weight_sparsity_static.csv` 非空
- [ ] `outlier_sidepath.csv` 非空
- [ ] `unit_sparsity.csv` 非空
- [ ] prefill row 存在
- [ ] denoise flow_step 0..9 全存在
- [ ] Linear activation/output 都存在
- [ ] QK/PV A/B/O 都存在
- [ ] `0 <= native_zero_rate <= 1`
- [ ] `0 <= native_sparse_bit_rate <= 1`
- [ ] `0 <= fp_sidepath_ratio <= 1`
- [ ] native denominator 非负
- [ ] weight row 数 = 当前 wrapped QuantizedLinear 数
- [ ] stats ON/OFF 固定输入数值等价测试已 PASS

任何一项失败：

```text
H0 = FAIL
禁止进入 H1
```

---

# 13. H1/H2 收敛判断

聚合以下主指标：

```text
VLM prefill activation sparse_bit_rate_native
Expert denoise activation sparse_bit_rate_native
Expert denoise output sparse_bit_rate_native
QK A/B/O sparse_bit_rate_native
PV A/B/O sparse_bit_rate_native
weight static sparse_bit_rate
unit_zero_rate
fp_sidepath_ratio
```

H1=10ep，H2=30ep。

推荐收敛 gate：

\[
|\Delta S_{bit}| < 0.5\text{ pp}
\]

\[
|\Delta Z| < 0.5\text{ pp}
\]

对 aggregate 大类成立。

layer-level 因 sample 少允许更大波动，不作为 H2 gate。

如果 aggregate 已收敛：

> 可以说明 sparsity characterization 不依赖 100ep 才稳定。

但 H3 仍有价值，因为它会重新验证 stats-on SR。

---

# 14. 最终结果表

## 14.1 Accuracy + coverage 总表

| Config | Goal SR | Quantized scope | MAC coverage | FP sidepath | Native bit sparsity | Unit sparsity |
|---|---:|---|---:|---:|---:|---:|
| S0 FP8-all |  |  |  |  |  |  |
| S1 Expert-W4 |  |  |  |  |  |  |
| S2 optional |  |  |  |  |  |  |

若 coverage denominator 不完整，MAC coverage 写：

```text
N/A — raw-op runtime denominator not instrumented
```

不能估。

## 14.2 Component 表

| Config | Component | Phase | Role | Zero native | Bit sparse native | Unit sparse | FP sidepath |
|---|---|---|---|---:|---:|---:|---:|
| S0 | VLM | prefill | activation | | | | |
| S0 | Expert | denoise | activation | | | | |
| S0 | Expert | denoise | output | | | | |
| S0 | QK | denoise | A | | | | |
| S0 | QK | denoise | B | | | | |
| S0 | QK | denoise | O | | | | |
| S0 | PV | denoise | A | | | | |
| ... | | | | | | | |

## 14.3 Flow-step 表

| step | Expert A bit sparse | Expert O bit sparse | QK A | QK B | QK O | PV A | PV B | PV O |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | | | | | | | | |
| 1 | | | | | | | | |
| ... | | | | | | | | |
| 9 | | | | | | | | |

---

# 15. 必做图

## Figure H1：Layer × operator heatmap

横轴：

```text
q k v o gate up down
```

纵轴：

```text
layer 0..15
```

分别画：

```text
VLM prefill
Expert denoise
```

指标：

```text
sparse_bit_rate_native
```

## Figure H2：flow-step sparsity curve

x：

```text
0..9
```

y：

```text
native bit sparsity
```

曲线：

```text
Expert activation
Expert output
QK A/B/O
PV A/B/O
```

## Figure H3：precision × sparsity × accuracy

至少对 S0/S1：

x：

```text
effective precision/deployment point
```

左 y：

```text
Goal SR
```

右 y：

```text
native sparse-bit opportunity
```

不要把 ideal upper bound 当实测 speedup。

## Figure H4：FP sidepath breakdown

展示：

```text
normal quant path
FP protected path
```

按：

```text
VLM / Expert
qkv/o / MLP
flow step
```

分解。

---

# 16. 结果解释规范

允许写：

> 在 Goal、n_action_steps=10 协议下，Expert-W4 配置维持与 FP8 anchor 相近的闭环 SR，同时在量化 Expert workload 上观察到 X% native bit sparsity。

不允许直接写：

> SmolVLA 整体有 X% sparsity。

除非 whole-model denominator 已经完整统计。

允许写：

> ideal sparsity upper bound 为 X×。

不允许写：

> 硬件 speedup 为 X×。

除非后续硬件 simulator 已纳入：

```text
memory traffic
FP sidepath
unit granularity
load imbalance
reduction
data movement
raw-FP fallback
```

---

# 17. 风险与注意

## R1：outlier 假稀疏

如果只看 masked normal code，会高估稀疏度。

缓解：

```text
reported + native + FP sidepath 三者同时导出
```

## R2：static weight 语义混淆

`collect_model_weight_sparsity()` 是 full-weight quantized intrinsic sparsity，不等于每次 runtime dynamic mask 后的 weight path。

必须在表头/metadata 标明。

## R3：flow step 被错误合并

只有 structured key 真正包含 flow_step 后，才允许画 step 0..9 曲线。

## R4：unit 数据粒度不足

旧 unit accumulator 若没 role/step，不用于正式 H2/H3 结论。

## R5：统计器改变结果

stats collector 只能 detach/read，不得：

```text
in-place 修改 code
改变 RNG
重新 quantize forward output 并替换原 tensor
```

## R6：SR anchor 协议漂移

Phase H 必须保持：

```text
n_action_steps=10
num_steps=10
seed=1000
```

否则 88% / 84% 不再是可直接比较的 anchor。

## R7：100ep 成本

先 H1→H2 做 convergence。不要一开始就 2×100ep。

---

# 18. 输出目录映射

| Group | Config source | Stage | Output |
|---|---|---|---|
| S0 | G1-A derived | H0 | `outputs/.../h0_smoke/s0/task00` |
| S1 | G1-D derived | H0 | `outputs/.../h0_smoke/s1/task00` |
| S0 | G1-A derived | H1 | `outputs/.../h1_10ep/s0/taskXX` |
| S1 | G1-D derived | H1 | `outputs/.../h1_10ep/s1/taskXX` |
| S0 | G1-A derived | H2 | `outputs/.../h2_30ep/s0/taskXX` |
| S1 | G1-D derived | H2 | `outputs/.../h2_30ep/s1/taskXX` |
| S0 | G1-A derived | H3 | `outputs/.../h3_100ep/s0/taskXX` |
| S1 | G1-D derived | H3 | `outputs/.../h3_100ep/s1/taskXX` |

---

# 19. 最终 Gate

Phase H 可以标记 `done` 的条件：

- [ ] code accounting tests 全 PASS
- [ ] H0 correctness PASS
- [ ] H1/H2 convergence 有记录
- [ ] H3 stats-on SR 完成
- [ ] reported/native/outlier 三种语义未混淆
- [ ] prefill/denoise 分开
- [ ] flow step 0..9 分开
- [ ] Linear A/W/O 与 MatMul A/B/O 覆盖完整
- [ ] task-level 数据可追溯
- [ ] aggregate 使用 numerator/denominator 加总
- [ ] upper bound 未冒充硬件 speedup
- [ ] `results.md` 记录 commit、scale hash、config、命令、输出目录

---

# 20. 后续行动

Phase H 完成后再进入硬件映射：

```text
Phase H:
accuracy-preserving quantization
        ↓
true/native sparsity
        ↓
outlier sidepath
        ↓
phase/flow-step workload
        ↓
Phase I:
CIM / bit-serial / sparse scheduler modeling
```

Phase I 的核心输入应来自结构化 CSV，而不是手工抄一个 overall sparsity。

