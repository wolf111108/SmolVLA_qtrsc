# Phase M — 论文双极性稀疏架构上的 VLM/expert 单次推理

状态：软件验证完成；完整 SmolVLA 推理由用户本地运行。没有芯片实测或完整模型性能结果。

## 目的

以一次真实量化 inference 为唯一 workload，分析论文双极性稀疏架构上 VLM prefill 与 expert denoise 的延迟来源、外部 EMA 读取量和部件占用。

## 环境与版本

使用仓库现有 `smolvla_eval` 环境、锁定的 LeRobot 版本与 `envs/smolvla_eval_pip_freeze.txt`。
离线重放：Python>=3.10、numpy、PyYAML；绘图：matplotlib；捕获：torch TorchDispatchMode。
`run_metadata.json` 自动记录源码 commit、工作区状态、安装包版本、checkpoint revision、seed 与输入来源。

## 模型与数据

一次 `predict_action_chunk`：batch=1、chunk=50、VLM prefill 一次、expert denoise 十步。
模型与 phaseL 相同：`lerobot/smolvla_libero`，revision `a24f1f380461b665158d52acd140152c79e8b36d`。
Vision encoder、connector 在主机 FP32 执行且不量化；state/action/time projection 与动作积分保持浮点外围路径。
硬件 workload 只采集 VLM/expert Transformer 的量化 GEMM，以及其 forward 范围内实际执行的 ATen 辅助操作。
VLM A8/W8/O8；expert Linear A8/W4/O8；全部 QK/PV A8/B8/O8。静态 per-tensor、per-site scales、无 outlier。
原有统计配置 d_bit=4/p=4 不用于论文压缩；新稀疏模型独立使用 2-bit BG 与 p=8/64。

完整 inference 是实际功能路径完整执行，不将模拟 VLM/expert 时间称为包含摄像头、主机和控制器的机器人端到端延迟。
原始 checkpoint 的 host 模型显式 `.float()`，关闭 autocast。模拟器对浮点辅助接口按实际容器宽度计流量。

## 评测协议

优先复用 phaseL 的单条 `input_batch.pt`，无数据集遍历；默认同观测校准一次但不采集，然后只采集一次完整量化 inference。
所有组离线重放同一份整数码、操作序列与稀疏掩码。完整验收要求一次 generation、一次 prefill、连续 step 0–9、有限 actions 与非空辅助操作。
未提供输入时生成 seed=42 的合成观测；这些数据用于硬件工作负载分析，不用于任务成功率评估。

## 实验变量与分组

控制变量：checkpoint、输入、整数码、量化精度、时钟、SRAM 容量、映射和外围边界均保持一致；仅改变下表指定功能或带宽/辅助吞吐。

| 组 | 设置 |
|---|---|
| D0 | 相同 digit 阵列，dense p64/b32，不做 BG 跳过、SSS、IAG |
| D1 | BSC 开启、固定 sparse p8/b256 |
| D2 | BSC+SSS，IAG 关闭 |
| D3 | BSC+SSS+IAG score-only，12 GB/s |
| D3_bw6 / bw24 | 6 / 24 GB/s |
| D3_ideal_memory | 保留原访问流量，去掉外存请求/服务时间 |
| D3_ideal_bsc | 保留存储与工作量，去掉 BSC 检测/压缩处理时间 |
| D3_aux0.5 / aux2 | 辅助运算速度 0.5× / 2× |

延迟：VLM prefill、每步 denoise、expert 总计；同时给出可加和关键路径分类、各资源 busy/service、digit 槽位利用率。
resource wait 总和可重叠，不能加到阶段延迟。关键路径是一个确定性调度路径，不代表唯一的瓶颈因果分解；用理想化重放补充判断。

EMA：payload 与 burst bus 字节分别记录。分类包括 weight、activation、vlm_kv、expert_kv、mixed_kv、partial_sum、metadata、auxiliary、boundary。
混合 self-attention KV 保持 mixed_kv 标签；不伪造 prefix/suffix 的精确切分。来自图外的 FP 输入按实际 FP 字节计。
唯一读取 payload 与累计读取区分；不能将唯一权重大小当作十步 denoise 的实际读取量。

## 运行命令

推荐在已有 `smolvla_eval` 环境运行。沿用仓库锁定的 LeRobot 环境与 `envs/smolvla_eval_pip_freeze.txt`；
离线重放只需要 Python>=3.10、numpy、PyYAML，绘图另需 matplotlib。捕获需要 torch 的 TorchDispatchMode。
不下载/遍历完整数据集；优先传 phaseL 的 `input_batch.pt`。未提供时使用 seed=42 的合成观测，明确不代表任务成功率。
因 vision/connector 精度变化，默认使用同一观测执行一次**不采集的校准**，随后只采集一次完整量化 inference。
可通过 `--reuse-scales` 使用已经验证兼容的 VLM/expert 尺度。

```bash
# 仓库根目录；激活已有 smolvla_eval 环境后运行
bash experiments/2026-10-19_phaseM_paper-bipolar-vlm-expert/scripts/run.sh \
  --batch outputs/2026-10-16_phaseL_gemm-single-inference-timeline/int8_w4/input_batch.pt
```

可传 `--checkpoint /local/model`、`--device cuda`、`--reuse-scales /local/scales`。
通过 `RUN_OUTPUT=/new/output/path` 选择新输出目录；捕获拒绝覆盖已存在目录。

分步执行（完整采集后可反复离线重放，无需 GPU 或 checkpoint）：

```bash
python experiments/2026-10-19_phaseM_paper-bipolar-vlm-expert/scripts/run_one.py --output-dir outputs/phaseM_local
python experiments/2026-10-19_phaseM_paper-bipolar-vlm-expert/scripts/replay.py --run-dir outputs/phaseM_local --suite
python experiments/2026-10-19_phaseM_paper-bipolar-vlm-expert/scripts/plot.py --run-dir outputs/phaseM_local
```

`replay.py` 默认只跑 D3；`--suite` 跑全部对照。默认仅 D3 导出完整 gzip 时间线，其他组保留汇总。
`--no-timeline` 可用于只关心 EMA/延迟的重放。已有成功且配置相同的组可复用；失败/配置不一致的目录拒绝覆盖。
完整量化 tensor 与事件文件较大；采用压缩、流式写入与分块事件调度，工作目录需预留足够空间。

## 输出目录映射

| 配置/产物 | 输出目录 |
|---|---|
| `configs/int8_w4.yaml` 主运行 | `outputs/2026-10-19_phaseM_paper-bipolar-vlm-expert/int8_w4/` |
| 一次捕获 | 主运行目录 `/workload/` |
| D0/D1/D2/D3 与敏感性组 | 主运行目录 `/replays/<group>/` |
| 软件小模型烟测 | `outputs/2026-10-19_phaseM_paper-bipolar-vlm-expert/software-smoke/`；不作为 SmolVLA 结果 |

## 风险与注意

论文没有给出完整 SoC 时序；外存、辅助单元、bank、块分配与算子边界策略按下文明确建模。
完整事件时间线和真实整数输入可能较大，重放耗时由真实序列/维度决定；采用流式压缩写入，只有主 D3 默认导出全事件。
默认使用独立输出目录，失败运行不得作为结果复用。未知辅助算子、超容量工作集、量化精度不匹配均明确失败。
当前只有小模型和软件数值/调度验证，完整 checkpoint 推理留给用户本地执行。

## 论文事实与系统假设

依据 Zhao 等《A 3.51 TOPS/mm² Transformer Accelerator Exploiting Bipolar Sparsity and Approximate Gating》，
DOI `10.1109/TCSI.2025.3601717`，主要参考 Fig.2–9、Table II。RTL chipds 仅辅助理解，不替代论文架构。

| 项目 | 参数 | 来源 |
|---|---|---|
| BPE | 2048，输入 2-bit BG、权重 4-bit digit | 论文 |
| 双 BSC | ping-pong，C1/C2/C3 | 论文 |
| SSS | sparse p8/b256，dense p64/b32，阈值 2/3 | 论文 |
| 时钟 | 390 MHz | 论文最高性能点 |
| SRAM | input 16 KiB，weight 128 KiB，metadata 8 KiB | 论文容量 |
| 输入/权重/metadata 读口 | 32/128/16 B/cycle | 图示及供数需求推导；不是外存带宽 |
| 部分和 | 2048×INT32=8 KiB，PE 内部状态 | 位宽与容量假设 |
| 外存 | 有效 12 GB/s，读写共享，64 B burst，100 ns 初始延迟 | 实验假设 |
| DMA | 8 outstanding **多 burst 块请求** | 假设；不是八个 64 B 请求 |
| SRAM bank | input/weight 各两个等容量块交错 bank，bank 内 LRU | 补充系统实现 |
| 缓存块 | 1024 B，可配置 | 假设，实际 EMA 按填充/逐出统计 |
| auxiliary | 32 elements/cycle，逐元素启动 1 cycle，特殊函数延迟 8 cycles | 参数化成本模型 |
| metadata 工作区 | 1 KiB 预留，剩余容量用于符号/坐标缓存 | 假设，溢出计入 DRAM |

SRAM 同一 bank 读写互斥，不同 bank 使用独立 bank 资源，但共享各自的读/写总带宽。
有效包 run 会在整个 run 中保留它涉及的 bank；这是保守的事务粒度，服务需求与占用时间分别报告。
BSC 输入 p64×16 的逻辑 BG 块有 256 B；32 B/cycle 下 C1 分八次，并与对应 C2 流水重叠，不能假设整块固定三周期。
每次按 p8 子块分配并消费，staging 期间锁定所需缓存块，防止使用前被逐出；超过 bank 容量明确报错。
p8×16 对应 32 B 主数据、16 B 符号。符号在量化时生成、有限 metadata SRAM 中保存，miss/写回均收费。
尾块的无效 token/通道不产生有效 MAC，物理端口与 pipeline 槽位仍按配置宽度占用。

## 算法与事务调度

1. Python 在 quant_awo 后捕获实际整数码；普通输入 hook 不代替整数观测。捕获不修改 forward 输出、RNG 或量化尺度。
2. A8 分四个 BG。负数使用逐组取反及最低位补偿；最低负 BG 保留补偿计算。只有同一 K 位置的全部 p 个有效 token 可跳过，才压缩整包。
3. SSS 在每个 64-token slab、每个尚未进入 dense 的 BG 上预扫描前 8 个 token。若保持 sparse，继续扫描其余 p8 组；dense 后不再预扫描。
4. 每次扫描支付 C1/C2、符号/坐标存储开销；全零 tile 跳过普通 C1/C2/C3。元数据容量不足会逐出和重读。
5. BG-major、输出 OS；每个输出 microtile 在一个 BG 内遍历全部 K 与 weight digits。BSC 两套缓冲交替准备/发射。
6. 累加器按逻辑 `(batch/head,m range,n range)` 管理。下一个 microtile 坐标相同可保留；被替换的部分和以 INT32 spill/restore。
   SSS 切换本身不收固定 64 KiB 费用。不同循环顺序会改变 spill，这个结果只属于已记录的映射。
7. 本版在 GEMM 前物化实际所需的量化表示，超容量部分经 DRAM 保存；输出恢复为实际浮点辅助接口。
   这是一种保守的算子边界策略，没有假设跨算子融合、量化与 GEMM 融合或无限片内驻留。
8. 连续块 W 从 DRAM 装载。BSC 控制阵列消费时的选中 SRAM 行；BG 稀疏率不会直接乘到 DRAM 字节数上。
   完全未消费的外存块可避免装载，实际收益由事务决定。
9. 算子按捕获顺序串行；算子内部允许 DMA/BSC/array 并行。SRAM 内容跨算子和 denoise step 保留，命中以真实有限容量为准。

`StreamingEventSimulator` 使用现有 `Action/Phase/PlanEvent/Architecture.lower_event`。
它按 planner 发射顺序使用可填补空隙的资源日历；所有依赖/资源占用和关键路径前驱都导出。
算子边界清理过去的日历区间，保留存储状态，避免完整 inference 事件图常驻内存。
计算 run 的 `effective_addresses`、BG、p/b、weight_digits 可用于独立重建整数 GEMM。

## IAG 与辅助算子

D3 为 score-only：PV 中从个体 BG 有效标志生成 token 分数，attention output projection 中生成 head 分数，
记录在 workload/iag。token/head 全部保留，不宣称 IAG 剪枝收益，不通过删除原 trace 伪造剪枝后的模型执行。
评分只在相应算子和首次输出通道块上收费。D2 关闭 IAG，用于对照评分开销。

辅助算子由 TorchDispatchMode 捕获：量化模块内部的 FP reference GEMM、日志、统计被屏蔽，避免重复计算。
真实 view/transpose/expand 共享 storage；clone/cat/repeat 等实体拷贝收费。
ATen materialized GQA 重复 K/V 按实际执行的拷贝表示计量，未增加优化后的逻辑 head 共享硬件。
embedding/index_select/gather/index 捕获实际选中地址，按所选行/元素及 cache block 计量，不扫描整个词表。其他
strided view 使用实际 storage span，包含洞的保守 overfetch；外存类别会显示其开销。
归约、Softmax、Norm 的性能是分块、多遍分析模型，不声称还原特定非线性单元 RTL。
未支持的辅助算子会明确失败，不会在完整阶段报告中静默忽略。主机侧无 tensor 输出的控制操作不进入目标时间。

特别注意：phaseL 的整数输出缩放路径没有饱和到 O8。捕获保持原数值，不新增 clamp；
manifest 记录超范围输出数、各 GEMM 记录实际 code 存储位宽，浮点功能接口保持真实宽度。
INT32 最坏累加上界超过范围时拒绝仿真，避免隐藏溢出。

## 输出与验收

- `run_metadata.json`、`resolved_config.yaml`、`precision_audit.json`：版本、checkpoint、输入、实际模块位宽与状态。
- `input_batch.pt`、`actions.pt`、`scales/`：可复用输入、动作和量化尺度。
- `workload/operators.jsonl.gz`、`codes/`、`sparsity/`、`iag/`、`lookups/`、`manifest.json`：一次实际前向的可重放证据。
- `replays/D3/report.md`、`summary.json`：主结果，先读这两个文件。
- `replays/*/operators.json`、`sparsity.csv`：逐算子贡献、模式与有效 tile 统计。
- `replays/D3/timeline/events.csv.gz`、`chrome_trace.json.gz`：完整事务与资源时间线，支持 Perfetto。
- `replays/D3/timeline/occupancy_bins.npz`、`figures/`：流水热图、局部时序、延迟/EMA 分解、对照图。
- `run.log`、`replay.log`、`plot.log`：一键脚本运行日志。
- `comparison.json`：同一 workload 的对照组数据。

软件检查：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_paper_bipolar.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_hardware*.py' -v
python experiments/2026-10-16_phaseL_gemm-single-inference-timeline/scripts/check_export.py -q
```

验收包括全部 INT8×INT4/INT8 标量组合的数学等价、负数 LSB、联合包/尾块、模式切换事务重建 GEMM、缓存/burst 守恒、
资源互斥和依赖、关键路径和总延迟一致，以及真实 QuantizedLinear/MatMul 的小模型捕获不改变输出。
完整 SmolVLA 验收还要求一次 generation、VLM/expert 均存在、step 0–9、有限 actions、辅助记录非空且无 unsupported 操作。
结果目录不进 git；完整模型与数据集未上传。
