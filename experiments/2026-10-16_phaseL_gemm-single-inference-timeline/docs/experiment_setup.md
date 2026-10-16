# 实验设置（Experiment Setup）

- 实验名称：2026-10-16_phaseL_gemm-single-inference-timeline
- 状态：draft（待本地运行；未产生真实 SmolVLA 性能结果）
- 创建日期：2026-10-16
- 前序：[组件框架说明](../../../docs/hardware_components.md)

## 1. 实验目的

采集一条观测触发的一次完整 SmolVLA `predict_action_chunk`，即一次 VLM prefill 和 10 次 expert denoise。使用当前组件框架分析所有捕获到的 Linear/QK/PV GEMM，得到数据流量、计算与 DRAM 服务时间、完整资源占用区间、资源利用率以及可以绘制流水瓶颈图的数据。不执行整个数据集或环境 rollout，不报告 success rate。

本实验固定 OS、双缓冲和 dense 阵列；不搜索数据流、不统计稀疏度、不启用跳零加速。结果仅对应当前映射策略，不能据此判断 WS/IS 的最优潜力。

## 2. 环境与版本

| 项目 | 要求或记录方式 |
|---|---|
| 推荐环境 | 已能运行本仓库 SmolVLA 的 `smolvla_eval`，GPU 推荐；可用 `--device cpu` 检查执行 |
| LeRobot / Transformers | 沿用仓库 `envs/smolvla_eval_pip_freeze.txt`；其中 LeRobot 为 `6adf51511b7625090eade8d82d9f61a1846ebe56` |
| 额外绘图依赖 | matplotlib、numpy；YAML 使用 PyYAML，均非硬件模拟器新增依赖 |
| 版本证据 | `run_metadata.json` 记录源 commit、dirty 状态、依赖版本、输入来源和模型设置；`resolved_config.yaml` 保存最终配置 |
| GPU | 在本地执行；未在此环境启动模型推理 |
| 模型下载 | 首次执行需要 checkpoint 和 VLM processor/config 的 Hugging Face 访问，或已缓存资源 |

## 3. 模型与数据

| 项目 | 设置 |
|---|---|
| checkpoint | `lerobot/smolvla_libero`，固定 revision `a24f1f380461b665158d52acd140152c79e8b36d`；也可传 `--checkpoint /local/path` |
| 输入 | 默认固定 seed=42 的合成观测，checkpoint 声明的全部相机、零状态、固定任务文本；不是 LIBERO 真实帧 |
| 相机和文本 | 不缩小 checkpoint 图像分辨率；按策略 `prepare_images` 路径处理；文本 padding 到 tokenizer_max_length |
| 推理长度 | chunk_size 沿用 checkpoint（50），n_action_steps=50，num_steps=10，batch=1 |
| 校准 | 默认用相同观测做一次不采集硬件事件的 scale_inspection action chunk，保存 per-site 尺度；仅用于本次硬件 smoke characterization，不作为量化精度验证 |
| 可选实际观测 | `--batch batch.pt`：可信文件，字典各值均为 batch=1 的 Tensor，已经完成模型侧预处理和 state 归一化，包含相机、language tokens/mask、state；不自动下载数据集 |
| 可选正式尺度 | `--reuse-scales /path`，必须与本次 INT8/INT4 方法和 per-site 名称匹配；不复用历史 FP8/outlier 尺度 |

精度定义为 activation / weight。Vision、connector 和 VLM Linear 使用 A8/W8/O8；expert Transformer Linear 使用 A8/W4/O8。所有 attention QK/PV 使用 A8/B8/O8：其 B 是 activation/K/V，不是 INT4 模型权重。action/state/time projection 等未包装的 action-head Linear 按实际浮点容器位宽采集并单独标注，不能声称所有 GEMM 都是低精度。脚本输出 `precision_audit.json`，并校验各实际安装模块的位宽和方法。

## 4. 评测协议

先构建、安装量化模块和运行上下文 hooks，再执行可选的单输入尺度校准。清空 policy 队列，重设随机种子，安装 HardwareManager hooks，仅调用一次 `predict_action_chunk`；结束后移除 hooks。真实主机上仍执行非 GEMM 算子，硬件模拟时间仅覆盖捕获到的 GEMM。

验收：actions 有限；仅一个 generation；denoise flow_step 恰好为 0…9；vision/connector/vlm/expert 均有 GEMM；无 capture truncation、无 unsupported estimate。失败时 `run_metadata.json` 标记 failed，shell 停止后续出图。统计和绘图不会把失败运行包装成完整结果。

| 指标 | 定义 |
|---|---|
| GEMM-only 延迟 | 按捕获调用顺序拼接每个 GEMM 的 simulation.total_cycles；不同算子串行，非 GEMM 间隔为零 |
| 计算服务时间 | 阵列资源需求周期累加/架构时钟；不含等待；不能直接与访存时间相加 |
| DRAM 服务时间 | 每阶段 ceil(bytes/rate) 累加/时钟，不含固定延迟 |
| DRAM 占用时间 | DRAM 锁定区间之和/时钟，包含固定延迟及同步事务限制 |
| 资源占用率 | 同一物理资源 busy_cycles/GEMM-only 总周期 |
| 服务需求比例 | 每资源 ceil(demand/rate) 周期之和/总周期；模型未定义这些服务在锁定区间内的准确位置 |
| 部件占用率 | 部件各端口区间的并集/总周期；不直接相加读写端口比例 |
| 有效 PE 利用率 | executed_macs/(rows×cols×总周期)，区别于阵列被锁定比例 |
| 每事件等待 | start-ready；不同事件等待会重叠，不是可相加的关键路径停顿 |

## 5. 实验变量与分组

| 组 | config | 精度 | 映射 | 硬件 |
|---|---|---|---|---|
| int8_w4 | int8_w4.yaml | VLM A8/W8，expert A8/W4；其余见 §3 | OS；64×64×128（M×N×K）；双缓冲 | 32×32 PE，1 GHz；dense、1 MAC/PE/cycle |

完整硬件架构自包含于 YAML。DRAM 为 102.4 GB/s×0.7，共享读写，100-cycle 固定延迟，64-byte burst；GB 4 MiB、Local 128 KiB、Accumulator 64 KiB；SRAM 独立读写端口均为 256 B/cycle，互连为 256 B/cycle，部分链路共享。INT4 减少权重流量，不自动提高阵列 MAC 吞吐。

## 6. 运行命令

在仓库根目录激活现有环境后执行：

```bash
conda activate smolvla_eval
bash experiments/2026-10-16_phaseL_gemm-single-inference-timeline/scripts/run.sh
```

复用本地 checkpoint、正式尺度或实际预处理观测（按需添加参数）：

```bash
RUN_OUTPUT=outputs/2026-10-16_phaseL_gemm-single-inference-timeline/local_input \
  bash experiments/2026-10-16_phaseL_gemm-single-inference-timeline/scripts/run.sh \
  --checkpoint /absolute/path/to/smolvla_libero \
  --batch /absolute/path/to/one_preprocessed_batch.pt \
  --reuse-scales /absolute/path/to/compatible_int8_int4_scales
```

不复用尺度时省略 `--reuse-scales`，会进行一次校准前向，再进行一次计入报告的量化推理。`PYTHON_BIN=/path/to/python` 可选择解释器；输出目录已存在会拒绝运行，用新的 RUN_OUTPUT 保留旧结果。

分步执行和重绘：

```bash
python experiments/2026-10-16_phaseL_gemm-single-inference-timeline/scripts/run_one.py
python experiments/2026-10-16_phaseL_gemm-single-inference-timeline/scripts/export_timeline.py \
  --run-dir outputs/2026-10-16_phaseL_gemm-single-inference-timeline/int8_w4/hardware
python experiments/2026-10-16_phaseL_gemm-single-inference-timeline/scripts/plot_timeline.py \
  --timeline-dir outputs/2026-10-16_phaseL_gemm-single-inference-timeline/int8_w4/timeline \
  --bins 1000 --detail-cycles 10000
```

导出器使用同一 manifest 和 mapping 重建 Action 资源需求，并核对存储访问计数和容量预留与原 simulation 一致；不会重新选择 mapping。请使用生成结果时的代码版本导出。`--output-dir` 可将第二次导出写入新目录。

无需模型或 GPU 的软件验收：

```bash
python experiments/2026-10-16_phaseL_gemm-single-inference-timeline/scripts/check_export.py -q
PYTHONPATH=src python -m unittest discover -s tests -p 'test_hardware*.py' -q
```

## 7. 输出目录映射

| config | 原始输出目录 | 状态 |
|---|---|---|
| int8_w4.yaml | outputs/2026-10-16_phaseL_gemm-single-inference-timeline/int8_w4/ | pending，待本地运行 |

| 文件（相对本次运行目录） | 内容 |
|---|---|
| resolved_config.yaml / run_metadata.json | 生效配置、版本、输入、运行状态、单次推理验收结果 |
| precision_audit.json | 实际包装的模块及三种操作数位宽 |
| input_batch.pt / actions.pt / scales/ | 可复用的输入、动作输出和本次尺度 |
| hardware/trace.jsonl | 原始 OperatorEvent 与 manifest，可离线重放 |
| hardware/simulations.jsonl | 每 GEMM 的局部资源时间线、访问计数与模型假设 |
| hardware/operators.csv / summary.json | 框架原有逐算子与聚合指标 |
| timeline/events.csv.gz | 每事件的全局整数周期起止、ready/wait、资源列表、动作类型、depends_on 依赖、固定延迟和逐资源服务周期 |
| timeline/chrome_trace.json.gz | 各物理资源分轨的完整时间线；标准 Chrome Trace X slices |
| timeline/operators.csv | 所有 GEMM 的全局起止、形状、位宽、tile、流量、计算与访存时间 |
| timeline/resources.csv / components.csv | 各物理端口资源占用与服务比例、部件端口占用并集 |
| timeline/summary.json / report.md | 总体与 prefill/每步 denoise 指标、可直接阅读的汇总 |
| timeline/occupancy_bins.npz | 图像对应的资源分箱占用率；精确区间交叠积分，不是采样命中率 |
| timeline/figures/pipeline_occupancy.png/pdf | 全推理各资源忙闲热图，竖线标记阶段切换 |
| timeline/figures/resource_utilization.png/pdf | 占用率与有效服务需求比例并排对照 |
| timeline/figures/detail_*.png/pdf | vision/connector/vlm/expert/action_head 等代表性 GEMM 的局部 Gantt 图 |

完整数据不进 git。Chrome Trace 可在 [Perfetto UI](https://ui.perfetto.dev/) 中选择本地文件打开；若工具不接收 gzip，先 `gzip -dk .../chrome_trace.json.gz`。JSON 时间戳用微秒，args 和 CSV 保存原始整数 cycle，可定位 module、tile、flow_step。先看 report.md 和两张总图，再在交互时间线查看具体 GEMM。全事件展开文件可能较大，已对 CSV/Chrome Trace 压缩；无需解压 CSV 即可绘图。

## 8. 风险与注意事项

- 结果是当前假设硬件上的 GEMM 模型，不是 GPU 主机耗时或真实芯片实测。host_seconds 包含 Python 量化、hook 和在线仿真开销。
- softmax、normalization、激活函数、量化尺度处理、图像 patch convolution 和其他非 GEMM 工作不进入目标时钟；没有跨算子驻留、跨算子流水或端到端策略延迟。
- Local read、阵列计算、Accumulator 访问被组合成同一同步事件，资源锁定持续到最慢参与者完成；因此高占用不一定意味着高有效吞吐。
- 默认 synthetic 输入和单观测校准仅用于 workload/流水检查，不支持量化精度或任务成功率结论。若需要有实际任务含义的观测，请传已经预处理好的单 batch 和正式尺度。
- 原始 actions 是模型空间输出，没有执行环境动作或 action unnormalizer，不可直接作为机器人控制命令。
- 严格通过完整 generation、flow-step、组件覆盖及 unsupported 检查后，才将该运行解释为完整 GEMM 结果。模型尚未在本次交付中实跑。
