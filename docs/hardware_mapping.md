# 逐层 GEMM mapping 搜索

实现位于 `src/vla_tcs2/hardware/timeloop.py`。这是项目自定义、仅依赖 Python 标准库的
**Timeloop-style 解析搜索器**，没有调用官方 Timeloop/PyTimeloop。
现有 `DenseAnalytic` backend 是搜索和最终估算共用的成本模型；backend 版本更新为 2。

## 1. 使用方式

### 在线采集／硬件建模

在现有评测 YAML 的 `hardware` 中加入 `mapping_search`：

```yaml
hardware:
  enabled: true
  backend: dense_analytic
  hardware:
    pe_rows: 16
    pe_cols: 16
    frequency_hz: 1000000000
    sram_bytes: 1048576  # global buffer，字节
    accumulator_bits: 32
    dram_bytes_per_second: 32000000000
    dram_mode: shared
    activation_read_bytes_per_second: 32000000000
    weight_read_bytes_per_second: 32000000000
    overlap_compute_memory: false
  mapping_search:
    enabled: true
    objective: memory_time
    candidate_policy: powers_of_two
    dataflows: [output_stationary, weight_stationary, input_stationary]
```

之后照常执行 `python main.py --config ...`。每个捕获的 Linear/QuantizedMatMul 事件都会
提供 shape、batch 和逻辑位宽，搜索最佳候选并将其用于该事件的性能估算。
相同 workload/hardware/search/model-version 复用内存缓存；不同层和 denoise step
仍各自保存一条结果，不假设权重／KV 跨调用驻留。

搜索默认关闭，保留旧配置的固定 mapping 行为；开启后 `mapping` 固定 tile 不再参与选择。
搜索暂时只支持 `backend: dense_analytic`，防止搜索模型与最终估算后端不一致。

### 离线 trace 重放

```bash
PYTHONPATH=src python -m vla_tcs2.hardware \
  --trace outputs/YOUR_RUN/hardware/trace.jsonl \
  --config configs/hardware/timeloop.json \
  --output-dir outputs/YOUR_RUN/mapping_search
```

不需要安装 PyTorch、LeRobot 或官方 Timeloop。若不传 config，会继承源 trace manifest
中的 mapping_search 设置。若要恢复固定 mapping，显式设置 `mapping_search.enabled: false`。
输出目录不能覆盖已有报告。

### 直接输入 A、B、C

```python
from vla_tcs2.hardware import HardwareSpec, SearchSpec, TimeloopMapper

mapper = TimeloopMapper(
    HardwareSpec(sram_bytes=1048576,
                 activation_read_bytes_per_second=32e9,
                 weight_read_bytes_per_second=32e9),
    SearchSpec(objective="memory_time"),
)
result = mapper.map_gemm(
    256, 1024, 4096,  # activation[A,B] @ weight[B,C]
    activation_bits=16, weight_bits=16, output_bits=32,
    save_path="outputs/mapping_example.json",  # 可选；独占创建，不覆盖
)
print(result.mapping)
print(result.estimate)
```

代码使用 M/N/K 命名：M=A、N=C、K=B；tile_m=x、tile_n=z、tile_k=y。
Linear 的捕获权重 shape 为 `[N,K]`，解析计算仍按 `[M,K] @ [K,N]` 建模。
`map_event(event)` 直接接受现有 OperatorEvent。

## 2. 搜索空间与最优性的含义

每个候选包含 `(tile_m, tile_n, tile_k, dataflow)`。先按容量剪枝，再由 DenseAnalytic 评估。
不模拟每次乘加，使用完整块／尾块分组计算，成本与矩阵元素总数无关。

| 配置 | 含义 |
|---|---|
| `objective: memory_time` | 最小化 DRAM 搬运服务时间，包含输出写入及 partial sum 读写 |
| `objective: latency` | 最小化 backend 的 GEMM core 延迟估计 |
| `objective: dram_bytes` | 最小化 DRAM 读写总字节数 |
| `candidate_policy: powers_of_two` | 默认：各维度的 2 次幂、对齐后的 2 次幂、完整维度 |
| `candidate_policy: all` | 所有满足对齐要求的整数 tile，外加完整维度；大矩阵会很慢 |
| `tile_m_candidates` / `tile_n_candidates` / `tile_k_candidates` | 非空正整数列表，覆盖该轴的自动候选 |
| `alignment_m` / `alignment_n` / `alignment_k` | 默认 1；只搜索该对齐倍数，完整维度作为尾块例外 |
| `dataflows` | 非空候选子集；默认 OS、WS、IS 全部 |

显式候选超过当前维度或不满足对齐时被过滤。无合法候选／buffer 不足返回 unsupported，
不伪造零延迟。结果记录实际候选列表、评估数、可行数、容量剪枝数。
相同目标值时依次选择更低延迟、更少总流量、更小 buffer，再按固定顺序打破平局。

**最优仅限已记录的候选集合和当前解析成本模型。** 即便使用 `all`，也不包含任意多级
tiling、跨算子融合、跨调用驻留或任意空间映射，不能宣称全架构全局最优。

## 3. buffer 与数据流

位宽为 a_bits/b_bits/acc_bits，按每次真实 tile 传输取整字节：

```text
pack(elements, bits) = ceil(elements * bits / 8)
buffer = pack(m*n, acc_bits) + copies * (pack(m*k, a_bits) + pack(k*n, b_bits))
copies = 2 if overlap_compute_memory else 1
```

global buffer 同时容纳一个 partial sum tile 和 A/B tile；启用理想计算／搬运重叠时，
A/B 采用双缓冲。尾块只传输有效元素。低位宽采用逻辑 packed 存储假设，不代表宿主
PyTorch float32 tensor 的物理存储，也不代表稀疏压缩。

| 数据流 | 外→内循环 | 保留数据 | K 多轮时的 partial sum |
|---|---|---|---|
| output_stationary | m,n,k | 输出 tile 跨 K | 片上完成全部归约 |
| weight_stationary | n,k,m | weight tile 跨 M | 每轮 K 的各输出 tile 写回，下轮读入 |
| input_stationary | m,k,n | activation tile 跨 N | 每轮 K 的各输出 tile 写回，下轮读入 |

WS 的 M 只有一个 tile、或 IS 的 N 只有一个 tile 时，同一输出归约连续，免除溢出。
其他情况，若 K 有 L 个 tile，所有输出的 partial sum 额外读和额外写各为
`(L-1) * sum(pack(m_tile*n_tile, accumulator_bits))`，最后输出按 output.bits 写入一次。
这避免把“权重只读一次”误判为没有代价的最优方案。

这些 stationary 名称指 global buffer 级时间复用；不意味着 PE 寄存器／阵列内部的
空间数据流。阵列仍采用现有统一计算周期公式：

```text
sum(batch * ceil(m_tile/pe_rows) * ceil(n_tile/pe_cols) * (k_tile+pipeline_cycles))
```

每个广播 batch 独立处理，不利用广播 operand 跨 batch 复用。

## 4. 带宽与时间

activation/weight 带宽未指定时回退到 `dram_bytes_per_second`。
输出和 partial sum 读写使用 `dram_bytes_per_second`。

```text
tA = activation_read_bytes / activation_bandwidth
tB = weight_read_bytes / weight_bandwidth
tOther = (partial_sum_read_bytes + dram_write_bytes) / dram_bandwidth

shared:      memory_seconds = tA + tB + tOther
independent: memory_seconds = max(tA, tB, tOther)
```

shared 表示串行服务；两种带宽可表示同一接口服务不同数据类型的有效速度。
independent 表示 A、B、其余流量三条独立通路的理想充分重叠下界，必须显式提供 A/B 带宽。
它不适用于实际共享同一物理 DRAM 链路的情况。

`overlap_compute_memory: false` 时延迟为计算时间＋搬运时间；true 时取两者最大值，
同时要求 A/B 双缓冲容量。它是理想化下界，不是流水调度仿真。
当前不包含 SRAM 端口／带宽、NoC、burst/stride、DMA 启动、bank 冲突、量化／scale／bias
开销与稀疏编码。outlier/mixed-precision/BitNet 等路径继续保留为 unsupported。

## 5. 保存内容

| 文件 | 内容 |
|---|---|
| `trace.jsonl` | manifest 含硬件和 search 设置；原始事件可重放 |
| `mappings.jsonl` | 开启搜索时每事件一条；物理层身份、阶段、step、shape、精度、mapping、循环顺序、分项流量、时间、候选集合和 cache_hit |
| `operators.csv` | 所选 tile/dataflow 和分项流量／memory_seconds，最终估算使用同一 mapping |
| `summary.json` | 聚合估算、unsupported 原因、唯一搜索 workload 数和缓存命中数 |

例如 A=256、B=1024、C=4096，输入／权重 16 bit、输出／累加 32 bit，1 MiB buffer：
默认候选中可选到 IS `tile_m=256, tile_n=16, tile_k=1024`。
整个 activation 驻留，流式遍历权重，DRAM 总流量为 `0.5+8+4=12.5 MiB`，达到该独立
GEMM 的首次读取／最终写出下界。它优于仅比较固定 tile 的 20/36.5/16 MiB 三个方案。

## 6. 验证与官方工具

```bash
python -m unittest discover -s tests -p 'test_hardware.py' -v
python -m unittest discover -s tests -p 'test_timeloop.py' -v
```

测试覆盖手算三种流量、accumulator 溢出、尾块位打包、batch、带宽驱动选择、
小矩阵完整整数空间参考枚举、双缓冲容量、缓存及在线事件／离线重放结果一致性。

多级存储与空间映射可进一步接入官方工具：

- [Timeloop mapper](https://timeloop.csail.mit.edu/v4/input-formats/mapper)
- [官方 Python 前端](https://github.com/Accelergy-Project/timeloop-python)
- [安装与依赖](https://timeloop.csail.mit.edu/v4/installation)
