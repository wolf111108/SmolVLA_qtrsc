# 组件化硬件建模协议 v1

本模块将硬件能力、算子数据流与资源调度分开。它复用已有 `OperatorEvent`、
`HardwareManager` 和 JSONL trace，在 `component_sim` 后端中提供可组合的 tile 级仿真。
这是**假设硬件上的 GEMM 核心模型**，结果不代表整次 SmolVLA 推理的端到端延迟。

## 快速运行

不需要 PyTorch、GPU、SystemC，也不增加第三方依赖：

```bash
python scripts/demo_hardware_components.py --output-dir outputs/component_demo

PYTHONPATH=src python -m vla_tcs2.hardware \
  --trace outputs/component_demo/trace.jsonl \
  --output-dir outputs/component_replay

# 直接重放已有硬件 trace，替换成组件架构
PYTHONPATH=src python -m vla_tcs2.hardware \
  --trace /path/to/trace.jsonl \
  --config configs/hardware/components.json \
  --output-dir outputs/component_run

# 有界候选搜索，评分来自同一个事件模拟器
python scripts/demo_hardware_components.py \
  --config configs/hardware/components_search.json \
  --output-dir outputs/component_search
```

输出目录不能覆盖已有实验。示例 workload 是明确标注的合成矩阵 `50×129 @ 129×70`，
用于验证尾块、存储流量与运行入口；不宣称它来自真实模型。

## 文件与职责

路径前缀为 `src/vla_tcs2/hardware/`：

| 文件 | 职责 |
|---|---|
| `modeling/protocol.py` | 版本化动作、端口、资源、阶段、执行计划、结果协议 |
| `modeling/components/storage.py` | 容量与聚合读写端口模型；也用于 local/accumulator |
| `modeling/components/dram.py` | 有效带宽、固定延迟、burst 取整、共享读写总线 |
| `modeling/components/interconnect.py` | 串行 link / DMA engine |
| `modeling/components/compute.py` | 稠密阵列与显式理想跳零模型 |
| `modeling/components/__init__.py` | 组件工厂注册 |
| `modeling/architecture.py` | 配置装配、端口连接、共享资源和容量校验 |
| `modeling/planner.py` | OS 分块、依赖、GB/local 双缓冲槽与 accumulator 驻留 |
| `modeling/mapper.py` | 复用现有 tile 候选规则，用事件模拟器搜索 |
| `modeling/runtime.py` | tensor-free 在线统计及可选 raw 输入零值采集器 |
| `modeling/simulator.py` | 确定性事件驱动调度；每次调用创建全新状态 |
| `backends/component.py` | 适配旧 capture/report API |

旧 `dense_analytic` 及 `timeloop.py` 保持可用。项目的 `timeloop.py` 是自研解析 mapper，
不是官方 Timeloop 的 Python 包。它原有的 WS/IS 策略尚未移植到组件 planner。

## 组件文件规范

每个新组件文件应定义：

1. `XxxSpec`：`frozen=True` dataclass，单位写进字段名，在构造时拒绝非法参数。
2. `XxxComponent`：实现 `HardwareComponent` 协议。它只负责校验和产生资源工作量。
3. 动态队列、时间、资源占用属于 `EventSimulator.run()` 的局部状态；组件不得修改全局时钟。

```python
class HardwareComponent(Protocol):
    id: str
    def ports(self) -> tuple[PortSpec, ...]: ...
    def resources(self) -> tuple[ResourceSpec, ...]: ...
    def validate(self, action: Action) -> None: ...
    def lower(self, action: Action) -> ActionModel: ...
```

`ActionModel.phases` 是串行执行的阶段。每个阶段包括固定周期和一组资源需求。
`ResourceDemand.amount` 是非负整数；`unit` 必须与对应 `ResourceSpec.unit` 一致。
例如：`bytes` 对应 `bytes/cycle`，阵列在 v1 中将计算工作换算成 `cycles`。

插件通过工厂注册，不修改模拟器：

```python
from vla_tcs2.hardware.modeling import register_component
register_component("my_sram", lambda name, params, clock_hz:
                   MySRAM(name, MySRAMSpec(**params)))
```

自定义组件需要兼容所替换角色的动作与 spec 属性。新增算子语义由新的 planner 实现。
每个物理资源必须只有一个全局唯一 ID；不能把同一条 DRAM 总线声明为多个独立资源。

## 动作组合与调度语义

`PlanEvent` 包含 `id/actions/depends_on`。同一事件内的 actions 表示一笔同步流式事务：
例如 DRAM read、DMA transfer、GB write 同时参与一次 tile 搬运。

- 同一个物理资源上的需求先相加；不同资源的服务时间取最大值。
- 一个阶段耗时为 `sum(fixed_latency) + max(ceil(work / rate))`。
- 阶段启动时一次性获取全部资源，完成时全部释放。
- 固定延迟期间也保留资源；该规则保守地限制了 outstanding 并行。
- 多参与者事务当前要求每个参与者只有一个阶段。单个组件动作支持多个阶段。
- 依赖完成的事件才进入 ready 集合。绝不提前为未 ready 事件预约未来资源。
- 同时间先完成并释放所有事件，再按计划中的稳定顺序调度 ready 事件。
- 独立资源可以重叠，共享资源非抢占串行服务。不支持流体带宽共享。
- 空计划返回零时间；零工作阶段可在同周期完成；依赖环、重复 ID、非法单位直接报错。

因此，DMA 与 DRAM 同时参与一笔传输时，服务时间不会按两次外存传输相加。
跨层访问量分别记账是合理的：DRAM read 64 B 与 GB write 64 B 属于两个存储层的访问。
这些计数不能相加后称作“DRAM 流量”。

## 架构配置

完整例子见 `configs/hardware/components.json`。根字段：

```json
{
  "schema_version": 1,
  "clock_hz": 1000000000,
  "components": {},
  "connections": [],
  "roles": {}
}
```

- `components`：实例名到 `type` 与参数的映射；未知字段会被拒绝。
- `connections`：有向 `from/to/via`，端点采用 `实例.data`。需要双向路径时写两条连接。
- `roles`：planner 的 `dram/global/local/accumulator/array` 角色到实例名的映射。
- 所有内核资源服务率采用架构时钟。DRAM 的 B/s 在装配时换算到 B/cycle。
- GB、local、accumulator 是三个独立容量预算；结果分别报告预留容量。

示例路径包括 DRAM↔GB、GB→local、local→array、array↔accumulator、accumulator→GB。
连接可以共享同一 link 实例，届时自然产生资源竞争。计算阶段的本地读、阵列工作和
累加器访问合并为一个同步阶段，相关链路也占用到阶段结束。

默认配置：32×32 阵列，1 GHz；DRAM 102.4 GB/s×0.7；GB 4 MiB、local 128 KiB、
accumulator 64 KiB；各聚合 SRAM 读写端口和 link 为 256 B/cycle。
数值是待扫描的实验假设，不对应已校准芯片。配置中声明支持的格式均假设 1 MAC/PE/cycle，
不自动根据 INT8/FP8/FP32 位宽改变算力。

不传显式架构时，后端用旧 `HardwareSpec` 生成默认组件。此时 GB、local、accumulator
**各自**拥有 `hardware.sram_bytes` 容量；要严格控制总 SRAM 面积应使用显式架构。
显式架构的 clock/array/storage 参数优先，旧 `hardware` 对应参数不参与计算。

## 算子与 mapping

`OperatorEvent` 继续作为算子协议，包含 Linear/MatMul 形状、精度、batch、phase、flow_step。
Planner v1 使用 output-stationary 的 `batch→M→N→K` 循环：

1. A/B 从 DRAM 经 DMA 进入 GB operand slot。
2. 通过 NoC 搬入 local slot。
3. local 经 operand link 向阵列提供数据；一次 K tile 的部分和保存在 accumulator。
4. 全部 K tile 完成后，accumulator 经 NoC 写 GB output slot，再经 DMA 写 DRAM。

一个输出 tile 的部分和始终驻留。不同输出 tile 串行执行；不跨 batch、算子或 denoise step
保存权重/KV。`double_buffer=True` 给 GB/local 各分配两个 operand slot，容量真实增加。
槽复用依赖前一次消费者完成，避免预取覆盖未使用的数据。

容量预留是整个算子期间的静态上界，结果中的 `reservations` 不代表动态分配追踪。
不够容量时返回 `unsupported`，不隐式 spill。尾块按真实大小计流量，阵列按行列波次计周期。

直接调用：

```python
from vla_tcs2.hardware.modeling import Architecture, MatmulPlanner, EventSimulator
from vla_tcs2.hardware import MappingSpec

arch = Architecture(config["backend_config"]["architecture"])
plan = MatmulPlanner(double_buffer=True).lower(event, MappingSpec(32, 32, 128), arch)
result = EventSimulator().run(plan, arch)
```

`ExecutionPlan.to_dict()/from_dict()` 可 JSON 往返，再交给模拟器重放。
`save_plan=True` 时完整计划写入 `simulations.jsonl`，默认只保存时间线和配置/统计证据。

## 在线统计与稀疏语义

`RuntimeStats` 包含 `event_id/source/scope/sample_count/activation_zero_fraction/tile_active_macs`。
通过 `event.metadata["runtime_stats"]` 保存到原有 trace，离线重放无需重新运行模型。

```python
from vla_tcs2.hardware.modeling import RuntimeStats
from dataclasses import replace

stats = RuntimeStats(event_id=event.event_id, source="measured", scope="tensor",
                     sample_count=4096, activation_zero_fraction=0.75,
                     allow_uniform_estimate=True)
event = replace(event, metadata=dict(event.metadata, runtime_stats=stats.to_dict()))
manager.submit(event)
```

以上 sample_count/零值率仅示范字段，应替换为实际测量。原始统计为 measured，并不意味着
由整层均值推导的 tile 工作量也已测量；结果会明确标记 `estimated_uniform`。

在线挂钩入口：

```python
from vla_tcs2.hardware import HardwareManager
from vla_tcs2.hardware.modeling import capture_raw_activation_zeros

with HardwareManager(output_dir, backend="component_sim", backend_config=backend_config,
                     runtime_stats_provider=capture_raw_activation_zeros).capture_model(model):
    output = model(inputs)
```

provider 签名为 `(event, module, args, kwargs, output) -> RuntimeStats`。
它在已安装的 forward hook 中、`submit()` 前执行。返回事件 ID 必须一致。
默认不扫描张量；上例显式启用 raw activation 扫描，会产生设备同步与统计成本，
该采集开销属于宿主 profiling，不加入假设加速器延迟。
`quant_forward` 的模块输入不等于内部量化操作数，内置 provider 会拒绝该模式。
量化稀疏实验需传入能取得实际量化操作数统计的自定义 provider。

阵列模式：

| mode | 稀疏处理 |
|---|---|
| `dense` | 无论零值率多少，执行稠密周期；统计仅随 trace 保存 |
| `ideal_mac_skip` | 使用 active MAC 数计算理想池化吞吐，显式排除解码/调度/不均衡成本 |

精确 tile 工作量的 key 是 `batch:m_offset:n_offset:k_offset:m_size:n_size:k_size`。
缺少统计、存在多余 tile key、事件 ID 错误或 MAC 数越界均拒绝。
整层零值率只有显式 `allow_uniform_estimate=True` 才能按均匀分布估算 tile 工作量。
此版本只实现元素零值的理想计算收益；bit-plane、块稀疏、编码/元数据流量仍需专门组件。
**所有模式使用稠密存储流量，零值率不会自动减少 DRAM 字节。**

## 搜索与结果解释

组件搜索复用 `SearchSpec` 和候选生成规则，但调用组件 simulator 评分。
仅允许 `dataflows=["output_stationary"]`；默认最多 256 个候选、每候选最多 200000 个事件。
超预算明确拒绝；容量或事件预算不满足的候选不在可行集合中。

- `latency`：完成时间。
- `dram_bytes`：DRAM 物理读写字节之和，包括 burst padding。
- `memory_time`：DRAM 资源占用时间，包括此模型的固定延迟和耦合事务占用。

使用 runtime_stats 进行搜索时标记 `oracle_runtime_stats`。这不代表芯片已实现在线自适应。
与具体分块绑定的 tile 统计仅用于固定 mapping，搜索不接受它；不同候选需要不同的 tile 测量。
搜索结果记录候选集、评价数、可行数、目标值和最优性范围。

`simulations.jsonl` 每个算子一条，包含 mapping、timeline、counters、resource_busy_cycles、
utilization、reservations、runtime_stats 和 assumptions。等待周期是各阶段 ready 后排队时间之和，
并行事件的等待可以重叠，不能拿它与 compute 时间直接相加推导总延迟。
`compute_cycles` 为阵列服务需求，资源 busy 包含被存储/链路拖长的占用，两者含义不同。

`operators.csv/summary.json` 沿用旧协议，其中 `peak_buffer_bytes` 是各独立片内预留容量之和；
`memory_seconds` 是 DRAM 占用时间；`pe_utilization` 使用实际调度 MAC 数除以阵列计算服务容量。
细分 activation/weight CSV 字段在新后端留空；总读写和各层流量以 simulations 为准。
不支持的算子保留 MAC 数和原因，延迟字段为空，绝不当作零延迟成功。

## 验证和后续扩展

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_hardware*.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_timeloop.py' -v
```

包含手算时间线、容量、尾块与广播流量、资源互斥、burst padding、双缓冲、统计语义、
插件、多阶段、搜索和 trace 重放测试。PyTorch 不可用时，GPU/真实 module hook 测试会跳过。

后续可以增加 banked SRAM、分组稀疏阵列、vector unit、codec、WS/IS planner 和跨算子驻留。
新增复杂队列/背压时可接 SimPy 后端，保留 ExecutionPlan 与结果协议。
目前没有能耗/面积参数；访问计数可以作为后续能耗模型的输入。

设计参考：
- [Python heapq](https://docs.python.org/3/library/heapq.html)：完成事件优先队列。
- [SimPy 时间调度](https://simpy.readthedocs.io/en/4.1.0/topical_guides/time_and_scheduling.html)：确定性调度参考。
- [Timeloop mapping](https://timeloop.csail.mit.edu/previous_versions/timeloop-accelergy-v3/timeloop/input-formats/mapping)：循环与存储映射参考。

初版选择标准库事件引擎，因为计划已展开为静态 DAG、需求仅为非抢占资源服务；未引入
完整 SimPy 或官方 Timeloop 依赖，也未宣称与其模型结果等价。
