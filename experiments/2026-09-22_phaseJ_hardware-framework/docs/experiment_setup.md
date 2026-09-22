# 实验设置（Experiment Setup）

- 实验名称：2026-09-22_phaseJ_hardware-framework
- 状态：draft；软件单元验证已完成，GPU rollout 待运行。
- 创建日期：2026-09-22
- 前序：Phase I vision-sparsity-compute，沿用已校准 VLIN 配置。

## 1. 实验目的

建立 `HardwareManager + OperatorEvent + dense_analytic + JSONL replay`。
保留 stat_manager 的独立生命周期，模型输出不变；不把 fake quant 的 GPU 执行时间作为目标芯片延迟。

## 2. 环境与版本

| 项目 | 要求 |
|---|---|
| 推理环境 | 当前 smolvla_eval / torch / LeRobot / MuJoCo |
| 离线重放 | Python >= 3.10，仅标准库 |
| 软件验证 | python -m unittest discover -s tests -p test_hardware.py -v |
| 版本证据 | git commit、依赖版本、checkpoint revision、scale 来源/哈希 |

main.py 在 manifest 记录 source_commit 与完整配置 SHA256；原配置保存在父输出目录。
示例保留前序 `revision: null`，正式比较应固定 checkpoint revision 并记录 scale 版本。

## 3. 模型与数据

| 组 | 模型/精度 | 校准 |
|---|---|---|
| HW0 | raw，采集注册的 nn.Linear | 无 |
| HW1 | VLIN FP8 + outlier；预期296量化Linear + 64量化MatMul，另采集未量化Linear | 强制 --skip-calibration，复用 VLIN scale |

模块数是前序配置的静态预期，不是本轮实测。检查 attached_modules、captured_module_calls、attached_but_unobserved。
HW0 原生 attention 不自动替换为量化 MatMul。

## 4. 评测协议

| 项目 | 值 |
|---|---|
| suite/task | libero_goal/task0 |
| episodes/seed | 1 / 1000 |
| n_action_steps / num_steps | 10 / 10，与VSC-1一致 |
| capture | 最前20000个支持的算子调用 |
| 指标 | 调用数、MAC、shape/format、覆盖、GEMM core cycles/traffic/latency |

max_events=0 表示全部采集。默认截断可能发生在 generation 中间，不能除以 generation 数宣称完整单次推理成本。
事件/CSV流式写入，不持有 tensor 或完整事件列表；聚合按模块/phase/flow_step，generation集合随采集数增长。

## 5. 实验变量与分组

| 组 | 配置 | 固定项 | 用途 |
|---|---|---|---|
| HW0 | hw0_raw.yaml | raw模型/任务/seed | Linear采集与稠密core估算 |
| HW1 | hw1_vlin_trace.yaml | VSC-1量化/scale/协议 | 量化trace与unsupported检查 |
| Replay | replay_32x32.json | HW0 trace | 阵列和M/N tile从16改为32 |

HW0/HW1覆盖不同，HW1 outlier尚未建模，禁止直接比较其latency得出量化加速比。
Replay是显式联合映射方案比较，不是单变量消融。

## 6. 运行命令

仓库根目录：

```bash
conda activate smolvla_eval
python -m unittest discover -s tests -p test_hardware.py -v
bash experiments/2026-09-22_phaseJ_hardware-framework/scripts/run_capture.sh raw
bash experiments/2026-09-22_phaseJ_hardware-framework/scripts/run_capture.sh vlin

# 默认复用源manifest硬件/映射，totals应一致。
PYTHONPATH=src python -m vla_tcs2.hardware \
  --trace outputs/2026-09-22_phaseJ_hardware-framework/hw0_raw/hardware/trace.jsonl \
  --output-dir outputs/2026-09-22_phaseJ_hardware-framework/replay_same

PYTHONPATH=src python -m vla_tcs2.hardware \
  --trace outputs/2026-09-22_phaseJ_hardware-framework/hw0_raw/hardware/trace.jsonl \
  --config experiments/2026-09-22_phaseJ_hardware-framework/configs/replay_32x32.json \
  --output-dir outputs/2026-09-22_phaseJ_hardware-framework/replay_32x32
```

拒绝覆盖已有trace/report；重复运行修改为同实验下新的输出子目录。
runner先检查torch并运行集成测试，再执行单task评测，不自动跑完整四suite。

## 7. 输出目录映射

前缀：`outputs/2026-09-22_phaseJ_hardware-framework/`。

| 配置 | 子目录 | 状态 |
|---|---|---|
| hw0_raw.yaml | hw0_raw/hardware/ | pending |
| hw1_vlin_trace.yaml | hw1_vlin_trace/hardware/ | pending |
| 源manifest | replay_same/ | pending |
| replay_32x32.json | replay_32x32/ | pending |

- trace.jsonl：manifest + schema v1算子事件，无tensor值。
- operators.csv：每算子MAC、周期、DRAM读写、buffer、状态和原因。
- summary.json：scope/assumptions、失败/截断、模块覆盖、分模块/phase/flow汇总。

estimated_core_macs与dense_macs分开；unsupported时latency/cycles为空，不是零成本。
总latency仅汇总estimated子集。重放保存源manifest及trace SHA256，completed只说明已读完存储事件，应同时检查原capture是否失败/截断。

## 8. 风险与注意事项

- 默认关闭。仅在evaluate外围attach，校准排除；异常清理hooks并输出failed summary。
- 采集nn.Linear/QuantizedLinear/QuantizedMatMul；Vision SDPA、functional MatMul、Conv、norm/softmax等未覆盖。
- hardware独立于sparsity/calibration，不替换stat_manager。
- metadata不包含activation值、outlier mask、稀疏分布。不能从该trace反推位串行/块稀疏收益或新量化方案。
- outlier、token混合精度、BitNet、test_forward保留逻辑MAC但不估算时延。
- 逻辑格式与实际容器dtype分开；disabled QuantSpec为pass-through，按实际dtype记录。
- 不建模跨算子SRAM/KV驻留；weight_id仅为扩展字段。
- Python采集有额外开销；不使用宿主执行时间作为芯片性能证据。
- 真实模型GPU验证待运行，无真实芯片校准结果。

## 9. 代码与扩展约定

| 文件 | 职责 |
|---|---|
| schema.py | TensorDesc/OperatorEvent/HardwareSpec/MappingSpec/OpEstimate |
| capture.py | 模块I/O转事件，保留batch/head与实际shape |
| interfaces.py | 后端Protocol |
| manager.py | 生命周期、hook、流式记录和聚合 |
| backends/ | 显式注册与后端公式 |
| trace.py / __main__.py | 版本校验和离线重放 |
| report.py | 口径与汇总 |

新增后端实现name/version/required_features/estimate(event,hardware,mapping)，用register_backend(name,factory)注册。
第一版仅metadata；声明额外特征的后端会被拒绝，禁止默默缺参。
下一阶段扩展版本化特征接口：在quant_methods中获取真实编码/分区，增加状态化执行模型和稀疏后端；不从累计统计字典反推。
HardwareSpec/MappingSpec暂为统一强类型契约，新硬件额外参数应明确扩展验证。

## 10. dense_analytic v1 公式与假设

假设支持的格式均为1 MAC/PE/cycle，不自动给低位宽更高吞吐。
output-stationary：输出tile在SRAM完成K归约，A/B每个输出tile、每个batch重新从DRAM加载。
每个真实尾块(m_t,n_t,k_t)：

- cycles = ceil(m_t/pe_rows) × ceil(n_t/pe_cols) × (k_t+pipeline_cycles)，累加所有tile/batch。
- read bytes = ceil(m_t×k_t×a_bits/8)+ceil(k_t×n_t×b_bits/8)，每tile累加。
- write bytes = ceil(m_t×n_t×o_bits/8)，每输出tile一次。
- buffer = A tile+B tile+accumulator；理想重叠时A/B双缓冲。
- 无重叠latency = cycles/frequency+traffic/bandwidth；理想重叠取max，属于明确的理想化下界。
- buffer超SRAM标unsupported，不自动换tile或溢出。

不含scale/bias/索引字节、重定标、SRAM带宽、互连和未显式指定的启动/排空；无能耗/面积。
逻辑MAC=batch×M×N×K；一次wrapper调用计一个逻辑算子，不计SQNR浮点参考计算。
汇总是串行GEMM core估计，不能称端到端或per-action latency。

## 11. 验收门槛

1. 3×7 @ 7×5、2×4 PE、2×4×4 tile、8bit：105 MAC、28 cycles、112 read bytes、15 write bytes、56 buffer bytes。
2. [2,1,3,7] @ [1,4,7,5]保留batch[2,4]：840 MAC。
3. 同配置在线/离线totals一致；改硬件不改逻辑MAC。
4. PyTorch raw/quantized adapter输出不变，异常后hook清理；校准排除。
5. HW1 outlier必须unsupported，不输出完整部署latency。
6. generation/phase/flow按实际context记录；与前序统计仅比较相同调用范围。
