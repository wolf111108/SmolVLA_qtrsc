# FPGA/HDL 硬件开发入口

根目录 `hardware/` 放置 HDL 行为模型、testbench 和仿真运行脚本。
现有 [`src/vla_tcs2/hardware/`](../src/vla_tcs2/hardware/) 继续负责 Python
算子事件采集与性能估算。两者当前没有自动连接。

本次加入的是 **DMA 命令/响应协议入门示例**，不是可上板的 DMA 实现。
当前没有 AXI 主接口、DDR/BRAM 数据搬运、矩阵计算或量化数值验证。

## 文件

| 文件 | 用途 |
|---|---|
| [`models/dma_cmd_model.sv`](models/dma_cmd_model.sv) | 独立 DMA 行为模型：接收命令，延迟若干周期后返回状态 |
| [`tb/tb_dma_cmd.sv`](tb/tb_dma_cmd.sv) | 自检查 testbench：驱动、协议监视、预期响应及复位检查 |
| [`scripts/run_dma_cmd.sh`](scripts/run_dma_cmd.sh) | Icarus Verilog 编译与仿真入口 |
| `.gitignore` | 忽略本目录生成的仿真文件与波形 |

## 运行

需要 Bash、Icarus Verilog 的 `iverilog` 和 `vvp`，不需要 PyTorch、GPU 或模型 checkpoint。
从仓库根目录执行：

```bash
bash hardware/scripts/run_dma_cmd.sh
```

脚本通过自身位置定位文件，因此也可在其他工作目录用绝对路径调用。
可用 `IVERILOG=/path/to/iverilog VVP=/path/to/vvp` 指定工具路径。
默认输出到 `hardware/build/dma_cmd/`：`compile.log`、`run.log`、`sim.vvp`、
`tb_dma_cmd.vcd`。再次运行会更新这些本地调试输出；该目录不入库。

编译或仿真失败会返回非零退出码。缺少工具返回 127。正常结束还必须包含如下标记：

```text
PASS: commands=7 responses=6 reset_aborted=1
```

它表示 7 条已接收命令，其中 1 条被测试复位取消，6 条完成响应被接收。
**这是预期输出，不是已获得的仿真证据。添加本目录时的环境没有 HDL 仿真器，
因此未完成 HDL 编译与运行；仅检查了脚本语法及缺少依赖时的退出行为。**

Vivado 用户可将两个 `.sv` 文件加入 Simulation Sources，将 `tb_dma_cmd`
设为仿真顶层并运行行为仿真至结束。Testbench 不是可综合模块；行为模型也不能作为
真实数据通路的替代品。请将工具生成目录放在 `hardware/build/` 下。

## 协议约定

- 共同上升沿采样时钟；同步低有效复位。testbench 在下降沿驱动输入。
- `cmd_valid && cmd_ready` 表示接收命令；`rsp_valid && rsp_ready` 表示消费响应。
- 命令或响应被阻塞时，发送方保持 valid 与载荷稳定。
- 每个模块最多一条未完成命令。响应等待期间 busy 保持为 1、cmd_ready 为 0。
- 响应消费后，随后周期才允许接收新命令。接收命令与完成命令是不同事件。
- `dir=0` 为 LOAD，`dir=1` 为 STORE；region 0/1/2 对应 A/W/C。
- 教学模型仅允许 LOAD A、LOAD W、STORE C。零长度返回 `0x01`，不支持的组合返回
  `0x03`，合法命令返回 `0x00`；不支持的组合优先于零长度检查。
- 外存地址 64 位、本地字节偏移与字节数各 32 位、region 2 位、状态码 8 位。
- 模型默认延迟 4 个 RUN 周期，可用 `LATENCY` 参数调整且必须大于等于 1；
  testbench 不要求精确完成周期，仅设全局超时。

## 测试范围与阅读顺序

1. TEST 1：普通命令，理解 `send_cmd` 与 `take_rsp` 是两次独立交接。
2. TEST 2：完成响应被阻塞时，同时提交下一条命令；检查反压下的稳定性和任务隔离。
3. TEST 3：零长度命令返回参数错误。
4. TEST 4：STORE A 返回不支持错误。
5. TEST 5：取消执行中的模型任务，确认复位后不遗留旧响应并能执行新命令。

监视器按握手计数，而不是按 valid 持续周期计数；还检查无任务不得返回响应、
响应等待时 busy 有效、命令和响应反压期间字段稳定。响应状态以预期队列检查。
接收后主动改变输入字段，用于发现实现错误地使用未锁存参数的问题。

行为模型仅有方向、region、长度影响响应结果。地址和偏移虽然被捕获，
本例不验证它们的数据搬运效果，也不验证范围检查、字节使能、AXI 突发、
缓冲提交时序、实际吞吐或数值精度。通过本测试不能证明真实 DMA 正确。

复位测试只适用于当前没有外部事务的模型。真实 DMA 复位必须排空事务或协调平台复位，
不能只清空控制状态而丢弃仍在互连上的请求。

## 后续扩展

- 替换模型为真实 DMA 后，补充外存/片上存储模型与逐字节数据检查，检查目标范围外不被改写。
- 添加矩阵引擎时，复用命令与响应握手规则，并加入独立数值参考和 A/W/C 缓冲模型。
- 验证顶层调度器时，将调度器设为 DUT，把 DMA/GEMM 行为模型当作从属执行模块。
- 本目录是硬件源码与通用单元验证入口；正式性能/模型实验仍遵循
  [`experiments/README.md`](../experiments/README.md)，设置集中在相应 `experiment_setup.md`。
