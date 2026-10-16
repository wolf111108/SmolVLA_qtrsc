# FPGA / HDL 硬件开发入口

本目录存放**可综合 HDL 设计**、testbench、SystemC/TLM 示例与仿真运行脚本。与
[`src/vla_tcs2/hardware/`](../src/vla_tcs2/hardware/) 是两回事：后者是 Python 侧的
算子事件采集与性能估算，两者目前没有自动连接。

## 目录结构

```text
hardware/
├── rtl/                          # 可综合 RTL
│   └── tiny_gemm.sv              # 固定尺寸有符号 INT8 矩阵乘加引擎
├── tb/                           # 验证代码（不可综合）
│   ├── tiny_gemm_mem.sv          # A/W 只读、C 可写存储模型
│   ├── tb_tiny_gemm.sv           # 自检查 testbench（含参考模型与协议监视）
│   └── tb_dma_cmd.sv             # DMA 命令协议入门示例
├── models/
│   └── dma_cmd_model.sv          # DMA 命令行为模型（教学用，非真实 DMA）
├── scripts/
│   ├── run_tiny_gemm.sh          # tiny_gemm 仿真入口（iverilog 或 xrun）
│   └── run_dma_cmd.sh            # DMA 示例仿真入口（iverilog）
├── systemc/                      # SystemC/TLM 自检查 C++ 示例（Linux）
│   ├── examples/                 # counter.cpp（含 VCD）/ tlm_memory.cpp
│   ├── scripts/run_examples.sh   # SystemC 示例编译运行入口
│   ├── README.md                 # 环境安装与模型讲解
│   └── VALIDATION.md             # 验证记录
├── docs/
│   └── tiny_gemm_v0.1_spec.md    # tiny_gemm 接口与数值规格（验收依据）
├── README.md
└── .gitignore
```

生成物一律落在 `build/` 下，不入库。

---

## 1. tiny_gemm —— 固定尺寸矩阵引擎 v0.1

### 计算内容

$$C_{m,n}=\sum_{k=0}^{7}A_{m,k}W_{k,n}$$

A 为 4×8、W 为 8×4、C 为 4×4，全部行优先。A/W 是 8 位有符号补码，乘积 16 位有符号，
累加和 32 位有符号。

v0.1 范围刻意收窄，只做一件事：**用一个乘法器和一个累加器，把「读数据 → 计算 → 写回 →
完成」的全过程做清楚。** 单时钟域、同步低有效复位、单任务、固定尺寸。
**不含** bias、scale、FP8、稀疏跳零、DMA、AXI、多 PE、运行时可变尺寸。

完整接口时序、地址公式、状态机与验收条件见
[`docs/tiny_gemm_v0.1_spec.md`](docs/tiny_gemm_v0.1_spec.md)。

### 模块边界

**A/W/C 存储模型在 DUT 外部**（`tb/tiny_gemm_mem.sv`）。DUT 只负责地址生成与
「读等待、写握手」协议，因此将来把 TB 存储模型换成片上 RAM 封装时，矩阵计算逻辑不用改。

| 模块 | 职责 | 可综合 |
|---|---|---|
| `tiny_gemm` | 状态机、循环计数、地址生成、乘加、写回 | 是 |
| `tiny_gemm_rom` | 只读存储模型：请求/返回两次握手，单笔未完成 | 否 |
| `tiny_gemm_cmem` | C 存储模型：写握手即提交，无写响应通道 | 否 |
| `tb_tiny_gemm` | 激励、参考模型、协议检查、结果检查、超时 | 否 |

### 关键设计点

- **A/W 读取串行化**：`发 A 请求 → 收 A 数据 → 发 W 请求 → 收 W 数据 → 乘加`。
  牺牲吞吐换取状态机可读性；后续可改成并行发请求，外部接口不变。
- **乘积先做符号扩展再运算**：A/W 先扩展到 16 位再相乘，避免把 `8'hFF`（−1）读成 255，
  也避免 `8'h80 × 8'h80` 被截断。
- **累加结果延迟一拍驱动总线**：最后一次累加完成后，下一周期才输出累加器，
  防止非阻塞赋值导致写出的结果少加最后一项。
- **地址是元素索引不是字节地址**，且 `a_addr={m,k}`、`w_addr={k,n}`、`c_addr={m,n}`
  正好是拼接运算，布局在端口映射上直接可见。
- **完成握手后一拍才恢复 `cmd_ready`**：避免外部保持 `cmd_valid=1` 时误启动第二个任务。

### 运行

需要一个 HDL 仿真器：**Icarus Verilog**（`iverilog` + `vvp`）或 **Cadence Xcelium**（`xrun`）。
不需要 Python、GPU 或模型 checkpoint。未指明仿真器时脚本自动探测；只有 Xcelium 时，
脚本会在 `xrun` 不在 PATH 上时自动 `module load cadence/xcelium/2203_01`。

```bash
# 默认跑两个场景：正常场景必须 PASS，注入错误的场景必须 FAIL
bash hardware/scripts/run_tiny_gemm.sh

# 只跑正常场景
SCENARIO=positive bash hardware/scripts/run_tiny_gemm.sh

# 只跑 checker 自检（故意改坏一个预期值），退出码非零即正确
SCENARIO=fault bash hardware/scripts/run_tiny_gemm.sh

# 强制指定仿真器 / 换 Xcelium module / 顺手存波形
SIM=xrun XCE_MODULE=cadence/xcelium/2203_01 DUMP=1 bash hardware/scripts/run_tiny_gemm.sh
```

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `SIM` | `auto` | 强制 `iverilog` 或 `xrun`，默认自动探测 |
| `SCENARIO` | `all` | `all` / `positive` / `fault` |
| `XCE_MODULE` | `cadence/xcelium/2203_01` | Xcelium 的 module 名 |
| `DUMP` | `0` | 置 `1` 额外导出 VCD |

输出写入 `hardware/build/tiny_gemm/`：`run_positive.log`、`run_fault.log`、
（iverilog 下）`compile_*.log`、`sim.vvp`、可选 `tb_tiny_gemm.vcd`。

脚本判据：

- 正常场景日志必须包含 `PASS: tiny_gemm all tests passed`，且不得出现 `ERROR:` 行
- 自检场景日志**不得**出现 PASS 标记，且必须包含 `ERROR: TB_FAULT_INJECT`

第二个判据是回归有效性的证明：如果 checker 连故意改坏的预期值都发现不了，
正常场景的 PASS 就没有意义。

### 测试覆盖

| # | 场景 | 检查重点 |
|---:|---|---|
| 1 | 全零 | C 全零 |
| 2 | 全一 | 每个 C 都等于 8 |
| 3 | 正负混合 | 有符号计算正确 |
| 4 | 极值（−128 / 127） | 乘积位宽与符号扩展正确 |
| 5 | 地址定向 | 行列索引与行优先布局正确 |
| 6 | 随机矩阵（固定 seed 并打印） | 全部输出匹配参考模型 |
| 7 | A/W 请求等待 + 返回延迟 | 不丢请求、不重复累加 |
| 8 | C 写反压 + 完成反压 | 写数据稳定；完成通知保持 |
| 9 | 连续三任务 | 累加器与循环状态正确清空 |
| 10 | 读取阶段复位 | 旧任务取消，无残留完成通知 |
| 11 | 写回阶段复位 | 同上，且能从部分写入中恢复 |
| 12 | 复位后恢复 | 新任务结果正确 |

monitor 持续检查：无命令不得写 C 或报完成；等待中的发送端字段必须保持；
完成握手的同一拍不得再接受命令；完成通知不得与 C 写同拍出现；单任务超过
10,000 周期即判失败。

在**未被复位中断**的任务上还会逐项断言精确握手次数：命令 1 次、A 读请求/返回各 128 次、
W 读请求/返回各 128 次、C 写 16 次、完成 1 次。

日志尾部会打印上一任务的「命令握手→最后一次 C 写」周期数、等待完成被接收的周期数，
以及各类等待周期数。**执行延迟与等待完成通知的周期数分开记录**，否则调整 `done_ready`
会让计算性能看起来发生变化。

> 第一版**只记录、不设目标 MHz**。随机等待上限 8 周期、单任务 10,000 周期超时是测试条件，
> 不是引擎对任意外部停顿都能按时完成的承诺。

### 复位语义的重要限制

复位规则要求 **DUT 与存储模型共同取消未完成事务**：DUT 丢弃待发请求与完成通知，
存储模型丢弃尚未返回的请求。被中断的任务可能已经写入部分 C，这些内容视为无效，
下一条完整任务会重写全部 16 个结果。

这套规则只在「没有外部未完成事务」的模型上成立。接真实总线后必须重新定义
取消或排空机制，不能只清空控制状态而把仍在互连上的请求丢掉。

---

## 2. DMA 命令协议入门示例

[`models/dma_cmd_model.sv`](models/dma_cmd_model.sv) 是一个**行为模型 stub，不是真实 DMA**：
不访问 AXI / DDR / SRAM，也无法验证数据正确性。它只用来演示命令/响应握手的
valid/ready 规则、单笔未完成事务、反压稳定性与复位取消。

[`tb/tb_dma_cmd.sv`](tb/tb_dma_cmd.sv) 是自检查 testbench：驱动、协议监视、
独立预期响应队列、复位检查与全局超时。

```bash
bash hardware/scripts/run_dma_cmd.sh
```

需要 `iverilog` + `vvp`（该脚本暂未接入 Xcelium 路径）。输出到
`hardware/build/dma_cmd/`。正常结束必须包含：

```text
PASS: commands=7 responses=6 reset_aborted=1
```

协议约定：

- 共同上升沿采样时钟；同步低有效复位。testbench 在下降沿驱动输入。
- `cmd_valid && cmd_ready` 表示接收命令；`rsp_valid && rsp_ready` 表示消费响应。
- 被阻塞时发送方保持 valid 与载荷稳定。
- 每个模块最多一条未完成命令；响应等待期间 `busy=1`、`cmd_ready=0`。
- 响应消费后，随后周期才允许接收新命令。
- `dir=0` 为 LOAD，`dir=1` 为 STORE；region 0/1/2 对应 A/W/C。
- 仅允许 LOAD A、LOAD W、STORE C。零长度返回 `0x01`，不支持的组合返回 `0x03`，
  合法命令返回 `0x00`；不支持的组合优先于零长度检查。

Vivado 用户可把 `.sv` 文件加入 Simulation Sources，将 `tb_*` 设为仿真顶层。

---

## 3. SystemC/TLM（Linux）

环境安装、编译命令和模型讲解见 [`systemc/README.md`](systemc/README.md)。
该目录提供时钟计数器（含 VCD 波形）和 TLM 内存读写两个独立的自检查 C++ 示例。
按教程安装 SystemC 后，从仓库根目录执行：

```bash
bash hardware/systemc/scripts/run_examples.sh
```

验证记录见 [`systemc/VALIDATION.md`](systemc/VALIDATION.md)。这些示例与上方 HDL
模型分别运行，当前均未自动连接 Python 算子采集框架。

---

## 4. 工具与环境

### 本机可用工具

`hardware/` 下的脚本都**不依赖 Python / GPU / conda 环境**。本机（`gpupro6000e`）现状：

| 工具 | 状态 | 用法 |
|---|---|---|
| Cadence Xcelium 22.03 | 已安装，需 module | `module load cadence/xcelium/2203_01` 后用 `xrun` |
| Icarus Verilog | **未安装** | 若安装则 `iverilog` + `vvp`，无需 module |
| Verilator / VCS / ModelSim | 未验证 | — |

`run_tiny_gemm.sh` 会先 `source /usr/share/Modules/init/bash`（若 `module` 不在 PATH），
再按 `iverilog` → `xrun` 顺序探测。所以本机直接跑：

```bash
bash hardware/scripts/run_tiny_gemm.sh
```

### Xcelium 调用方式

```bash
module load cadence/xcelium/2203_01
xrun -64bit -sv -access +rwc -timescale 1ns/1ps \
     -xmlibdirname build/tiny_gemm/xcelium.d \
     -top tb_tiny_gemm \
     rtl/tiny_gemm.sv tb/tiny_gemm_mem.sv tb/tb_tiny_gemm.sv
```

要点：

- `-sv` 按 SystemVerilog 编译；`-top` 指定仿真顶层，避免误选被例化的模块。
- `-xmlibdirname` 把编译库收进 `build/`，保持源码树干净。
- `-timescale` 与源码里的 `` `timescale 1ns/1ps `` 保持一致，避免三方文件时间单位不一致。
- 注入错误用 `+define+TB_FAULT_INJECT`，导出波形用 `+define+TB_DUMP`。

### 与 IC_flow 参考流程的关系

`/home/lfwang/lrn/IC_flow` 是一套完整的数字前端参考流程（RTL simulation →
regression → SpyGlass lint / CDC / RDC → Design Compiler 综合 → Formality LEC →
PrimeTime STA → 零延迟 GLS → SDF GLS），已经处理过本机服务器上的工具兼容问题
（SpyGlass freetype、PTS 旧版库、PrimeTime 只支持 SDC 2.0、SDF annotation 需用
`-sdf_cmd_file` 等）。

`hardware/` 目前**独立运行**，不依赖 IC_flow 的 Makefile。后续要把 `tiny_gemm`
接进完整前端流程时，可复用其目录约定与以下经验：

| IC_flow 中的做法 | 迁移到本目录时的对应物 |
|---|---|
| `dv/filelists/rtl.f`、`tb.f` | 建 `hardware/dv/filelists/`，把 `run_tiny_gemm.sh` 的文件列表抽出去 |
| `constraints/clock.sdc` 等 | `tiny_gemm` 单时钟，只需一条 `create_clock` + reset 约束 |
| SpyGlass lint / CDC / RDC | 单时钟域设计，CDC/RDC 主要用于验证流程跑通 |
| DC 综合 + Formality LEC | 综合前需确认 RTL 无 latch、无组合环 |
| PrimeTime STA | 需要标准单元库（本机 SMIC55 LL HD RVT） |
| 零延迟 / SDF GLS | 需要门级网表与标准单元 Verilog 模型 |

> 迁移时注意：IC_flow 里 `make gls_sdf` 用的是 `xrun -sdf_cmd_file`，而不是 testbench
> 内部的 `$sdf_annotate`——后者会被 Xcelium 忽略并报 `CUSSTI`。

---

## 5. 编码约定

- 统一 `` `timescale 1ns/1ps ``。
- 所有手写接口握手上所有字段用 `logic` 显式位宽；不使用隐式宽度推断。
- 有符号运算显式声明 `signed`，并显式写符号扩展。
- 组合逻辑赋给 `wire`/`assign`，时序逻辑统一用非阻塞赋值。
- testbench 在**下降沿驱动、上升沿采样**，避免与 DUT 在上升沿采样形成 race。
- testbench 比较结果用 `!==`，让 X/Z 被判为失配而不是静默通过。
- 每个测试都有全局超时，避免死锁导致仿真挂死。
