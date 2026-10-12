# Linux 上搭建 SystemC/TLM 环境与运行示例

本教程接续“硬件仿真平台比较”中的 SystemC/TLM 讨论，提供从安装、编译、
运行到理解模型的完整路径。默认使用 **SystemC 3.0.2 + C++17 + CMake**，
安装在用户目录。TLM 和 `tlm_utils` 已随官方 SystemC 源码提供，不用再安装一个 TLM 库。

示例只依赖 C++ 工具链，不依赖 Python、PyTorch、CUDA、LIBERO 或模型权重。
它们是独立教学模型；当前没有读取本仓库的算子 trace，也没有连接 HDL DMA 示例。

## 1. SystemC 在系统中的位置

SystemC 是 C++ 库：模块用 C++ 描述，仿真内核管理进程、事件、信号更新和仿真时间。
最终运行的是普通 Linux 可执行程序，入口为 `sc_main()`。

| 建模方式 | 模块如何交互 | 本目录示例 | 适合回答的问题 |
|---|---|---|---|
| 时钟/信号模型 | 端口与 `sc_signal`，时钟边沿触发 | `counter.cpp` | 每个边沿发生什么、信号何时更新 |
| TLM 事务模型 | socket 传递读写事务，标注延迟 | `tlm_memory.cpp` | 请求是否正确、模型指定的延迟如何推进时间 |

SystemC 的时间精度取决于你写的模型。TLM 不会自动变成周期精确的总线模型；
软件运行耗时也不等于模拟硬件耗时。此处内存的 10 ns 是显式设定的教学参数。

## 2. 安装 Linux 编译工具

Ubuntu / Debian：

```bash
sudo apt update
sudo apt install -y build-essential cmake git
```

Fedora / Rocky Linux 等使用 DNF 的发行版可安装对应工具：

```bash
sudo dnf install -y gcc gcc-c++ make cmake git
```

确认工具版本：

```bash
g++ --version
cmake --version
git --version
```

本示例要求 CMake 至少 3.16；SystemC 3.x 使用 C++17。
建议使用 GCC 9.3 或更新版本。无 sudo 权限的服务器应使用管理员已安装的工具链，
或自己的用户级工具链；后续 SystemC 的编译和安装不需要 sudo。

## 3. 从官方源码安装 SystemC 3.0.2

固定版本便于复现。下面的 `git clone` 用于首次安装，下载目录应尚不存在。
命令可在任意工作目录执行：

```bash
export SYSTEMC_VERSION=3.0.2
export SYSTEMC_PREFIX="$HOME/.local/opt/systemc-$SYSTEMC_VERSION"
export SYSTEMC_SOURCE="$HOME/.local/src/systemc-$SYSTEMC_VERSION"
export SYSTEMC_LIB_BUILD="$HOME/.local/build/systemc-$SYSTEMC_VERSION"
mkdir -p "$HOME/.local/src" "$HOME/.local/build"

git clone --depth 1 --branch "$SYSTEMC_VERSION" \
  https://github.com/accellera-official/systemc.git "$SYSTEMC_SOURCE"

cmake -S "$SYSTEMC_SOURCE" -B "$SYSTEMC_LIB_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DBUILD_SHARED_LIBS=ON \
  -DENABLE_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="$SYSTEMC_PREFIX"

cmake --build "$SYSTEMC_LIB_BUILD" --parallel 2
cmake --install "$SYSTEMC_LIB_BUILD"
```

`--parallel 2` 限制编译并发，便于在共享服务器或小内存机器上运行；资源充足可增加。
官方 tag 检出后显示 detached HEAD 属于正常现象。

| 参数 | 作用 |
|---|---|
| `-S` / `-B` | 分开源码目录与编译目录 |
| `CMAKE_INSTALL_PREFIX` | 把头文件、库、CMake 配置安装到用户目录 |
| `CMAKE_CXX_STANDARD=17` | 固定库使用的 C++ 标准，示例从库配置读取同一标准 |
| `BUILD_SHARED_LIBS=ON` | 构建 Linux 动态库 `libsystemc.so` |
| `ENABLE_EXAMPLES=OFF` | 省略官方示例的编译，本仓库示例稍后单独编译 |

安装后检查：

```bash
find "$SYSTEMC_PREFIX" -name SystemCLanguageConfig.cmake -print
find "$SYSTEMC_PREFIX" -name 'libsystemc.so*' -print
test -f "$SYSTEMC_PREFIX/include/systemc"
test -f "$SYSTEMC_PREFIX/include/tlm"
```

库目录因系统配置不同会使用 `lib` 或 `lib64`，不要把它写死在编译命令中。
`SYSTEMC_PREFIX` 是本教程约定的安装路径变量；实际由下面的 CMake 参数传入，
单独设置这个变量不会让所有外部程序自动找到 SystemC。

新终端可重新执行：

```bash
export SYSTEMC_PREFIX="$HOME/.local/opt/systemc-3.0.2"
```

## 4. 编译和运行本仓库示例

进入 `SmolVLA_qtrsc` 仓库根目录：

```bash
bash hardware/systemc/scripts/run_examples.sh
```

脚本默认使用 `$HOME/.local/opt/systemc-3.0.2`，默认编译到 `hardware/build/systemc/`。
自定义安装目录时：

```bash
SYSTEMC_PREFIX=/your/absolute/path/systemc \
  bash hardware/systemc/scripts/run_examples.sh
```

脚本会配置、编译并执行 CTest，两项检查通过后返回 0；失败返回非零退出码。
它通过自身位置定位仓库，因此也支持在其他目录用绝对路径调用。
可通过 `CMAKE=/path/to/cmake`、`BUILD_JOBS=4`、`SYSTEMC_BUILD_DIR=/path/to/build`
覆盖工具路径、并发和输出目录；输出目录建议使用绝对路径。

等价的手动流程如下，便于理解每一步。先设置 `SYSTEMC_PREFIX`：

```bash
cmake -S hardware/systemc -B hardware/build/systemc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$SYSTEMC_PREFIX" \
  -DBUILD_TESTING=ON
cmake --build hardware/build/systemc --parallel 2
ctest --test-dir hardware/build/systemc --output-on-failure

# 切换目录使直接运行产生的 VCD 文件也位于 build 中。
(
  cd hardware/build/systemc
  ./counter
  ./tlm_memory
)
```

不要在 `hardware/systemc/` 源码目录直接执行 `cmake .`。
所有本教程默认生成物都在 `hardware/build/`，已由现有 `.gitignore` 排除。

### CMake 如何找到并链接库

`CMakeLists.txt` 中的关键步骤：

```cmake
find_package(SystemCLanguage CONFIG REQUIRED)
set(CMAKE_CXX_STANDARD ${SystemC_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
target_link_libraries(counter PRIVATE SystemC::systemc)
```

`CMAKE_PREFIX_PATH` 告诉 CMake 到哪个安装目录找配置。
导入目标 `SystemC::systemc` 提供头文件路径、库和必要的链接依赖，
无需手动拼接 `-I`、`-L` 和 `-l`。TLM 示例链接相同目标即可。
示例沿用库的 C++ 标准，避免编译应用和库时标准不一致造成 API 版本符号链接错误。

## 5. 示例一：时钟、模块、信号与波形

文件：[`examples/counter.cpp`](examples/counter.cpp)。

系统包含一个 10 ns 周期的 `sc_clock`、一个 `Counter` 和一个 `Monitor`。
`Counter` 在每个上升沿加一，通过 `sc_out` 写入 `sc_signal<unsigned>`；
`Monitor` 在每个上升沿之后 5 ns 读取信号，检查 1 到 10。

| 代码元素 | 含义 |
|---|---|
| `sc_module` / `SC_CTOR` | 模块类型与注册仿真进程的构造函数 |
| `SC_METHOD(tick)` | 边沿触发的短进程；函数中不能调用 `wait()` |
| `sensitive << clk.pos()` | 只对上升沿敏感 |
| `dont_initialize()` | 不额外执行初始化调用；0 ns 的第一个上升沿仍会触发 |
| `SC_THREAD(check)` | 可以通过 `wait()` 挂起和恢复的仿真线程 |
| `sc_signal` | 将模块端口连接起来，信号写入在更新阶段生效 |
| `sc_start(100, SC_NS)` | 运行到 100 ns；内核暂停后回到 `sc_main` 检查结果 |

**delta cycle** 是同一仿真时间内的调度/更新轮次，不消耗物理时间。
在上升沿调用 `count.write()` 后，同一轮立即读信号会读到旧值。
示例选择半周期后检查，避免把进程执行顺序当成信号更新保证。

成功输出的最后一行：

```text
PASS: counter samples=10 final_count=10 time=100 ns
```

波形位于 `hardware/build/systemc/counter.vcd`。Linux 桌面可选安装 GTKWave：

```bash
sudo apt install -y gtkwave
gtkwave hardware/build/systemc/counter.vcd
```

把 `clk` 和 `count` 加入波形窗口即可查看。无图形界面的服务器只需运行 CTest，
也可把 VCD 拷到桌面查看。本例没有复位信号或溢出测试，检查范围是前十次递增。

## 6. 示例二：TLM 内存读写与延迟

文件：[`examples/tlm_memory.cpp`](examples/tlm_memory.cpp)。

`Initiator` 使用 `simple_initiator_socket` 发出事务，绑定到 `Memory` 的
`simple_target_socket`。内存有 64 字节，每个成功读写标注 10 ns 延迟。

一次事务的步骤：

1. 发起端构造 `tlm_generic_payload`：读/写命令、字节地址、数据指针、长度和 streaming width。
2. 调用 `socket->b_transport(tx, delay)`；此时直接执行目标端注册的回调。
3. 目标验证请求，通过后复制字节、把延迟增加 10 ns、设置 `TLM_OK_RESPONSE`。
4. 发起端检查响应，然后 `wait(delay)`，让仿真时间前进。

本例采用 **loosely-timed（LT）的阻塞传输**：目标只标注延迟，发起端等待。
目标回调不会等待；同一份延迟只计一次。`b_transport` 的“阻塞”指事务调用的同步接口，
并不意味着它必然自动推进时间。本例也没有 quantum keeper 或时间解耦。

| 检查 | 预期结果 |
|---|---|
| 地址 16 写入四个字节，再读回 | 两次 OK；字节一致；累计 20 ns |
| 地址 63 写四个字节 | `TLM_ADDRESS_ERROR_RESPONSE` |
| 使用 byte enable | `TLM_BYTE_ENABLE_ERROR_RESPONSE`，本模型不支持字节使能 |
| streaming width 小于数据长度 | `TLM_BURST_ERROR_RESPONSE`，不支持回绕传输 |
| 空数据指针 / 零长度 | `TLM_GENERIC_ERROR_RESPONSE` |
| `TLM_IGNORE_COMMAND` | `TLM_COMMAND_ERROR_RESPONSE` |
| 拒绝上述请求后重新读地址 16 | OK；原数据保持；累计 30 ns |

总共执行九个事务，其中三个成功；本模型约定被拒绝的事务不消耗时间。
使用显式字节序列，避免把主机整数的大小端行为混入仿真协议。
成功输出的最后一行：

```text
PASS: tlm_memory transactions=9 time=30 ns
```

这是单发起端、单目标、一次处理一个请求的模型。没有总线仲裁、并发请求、AXI burst、
缓存一致性或真实 DRAM 时序；不能用该 10 ns 参数推断 SmolVLA 的硬件性能。
本例只生成终端日志；计数器示例生成 VCD。

## 7. 与本仓库硬件框架的关系

现有 [`src/vla_tcs2/hardware/`](../../src/vla_tcs2/hardware/) 负责 Python 算子事件采集、
JSONL trace 和解析式性能估算。`HardwareSpec` 描述 PE 数量、频率、带宽、SRAM 等假设，
它自身不会创建 SystemC 模块或运行事件调度。

今后接入 SystemC 时，可分阶段实现：

1. 将 `OperatorEvent` 的 `m/n/k/batch` 与张量位宽转换为任务；检查 trace manifest 的 schema 版本。
2. 加入片外内存、DMA、片上缓冲和 GEMM 模块；明确数据搬运量、tile 和计算延迟。
3. 用小矩阵验证数值和时序，再与现有 dense backend 的估算对照。
4. 引入资源竞争、缓冲限制、重叠和调度，再开展完整 workload 性能实验。

这些是扩展路线，当前示例没有实现 trace adapter、DMA 或 GEMM。
需要注意现有 trace 的 `TensorDesc` 是张量描述，**不携带实际张量数值**；
数值验证必须另外提供测试数据和独立参考结果。
正式模型/性能实验仍遵循 [`experiments/README.md`](../../experiments/README.md)。

## 8. 常见问题

| 报错或现象 | 排查与解决 |
|---|---|
| `cmake: command not found` | 先安装工具；使用用户级 CMake 时设置 `CMAKE=/absolute/path/cmake` |
| 找不到 `SystemCLanguageConfig.cmake` | 检查是否执行 `cmake --install`，以及 `CMAKE_PREFIX_PATH` 是否为安装前缀；也可显式设置 `SystemCLanguage_DIR` 为配置文件所在目录 |
| 找不到 `systemc` / `tlm` 头文件 | 检查安装的 include 目录；使用本例 CMake 导入目标，不要单独用裸 `g++` 编译 |
| 链接出现 `sc_api_version_...` 未定义 | 检查库和应用的 C++ 标准、编译器/ABI 是否一致；使用新的编译目录重新配置 |
| `libsystemc.so` 无法加载 | 本例 CMake 的 build RPATH 通常处理本地运行；移动程序后需重新配置或设置正确库目录的 `LD_LIBRARY_PATH` |
| 修改安装路径后仍找到旧库 | 用全新 `SYSTEMC_BUILD_DIR` 配置，避免沿用缓存中的旧 `SystemCLanguage_DIR` |
| 编译进程被 OOM 杀死 | 减少 `--parallel` 或将脚本的 `BUILD_JOBS` 设为 1 |
| 波形的输出似乎晚于写入 | 检查 delta cycle 与采样阶段，`sc_signal::write()` 不会立即更新读值 |

动态库排查示例：

```bash
ldd hardware/build/systemc/counter
find "$SYSTEMC_PREFIX" -name libsystemc.so -print
# 将实际找到的库所在目录用于下一条命令；例如安装在 lib 下：
LD_LIBRARY_PATH="$SYSTEMC_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  hardware/build/systemc/tlm_memory
```

不需要通过关闭 SystemC API 版本检查来规避标准/ABI 错误，应统一库与应用的构建配置。

## 9. 验证记录与官方参考

本教程加入时在 Linux 上执行了官方库源码构建、安装，以及本目录两个程序的编译与 CTest。
实际验证环境与输出见 [`VALIDATION.md`](VALIDATION.md)。
该记录只覆盖下列示例，不代表已完成仓库全部 Python 或 HDL 测试。

- [SystemC 官方仓库（固定 3.0.2）](https://github.com/accellera-official/systemc/tree/3.0.2)
- [3.0.2 官方安装说明](https://github.com/accellera-official/systemc/blob/3.0.2/INSTALL.md)
- [3.0.2 官方 TLM 示例](https://github.com/accellera-official/systemc/tree/3.0.2/examples/tlm)
