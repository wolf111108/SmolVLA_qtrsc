# SystemC 示例验证记录

验证日期：2026-10-12。

## 环境

| 项目 | 实际版本/配置 |
|---|---|
| 操作系统 | Ubuntu 24.04.3 LTS，Linux x86_64 |
| 编译器 | GCC / G++ 13.3.0 |
| CMake | 3.31.6 |
| SystemC | 官方 tag `3.0.2` |
| SystemC 源码 commit | `70b0fc8e4a74acc677b0fc73cea08f940c2115d5` |
| 库配置 | Release、C++17、共享库、`ENABLE_EXAMPLES=OFF` |
| 示例配置 | Release、库导出的 C++ 标准、`BUILD_TESTING=ON` |

测试容器的 apt 权限受限，因此验证使用用户目录安装的 CMake；
SystemC 由官方源码构建并安装到临时用户可写目录，再通过 `SYSTEMC_PREFIX` 指定。
教程中的 Ubuntu / DNF 系统包安装命令未在本次验证中执行成功；
后续源码构建、安装、示例配置、编译和运行步骤均已实际执行。

## 执行流程

1. 按 README 中的 CMake 参数构建并安装官方 SystemC 库。
2. 设置 `CMAKE` 为工具绝对路径、`SYSTEMC_PREFIX` 为实际安装前缀，运行
   `bash hardware/systemc/scripts/run_examples.sh`。
3. 检查两个自检查程序的运行输出，以及 `counter.vcd` 波形文件的生成。
4. 检查脚本语法、Markdown 本地链接及 `git diff --check`。

CTest 结果：

```text
1/2 Test #1: counter .......................... Passed
2/2 Test #2: tlm_memory ....................... Passed
100% tests passed, 0 tests failed out of 2
```

程序结果：

```text
PASS: counter samples=10 final_count=10 time=100 ns
PASS: tlm_memory transactions=9 time=30 ns
```

计数器检查 5、15、…、95 ns 的输出分别为 1、2、…、10。
TLM 示例检查正常读写、地址越界、不支持的字节使能/回绕、空指针、
不支持命令和零长度，并确认拒绝的写入不改变有效地址的数据。

构建目录和 VCD 是本地生成物，不提交。验证范围是本目录教学程序，
未执行完整 Python 模型评测、HDL 仿真、GTKWave GUI 验证或真实板卡验证。
