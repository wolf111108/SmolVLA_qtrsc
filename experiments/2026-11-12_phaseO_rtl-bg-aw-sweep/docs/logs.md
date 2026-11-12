# Phase O — 运行日志

## 当前状态

implemented / CPU verified / pending GPU run。

## 运行日志

2026-11-12，在隔离源码快照中执行，Python 3.13.5、NumPy 2.3.5、PyTorch 2.10.0+cpu、PyYAML 6.0.3：

```text
python tests/test_rtl_bit_group_sparsity.py -v                    14/14 PASS
python tests/test_bit_group_sparsity.py -q                       10/10 PASS
python experiments/2026-11-09_phaseL_int8-goal/scripts/test_experiment.py 10/10 PASS
Phase O run.py --preflight                                     PASS
Phase N run.py --preflight                                     PASS
Phase L run_arm.py preflight                                   PASS
```

新增覆盖：INT8/INT4 全值重构、BG0 补偿、INT4 高位稀疏、独立 scalar oracle、batch/head/转置、尾包/浮点 mask、K 分块等价、p 单调性、weight 去重、六组配对校验、缺失站点/step 拒绝、真实 Linear/QK/PV 输出等价、完整 CSV 导出。

## 后台任务

无。未启动 GPU 作业、训练或 rollout。

## 异常与处理

源码获取通过 GitHub API 的固定 blob SHA 校验；容器无法直接 git clone。依赖源码沿原仓库恢复，未用 mock 模块替代生产 quantization 实现。

## 修订记录

- 2026-11-12：新建并完成 CPU 回归。运行结果与未完成边界见 `results.md`。
