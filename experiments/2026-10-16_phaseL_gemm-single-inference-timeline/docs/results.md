# 结果记录（Results）

- 实验名称：2026-10-16_phaseL_gemm-single-inference-timeline
- 状态：draft，真实模型实验待用户本地执行
- 最后更新：2026-10-16

## 1. 摘要

已准备自包含配置、单次 action-chunk 推理入口、完整时间线导出与绘图。没有在本次交付中运行 SmolVLA 模型，没有真实模型性能数字。软件验收仅使用确定性的小矩阵 GEMM。

## 2. 总结果表

| 组 | GEMM-only 延迟 | DRAM 读/写 | 计算服务时间 | DRAM 占用时间 | 有效 PE 利用率 | 状态 |
|---|---|---|---|---|---|---|
| int8_w4 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | pending |

运行后首先查看 `outputs/2026-10-16_phaseL_gemm-single-inference-timeline/int8_w4/timeline/report.md`，再将汇总结果与输出目录记录到本表。时间口径见 setup §4。

## 3. 分组结果与分析

运行后检查 pipeline_occupancy、resource_utilization 和 detail_* 图。判断瓶颈时同时观察 DRAM、DMA、NoC、Local/Accumulator 端口和阵列；高锁定低服务需求可能是固定延迟或同步事务限制。结合 operators.csv 和 Perfetto 定位具体模块与 tile。组件的读写端口可并行，不把端口百分比相加当作组件利用率。

按实验规范，确认运行成功后将最终结果图片复制到本实验 docs/figures，并在本节添加相对路径引用：

```bash
cp outputs/2026-10-16_phaseL_gemm-single-inference-timeline/int8_w4/timeline/figures/*.png \
   experiments/2026-10-16_phaseL_gemm-single-inference-timeline/docs/figures/
```

软件检查：双缓冲小矩阵的两个 GEMM 各 9 cycles，拼接后为 18 cycles；Local.read 占用为 8 cycles，而服务需求为 4 cycles。该数字只用于验证导出时间偏移、占用/服务区分和图像积分，不是 SmolVLA 性能。

## 4. 结论

尚无模型级性能结论。当前产物提供可复现执行入口与无损时间线数据，不宣称某一资源已经构成实际 SmolVLA 瓶颈。

## 5. 问题与后续

用户同步并本地运行后，核对 run_metadata.status=completed、完整 generation/denoise 覆盖和硬件 summary。保留配置、trace 和 simulations 以复查。若有 unsupported 或尺度不兼容，先修复再运行新的输出目录。

## 6. 修订记录

| 日期 | 修改内容 |
|---|---|
| 2026-10-16 | 创建实验；添加单次推理、时间线/利用率导出及软件验收；模型实验待本地执行 |
