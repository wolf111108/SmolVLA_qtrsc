# 实验日志（Logs）

- 实验名称：2026-10-16_phaseL_gemm-single-inference-timeline
- 状态：draft
- 最后更新：2026-10-16

## 1. 当前状态

| 阶段 | 状态 | 备注 |
|---|---|---|
| 配置与脚本 | 已准备 | A8/W8 VLM、A8/W4 expert Linear，dense OS 双缓冲 |
| 软件验收 | 已通过 | 小矩阵导出/时间单位/全局偏移/占用积分/出图；不加载模型 |
| 真实模型单次推理 | 待用户本地执行 | 按用户最新要求，本次不执行模型实验 |
| 模型瓶颈分析 | 待运行产物 | 自动生成 timeline/report.md、CSV、Chrome Trace 与图表 |

## 2. 运行日志

| 日期 | 阶段 | 命令 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-10-16 | 软件验收 | python experiments/2026-10-16_phaseL_gemm-single-inference-timeline/scripts/check_export.py -q | passed | 临时目录 | 不产生模型性能数据 |
| 待运行 | 模型推理 | bash experiments/2026-10-16_phaseL_gemm-single-inference-timeline/scripts/run.sh | pending | outputs/2026-10-16_phaseL_gemm-single-inference-timeline/int8_w4 | 同步后在现有模型环境执行 |

## 3. 后台任务

无模型实验后台任务。

## 4. 异常与处理

无真实模型运行异常记录。首次交付仅验证软件出口，若本地环境的 LeRobot/Transformers 接口与仓库冻结版本不同，先核对依赖和本仓库既有量化测试。脚本失败会保留 failed metadata，shell 不继续出图；不要将失败捕获当作完整推理。

## 5. 修订记录

| 日期 | 修改内容 |
|---|---|
| 2026-10-16 | 按 experiments/README.md 创建实验结构；根据用户要求将模型执行留给本地 |
