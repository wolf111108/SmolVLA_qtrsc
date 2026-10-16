当前没有这组配置，已新增实验并推送到 `phaseI/vision-linear-full-integration`，提交 **`56c64cb`**。

| 组件 | Linear 权重 | 激活与 QK/PV |
|---|---|---|
| Vision Encoder | INT4 | FP8 E4M3 PoT |
| VLM | INT8 | FP8 E4M3 PoT |
| Expert | INT4 | FP8 E4M3 PoT |

保留 1% outlier；**整数权重 scale 为连续值，PoT 用于激活及 MatMul scale**。

实验为 Goal **10 tasks × 10 episodes**，独立校准，先跑 1 episode 预检。只测 SR，不重跑 baseline、不收集稀疏度。

更新仓库后运行：

```bash
GPU=0 bash experiments/2026-09-29_phaseK_vision-w4-vlm-w8-expert-w4-goal/scripts/run_all.sh
```

已通过静态与汇总逻辑检查，未安装 PyTorch、未运行模型。