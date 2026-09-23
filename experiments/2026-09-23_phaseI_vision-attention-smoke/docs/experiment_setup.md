# 实验设置（Experiment Setup）

- **实验名称**：2026-09-23_phaseI_vision-attention-smoke
- **状态**：draft
- **创建日期**：2026-09-23
- **相关前序实验**：2026-09-22_phaseI_vision-smmm-sparsity

## 1. 实验目的

补齐 Vision Encoder 12 层的 QK/PV 量化路径。先验证 raw attention 与原 eager/SDPA 的数值等价性，再以 Vision QK/PV-only FP8 PoT 跑单 episode，检查量化、校准及 S|MMM 稀疏统计链路。这里只做接入验证，不执行此前规划的 W8/W4 混合精度正式实验。

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda | 沿用 smolvla_eval |
| 分支 | phaseI/vision-linear-full-integration |
| 基础提交 | 2b909969a3a615097b8a89dc92f0afa21c4e5ac3 |
| 运行提交 / 依赖 | runner 保存 commit.txt、worktree.patch、packages.txt |
| GPU / LeRobot / MuJoCo / EGL | 沿用现有评测环境；本地运行时记录 |
| attention 适配依据 | transformers 4.52.4 SmolVLMVisionAttention；eager/SDPA |

不修改全局 attention registry 或共享 config。其他版本由本地 preflight 实测确认，不默认兼容所有 transformers 版本。FlashAttention/flex 及 causal attention 明确拒绝。

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy | lerobot/smolvla_libero，legacy public checkpoint |
| 数据 | HuggingFaceVLA/libero，v3.0 |
| 校准 | 1 episode，batch_size=1，frame_stride=16，seed=42 |
| 量化范围 | 仅 Vision 24 MatMul；所有 Linear、VLM/Expert MatMul、connector raw |
| 格式 | QK/PV A/B/O=e4m3，pot_fp8_outlier，outlier_ratio=0.01 |
| scale | 全新独立目录，per_site，24×3=72 个文件 |

校准数据 episode 不必属于 Goal；这是小规模工程验证，不用于正式 PTQ 精度结论。

## 4. 评测协议

Goal task0 × 1 episode，seed=1000，batch_size=1，n_action_steps=10，num_steps=10，chunk_size 沿用 checkpoint。关闭视频与 unit sparsity。保留现有 camera rename。

统计使用 native counters，排除 outlier sidepath 人工零；FP8 bit 口径为 S|MMM v1。单 episode success 仅作 smoke，不能作为 Goal suite 成功率。

## 5. 实验变量与分组

| 组 | 配置 / 入口 | 变量 | 验收 |
|---|---|---|---|
| 单元验证 | tests/test_vision_attention.py | 真实 transformers 小 attention，随机权重 | mask/no-mask raw 等价、scale 保存加载、quant 非 identity、统计角色、异常恢复 |
| checkpoint raw gate | scripts/preflight.py | 原 attention vs 新 QK/PV raw | 12 层×2 mask 情形；1024 tokens；24 site 每个命中2次 |
| VM1 | configs/vision_qkpv_fp8.yaml | Vision QK/PV-only FP8 PoT | rollout 正常结束；24 manifest、72 scales、72 runtime rows |

实现约定：QK quantizer 输出为未乘 head scale 的矩阵，随后乘 self.scale、加 additive mask、FP32 softmax 后转 query dtype，再经 dropout 和 PV quantizer，最后 out_proj。mask 必须为 4D additive float，保持非因果语义。CURRENT_ATTN_KIND 在 finally 中恢复。

配置入口：

```yaml
quantization:
  quantize_matmul: false  # 此旧开关仍只控制 VLM/Expert
  matmul_scale_granularity: per_site
  vision:
    enabled: true
    matmul:
      enabled: true
      calibration_policy: recalibrate
```

vision.matmul 缺省关闭；开启时支持 per_site/per_component，拒绝会跨组件混用 scale 的 global/per_layer。方法和 A/B/O 格式继承现有 qk_matmul/pv_matmul 配置。本实验不启用 Vision Linear；以后同时启用全部 Vision/VLM/Expert 时，connector raw 下预期 296 Linear + 88 MatMul。

## 6. 运行命令

在本地已有评测环境、checkpoint 和数据缓存可用后执行：

```bash
conda activate smolvla_eval
bash experiments/2026-09-23_phaseI_vision-attention-smoke/scripts/run_smoke.sh
```

runner 顺序：保存版本 → 小型单元测试 → checkpoint raw gate → fresh calibration + task0 rollout → coverage 检查。任何步骤失败即停止。已有 run.log 或 scale 目录时拒绝覆盖，先归档旧输出再重跑。EGL 与 GPU 选择沿用当前环境，不硬编码设备编号。

raw gate 容差：FP32 atol=1e-5 / rtol=1e-4；低精度 atol=5e-3 / rtol=5e-2，输出每层 max/mean absolute error。该 gate 是局部 attention 数值验证，并不等价于完整闭环 backend 对照。

## 7. 输出目录映射

| 输入 | 输出 | 状态 |
|---|---|---|
| preflight.py | outputs/2026-09-23_phaseI_vision-attention-smoke/preflight.json | pending |
| vision_qkpv_fp8.yaml | outputs/2026-09-23_phaseI_vision-attention-smoke/vision_qkpv_fp8/ | pending |
| scales | scales/2026-09-23_phaseI_vision-attention-smoke/vision_qkpv_fp8/ | pending |

核心产物：原有评测结果、sparsity/module_sparsity.csv、quantization_manifest.csv、outlier_sidepath.csv、coverage_summary.json、run.log。coverage 检查按全部24个 module_id 与 A/B/O 三角色精确匹配，并验证 phase=prefill、component=vision、attention_kind=self。汇总比例按计数求和后相除。

## 8. 风险与注意事项

- 显式 attention 会物化 attention scores/probabilities，比 SDPA 占用更多显存；初次校准固定 batch_size=1。
- 本次是 fake quantization 接入，不提供原生 FP8 GEMM 加速。
- 不复用旧 Vision Linear scale，不改变既有实验配置。
- checkpoint raw gate、GPU 校准、LIBERO rollout 由用户本地执行；不能将创建实验记为已完成评测。
- 后续正式混合精度实验需另行评估 backend 切换影响，并保持校准/协议一致。
