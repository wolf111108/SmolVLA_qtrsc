# 实验设置（Experiment Setup）

- **实验名称**：2026-09-25_phaseI_vision-joint-smoke
- **状态**：draft
- **创建日期**：2026-09-25
- **相关前序实验**：2026-09-23_phaseI_vision-attention-smoke（run #6）；Phase I VLIN

## 1. 实验目的

验证 Vision Encoder 的 72 Linear 与 24 QK/PV MatMul 同时启用 FP8 PoT 时，模块路由、raw 前向、独立校准、LIBERO rollout 与稀疏统计可联合运行。该实验是 Vision-only 工程 smoke，不是全模型 W8/W4 混合精度正式评测。

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| 分支 | phaseI/vision-linear-full-integration |
| 基础提交 | bcba565：含校准统计器 finally 恢复修复 |
| 环境 | 沿用 smolvla_eval、GPU、MuJoCo/EGL 设置 |
| 前序实际依赖 | torch 2.7.1+cu118 / transformers 5.5.4 |
| 本次实际版本 | runner 保存 commit.txt、worktree.patch、packages.txt 与 config.yaml |

不安装或更新依赖。由用户本地执行，本轮仅创建实验。

## 3. 模型与数据

| 项目 | 配置 |
|---|---|
| policy | lerobot/smolvla_libero（legacy public），revision=null 沿用现有设置 |
| Vision Linear | 12×(q/k/v/out_proj、fc1/fc2)=72；A/W/O 均 E4M3 |
| Vision MatMul | 12×(QK/PV)=24；A/B/O 均 E4M3 |
| 方法 | pot_fp8_outlier；outlier_ratio=0.01；所有 scale 为 PoT |
| VLM / Expert / connector | raw；全局 quantize_matmul=false 仅关闭 VLM/Expert MatMul |
| 校准 | HuggingFaceVLA/libero v3.0，1 episode，batch=1，stride=16，seed=42 |
| scale | 新目录全部 recalibrate，per_site；216 Linear + 72 MatMul = 288 文件 |

本次 W 也是 FP8，不是 INT8/INT4。后续正式对比前应固定 checkpoint revision；这里沿用前序 checkpoint 缓存和协议。

## 4. 评测协议

LIBERO Goal task0 × 1 episode，seed=1000，batch_size=1，n_action_steps=10，num_steps=10；chunk_size 沿用 checkpoint。保留 camera rename，关闭视频和 unit sparsity。

单 episode 只报告成功/失败，不能作为 suite SR 或量化无损的证据。coverage PASS 表示工程产物齐全，episode_success 单独记录，不要求必须成功才承认链路运行完成。

## 5. 实验变量与分组

| 组 | 配置 | 变量 | 固定项 |
|---|---|---|---|
| VJ1 | vision_joint_fp8.yaml | 同时开启 Vision 全 Linear + QK/PV FP8 PoT | VLM/Expert/connector raw、1ep 协议 |

### Gate 与统计口径

1. raw preflight：先 build 无量化 control，注入72个raw Linear，并逐个与相同权重/输入的 F.linear 对比，再注入24个raw MatMul，执行12层×mask/no-mask三方对照。
2. 硬门槛：Linear raw 全部通过、adapter_vs_eager 通过、24 MatMul 各调用2次、无异常、数值有限。有限 SDPA/eager 超差仅报告 PASS_WITH_BACKEND_DRIFT；不放宽原容差。
3. fresh calibration + rollout：96 sites 全 recalibrate，禁止复用此前仅 Linear / 仅 MatMul scale。
4. 精确验收：manifest=96；runtime=216（Linear activation/output=144，MatMul A/B/O=72）；static weight=72；全部 module_id 与角色匹配、component=vision、phase=prefill。MatMul attention_kind=self；MLP 无需此标签。
5. scale 必须恰好288个指定文件、数值有限且正、为2的整数次幂；写 scale_audit.json（数值与SHA256）。
6. runtime 使用 native counters 排除 outlier 人工零；FP8 bit 使用 S|MMM v1。分别输出 runtime_linear、runtime_matmul、runtime_pooled、static_weight；保留分子分母，先求和再相除。权重不与运行时张量混池。

联合 preflight 只验证算子 raw 语义和路由；真实图像路径覆盖由 rollout 的全部216条记录验收，不将随机输入测试当作完整闭环等价证明。正式精度实验仍需同协议 eager raw 基线。

## 6. 运行命令

```bash
conda activate smolvla_eval
git pull --ff-only
bash experiments/2026-09-25_phaseI_vision-joint-smoke/scripts/run_smoke.sh
```

runner 保存版本/config → 现有 attention 单元测试 → 联合 preflight → main.py fresh calibration + rollout → check_outputs。任何硬门槛失败立即停止。

本实验使用全新目录，无需移动此前 QK/PV-only 的结果。重跑本实验时，先自行归档其 outputs 与 scales；已有 run.log/preflight.json/scale目录时 runner 拒绝覆盖。

## 7. 输出目录映射

| 输入 | 输出 | 状态 |
|---|---|---|
| preflight.py | outputs/2026-09-25_phaseI_vision-joint-smoke/preflight.json | pending |
| vision_joint_fp8.yaml | outputs/2026-09-25_phaseI_vision-joint-smoke/vision_joint_fp8/ | pending |
| 校准 scale | scales/2026-09-25_phaseI_vision-joint-smoke/vision_joint_fp8/ | pending |

请回填 preflight.json、coverage_summary.json、scale_audit.json、result.json、run.log、config.yaml、commit.txt、worktree.patch、packages.txt 及 sparsity/module_sparsity.csv、weight_sparsity_static.csv、quantization_manifest.csv、outlier_sidepath.csv。原始产物保存在 outputs 下，文档只记录摘要及证据链接；无需上传模型或数据集。

## 8. 风险与注意事项

- 显式 attention 比 SDPA 占用更多显存，校准batch固定1。
- 本次仅工程校准1 episode，不能承诺正式精度；联合量化可能增加误差。
- 本实验只创建配置和脚本，未执行测试、校准或 rollout。
- 不改变核心量化实现；依赖 bcba565 的统计器生命周期修复。
- 前序单独 Linear / QK/PV 的成功不能替代联合验证；历史实验的尺度和统计不得直接混用。
