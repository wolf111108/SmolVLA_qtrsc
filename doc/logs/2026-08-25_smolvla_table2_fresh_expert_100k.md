# 2026-08-25 MT50 结果回填 + hfvla 四 suite 收尾 + SmolVLA 论文 recipe 自微调（fresh expert 100k）启动

## 今日任务

### Meta-World 线（gpupro6000e）
- [x] MT50 全量评估完成（8-24 20:59 启动，PID 2100978）→ 结果回填见下
- [x] push-v3 独立 10 集复核（`outputs/metaworld_push_v3_10ep`）→ 80%，与 MT50 内该任务一致

### LIBERO 基线线（H100 机）
- [x] hfvla 四 suite 全量评测完成（8-24 16:19 启动，PID 2277367）→ 结果回填见下
- [x] 生成 `outputs/hfvla_vs_baselines.md`，5 个公开 checkpoint 全部评测完毕，基线选型最终定论

### 自微调线（gpupro6000d）
- [x] base 模型 + LIBERO 微调踩坑（accelerate 缺失、dataset key 对齐）
- [x] 决策：放弃 `smolvla_base` 微调路线当次尝试，改为 **从零初始化 expert 的论文 recipe 训练**（`--policy.type=smolvla` fresh，非 `--policy.path` 加载 checkpoint）
- [x] 启动 fresh-expert 100k 全量训练（**gpupro6000d**）

## 设备归属（重要）

| 任务 | 设备 | 说明 |
|---|---|---|
| fresh-expert 100k 训练（本文主角） | **gpupro6000d** | 单卡 RTX PRO 6000 Blackwell（98GB），torch 2.11.0+cu130，驱动 590.48.01 |
| MT50 评估 | **gpupro6000e** | 8-24 20:59 启动，8-25 凌晨完成 |
| hfvla_libero 四 suite 全量评测 | **H100 机**（8-24 启动，PID 2277367） | 已跑完，结果回填见下 |
| HuggingFaceVLA/libero 数据集（lerobot 格式副本） | **gpupro6000**（暂名） | `~/.cache/huggingface/lerobot/hub/datasets--HuggingFaceVLA--libero/`（snapshot 86958911...，data/chunk-000 383 parquet + meta/） |

本工作区经共享文件系统（NFS 类）在多台设备间同步：`gpupro6000d`（本机，1×RTX PRO 6000 Blackwell）与 A100 所在机器（hostname 不同、存储同一份）。数据集 / conda 环境 / HF cache 两边均可见。

> 待确认（设备拓扑）：gpupro6000e（LIBERO/MT50 评估）、gpupro6000d（训练）、gpupro6000（暂名，数据集副本）、H100 机（hfvla）、A100 机之间的 hostname 对应关系尚未完全核实。
>
> 数据集两种缓存格式：① lerobot 格式 `~/.cache/huggingface/lerobot/hub/datasets--<repo>/`（data/chunk-XXX/*.parquet + meta/，经典 LeRobotDataset API 用）；② datasets 库 arrow 格式 `~/.cache/huggingface/datasets/parquet/<hash>/*.arrow`（lerobot-train 流式加载用）。gpupro6000e 上只有 ②（63 个 arrow 分片，来自 8-24 自微调尝试），lerobot 格式权威副本在 gpupro6000。

---

## MT50 全量评估结果（gpupro6000e，8-24 启动 → 8-25 完成）

### 结果（50 任务 × 10 集 = 500 集，seed 1000）

| 难度 | 本次复现 | 论文 0.45B | 差异 |
|---|---:|---:|---:|
| Easy（28 任务） | **79.6%** | 82.5% | -2.9 pp |
| Medium（11 任务） | **38.2%** | 41.8% | -3.6 pp |
| Hard（6 任务） | **56.7%** | 45.0% | +11.7 pp |
| Very Hard（5 任务） | **42.0%** | 60.0% | -18.0 pp |
| 整体（500 集加权） | **64.0%** | — | — |
| 四难度算术平均 | **54.1%** | 57.3% | -3.2 pp |

- 结果文件：`outputs/metaworld_mt50_lerobot_smolvla/eval_info.json`
- push-v3 独立复核：`outputs/metaworld_push_v3_10ep` → 80.0%（8/10，失败集 ep2/ep4），与 MT50 内一致

### MT50 结论
1. **Meta-World 浮点基线成立**：四难度算术平均 54.1% vs 论文 57.3%（口径注意：论文为四难度算术平均，勿与 500 集加权 64.0% 混用）；
2. **两个待查偏差**：Very Hard 低 18pp（最大缺口，怀疑集中在 stick-pull/push 或 shelf-place）；Hard 高 11.7pp（方向相反，排除整体退化，是任务级分布差异）；
3. 评估参数已固化：`--rename_map`（image→camera1）+ `--empty_cameras=2` + 原版 config（state=[6] 不需改）。

---

## hfvla 四 suite 全量评测结果（H100 机，8-24 启动 → 8-25 完成）

### Suite 级总表（5 个公开 checkpoint 全部评测完毕）

| Suite | 论文 | **HuggingFaceVLA** (官方 ref, 32层/0.5) | lerobot (A, 16层/0.75) | k1000dai (D) | tiantianx (C) |
|---|---:|---:|---:|---:|---:|
| Spatial | 90% | 65% | **81%** | 64% | 76% |
| Object | 96% | 71% | 60% | **82%** | 59% |
| Goal | 92% | 72% | **77%** | 70% | 63% |
| Long | 71% | 37% | **59%** | 46% | 40% |
| **平均** | **87.3%** | 61.2%（第 3） | **69.2%** | 65.5% | 59.5% |

- 结果文件：`outputs/hfvla_libero_{spatial,object,goal,10}/eval_info.json` + `outputs/hfvla_vs_baselines.md`

### hfvla 结论
1. **官方 32 层大架构只排第 3**：参数更多（~0.6B vs 0.45B）并不更好；Object 有优势（71%）但 Long 全线最差（37%）；
2. **Long 是所有公开 checkpoint 的共同短板**（最好 A 类 59% vs 论文 71%），提示论文在 Long suite 上有特殊处理（训练数据配比/训练时长/eval 细节）；
3. **量化 baseline 最终定论：A 类 `lerobot/smolvla_libero`（69.2%）**——绝对成绩最高，四 suite 齐全且逐 suite 领先或接近领先；
4. 5 个公开 checkpoint 无一接近论文 87.3%（最佳 69.2%，差 18pp）→ 「论文数字依赖未公开细节，无法通过公开 checkpoint 复现」已有多模型证据；**自微调（论文 recipe）是拿到受控 baseline 的唯一路径**。

## 已解决的坑（继承自 8-24，今日确认）

1. `--policy.type` 与 `--policy.path` 互斥 → 二选一；
2. `ImportError: 'accelerate' is required` → `pip install accelerate`（1.14.0），已解决；
3. torchcodec 加载失败（缺 ffmpeg libavutil.so.*）→ 仅 WARNING，自动 fallback `pyav`，可忽略；
4. dataset key 对齐报错：
   ```text
   Missing: observation.images.camera1/camera2/camera3
   Extra:   observation.images.image / image2
   ```
   - 根因：`smolvla_base` config 声明 camera1/2/3 + 6D state，而 `HuggingFaceVLA/libero` 是 image/image2 + 8D state；
   - `--policy.type` fresh 路线的 input_features 从 dataset 推导，无此问题 → 今日改走 fresh 路线绕开；
   - `--policy.path=lerobot/smolvla_base` + rename_map 路线保留待验（state 6D vs 8D 仍是隐患）。

## 运行进程

### fresh-expert 100k 训练（gpupro6000d）

| 进程 | PID | 命令要点 | 输出位置 | 状态 |
|---|---|---|---|---|
| lerobot-train fresh expert 100k | **4159989** | `nohup lerobot-train --policy.type=smolvla ...` | `outputs/train_smolvla_libero_table2_fresh_expert_100k/` | 🟢 运行中 |

```bash
# gpupro6000d, conda smolvla_eval, 2026-08-25 15:28 启动
nohup lerobot-train \
    --policy.type=smolvla \
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --policy.load_vlm_weights=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.num_vlm_layers=16 \
    --policy.num_expert_layers=-1 \
    --policy.expert_width_multiplier=0.75 \
    --policy.attention_mode=cross_attn \
    --policy.self_attn_every_n_layers=2 \
    --policy.add_image_special_tokens=false \
    --policy.prefix_length=-1 \
    --policy.pad_language_to=longest \
    --policy.chunk_size=50 \
    --policy.n_action_steps=1 \
    --policy.num_steps=10 \
    --policy.max_state_dim=32 \
    --policy.max_action_dim=32 \
    --policy.optimizer_lr=0.0001 \
    --policy.optimizer_betas='[0.9,0.95]' \
    --policy.optimizer_eps=1e-8 \
    --policy.optimizer_weight_decay=1e-10 \
    --policy.optimizer_grad_clip_norm=10 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps=30000 \
    --policy.scheduler_decay_lr=2.5e-6 \
    --policy.compile_model=false \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.image_transforms.enable=false \
    --batch_size=64 \
    --steps=100000 \
    --accelerator.mixed_precision=bf16 \
    --num_workers=4 \
    --log_freq=200 \
    --save_freq=20000 \
    --env_eval_freq=0 \
    --seed=1000 \
    --policy.push_to_hub=false \
    --save_checkpoint_to_hub=false \
    --output_dir=outputs/train_smolvla_libero_table2_fresh_expert_100k \
    > outputs/train_smolvla_libero_table2_fresh_expert_100k.log 2>&1 &
```

启动验证（15:29）：

```text
num_total_params     = 450,046,176 (450M)   ← 论文 0.45B ✓
num_learnable_params = 99,880,992  (100M)   ← 论文 expert ~100M ✓ (expert-only 生效)
dataset.num_frames   = 273,465 (273K) / 1693 episodes  ← 与 tiantianx normalizer count 一致
effective batch      = 64                     ← 论文 batch64 ✓
显存                  ≈ 12.5GB / 98GB（安全）
```

查看进度：

```bash
tail -f outputs/train_smolvla_libero_table2_fresh_expert_100k.log
ps -p 4159989 -o pid,etime,cmd
```

## 结果

（待回填：最终 loss 曲线 / 各 20k checkpoint 评测成绩）

- 初始速度 ~5.9s/step → ETA ~164h（≈7 天）；前几步含预热，稳定速度待观察；
- 同卡另有进程 PID 4128301 占 33GB + 算力（GPU util 36%、85°C），抢资源，速度受影响。

## 结论与下一步

### 已定论
1. **Meta-World 浮点基线**：`lerobot/smolvla_metaworld` 四难度算术平均 54.1%（见上表）；
2. **LIBERO 量化 baseline 最终确认**：A 类 `lerobot/smolvla_libero`（69.2%），hfvla（61.2%）加入后结论不变；5 个公开 checkpoint 全部评测完毕，无一接近论文 87.3%；
3. **fresh-expert 训练已启动**：450M 总参 / 100M 可学习，与论文一致。

### 计划（按优先级）
1. [ ] 观察稳定 step 速度；若持续 >5s/step，考虑：`--num_workers=8`、错峰共享进程、或迁移（见下）；
2. [ ] 多卡调研：**gpupro6000d 仅 1 卡**；A100 在另一台机器（hostname 不同、NFS 共享存储）。跨机 2×1 卡 DDP 不划算（无 IB 证据）；可选方案是把训练整体迁到 A100 机 torchrun 单机多卡，或两机各跑一条独立路线对照；
3. [ ] 20k checkpoint（明天 ~20h 后）出来先跑单 suite 快评，提前判断 fresh-expert 路线成色；
4. [ ] （可选）`smolvla_base --policy.path` + rename_map 微调路线补测 state 6D/8D 行为；
5. [ ] 写 `scripts/summarize_metaworld_vs_paper.py`：按任务明细对比论文，定位 Very Hard 缺口任务，输出 `outputs/metaworld_vs_paper.md`；
6. [ ] 冻结 Meta-World 基线（checkpoint sha cd6778d2 + 评估参数），开始 PTQ 实验（手册 §15）；
7. [ ] 量化框架推进：QuantizedLinear + calibration（src/vla_tcs2/），A 类 LIBERO + Meta-World 双基线；
8. [ ] git 提交新结果（`hfvla_vs_baselines.md`、四个 suite 目录、logs/*.md、新脚本）。

## 备查

- torchcodec/pyav WARNING、kernel 4.18<5.5 warning（accelerate 提示可能 hang，暂观察）均为非致命；
- HF Hub 未设 HF_TOKEN（限速 WARNING），数据集已缓存完整（34.9GB），影响小；
- fresh 路线无需 rename_map（features 从 dataset 推导为 image/image2 + 8D state）；
- 此环境 torch 2.11.0+cu130 与 H100 机器（2.7.1+cu118）不同，属 gpupro6000d 新装环境，训练依赖今日起逐步补齐。
- MT50 评估命令（固化，gpupro6000e）：
```bash
lerobot-eval \
    --policy.path=/home/lfwang/.cache/huggingface/lerobot_smolvla_metaworld \
    --policy.n_action_steps=1 --policy.num_steps=10 \
    --policy.empty_cameras=2 \
    --rename_map='{"observation.image":"observation.images.camera1"}' \
    --env.type=metaworld --env.task=easy,medium,hard,very_hard \
    --eval.n_episodes=10 --eval.batch_size=1 --seed=1000 \
    --output_dir=outputs/metaworld_mt50_lerobot_smolvla
```
- MT50 checkpoint: `lerobot/smolvla_metaworld` sha `cd6778d2cfa724c1bf5fc637490548e54d81dc4c`；metaworld 3.0.0 / mujoco 3.8.1；EGL `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=2`。
