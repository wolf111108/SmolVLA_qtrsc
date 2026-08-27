# 2026-08-24 LIBERO 基线收尾 + Meta-World 第二基准接入

## 今日任务

### LIBERO 线
- [x] HuggingFaceVLA/smolvla_libero_ckpts@100k 四 suite → **smoke 0/1 失败，审计后关闭**（见下）
- [x] k1000dai/smolvla_libero_finetune 四 suite (社区 16/0.75, 100k, batch64) → **完成，平均 65.5%**
- [x] 三可用 checkpoint 结构与 baseline 总结（`outputs/checkpoint_baselines_summary.md`）
- [x] 量化 baseline 选型定论：`lerobot/smolvla_libero` (A 类, 69.2%)
- [x] 论文 recipe 自微调可行性确认（lerobot-train + smolvla_base + HuggingFaceVLA/libero）
- [x] `HuggingFaceVLA/smolvla_libero` 架构核实（32 层/0.5 宽/500M-Instruct，非论文结构）
- [ ] `HuggingFaceVLA/smolvla_libero` 四 suite 全量评测（进行中，PID 2277367，另一台机器）
- [ ] 自微调冒烟（500 steps）——已踩坑：`--policy.type` 与 `--policy.path` 互斥、缺 accelerate、缺 repo_id，逐步解决中

### Meta-World 线（第二基准）
- [x] 环境重建验证：`smolvla_eval` 从空环境重建（torch 2.11.0+cu130 + libero 栈），复现 `lerobot/smolvla_libero` spatial 81%
- [x] metaworld 3.0.0 安装（`pip install -e ".[metaworld]"`）
- [x] 模型 `lerobot/smolvla_metaworld` 下载（~1GB，本地 `~/.cache/huggingface/lerobot_smolvla_metaworld`）
- [x] 模型结构 / 数据集接口审计（config、normalizer 权重、meta/info.json、tasks.parquet）
- [x] 单任务冒烟 `push-v3` → **100% 通过**（原版 config，未改任何文件）
- [x] 全量 MT50 评估启动（50 任务 × 10 集，PID 2100978，进行中）

## 运行进程
```bash
conda activate smolvla_eval
cd ~/VLA_tcs2
nohup bash scripts/run_new_candidates_libero.sh > outputs/new_candidates_run.log 2>&1 &
```

流程: preflight(自动 rename_map) -> smoke(task0 x1) -> 四 suite x100 episodes(幂等) -> 自动对比。
smoke 失败的模型自动跳过, 不阻塞另一个。

已于 10:17 全部结束，无遗留进程。

## 结果

### k1000dai_ft100k (D 类) 四 suite（10:17 完成）

| Suite | k1000dai (D) | 论文 | tiantianx (C) | lerobot (A) |
|---|---:|---:|---:|---:|
| Spatial | 64% | 90% | 76% | 81% |
| Object | **82%**（可用模型中最高） | 96% | 59% | 60% |
| Goal | 70% | 92% | 63% | 77% |
| Long | 46% | 71% | 40% | 59% |
| **平均** | **65.5%** | 87.3% | 59.5% | **69.2%** |

详: `outputs/k1000dai_ft100k_libero_{spatial,object,goal,10}/eval_info.json`、`outputs/new_candidates_run.log`

### hfvla_ckpts100k：smoke 0/1 → 审计关闭

smoke 无异常跑完 280 步但 pc_success=0。审计（`scripts/audit_hfvla_ckpts100k.py` → `outputs/audit_hfvla_ckpts100k.txt`）：

- `text embed_tokens=(49280, 2048)` → **VLM 是 SmolVLM2-2.2B-Instruct**（非论文 500M-Video），vision 是 SigLIP-SO400M 级 (1152)；
- normalizer **完全没有 state stats** → pi 风格无 state 数据训练（纯视觉），state_proj 层存在但训练时输入全零；
- 与论文结构/数据均偏离 + repo 标记 DEPRECIATED → **判定不可修/不值得修，关闭**。

（另注：run log 里早先的 `Repo id must be in the form...` 报错是原作者 /raid/jade 本地路径问题，preflight 已迁移为 HF repo id；真正导致跳过的是上面这条 0/1。）

### HuggingFaceVLA/smolvla_libero（官方 current reference）架构核实

本地 config（revision 6721902）核对 + 源码 `smolvlm_with_expert.py`：

| 项 | 值 | 论文(=smolvla_base) |
|---|---|---|
| vlm_model_name | SmolVLM2-500M-**Instruct** | SmolVLM2-500M-**Video** |
| num_vlm_layers | 0 → 不截断 = **32 层** | 16 层 |
| num_expert_layers | -1 → 跟随 = **32 层** | 16 层 |
| expert_width_multiplier | **0.5** → hidden 480 | 0.75 → 720 |
| 总参数 | ~0.6B（模型卡标注） | ~450M |
| 输入 | 8D state + image/image2 | 6D state + camera1/2/3 |

源码证据：`if num_vlm_layers > 0: 截取; 否则 self.num_vlm_layers = len(...) = 32`。
**结论**：官方 LIBERO checkpoint 架构已漂移，非论文结构；官方模型卡/leaderboard 均无此变化的说明。
意义：仅作 simulator/pipeline 正确性 reference，不能作论文结构量化 baseline。

### HuggingFaceVLA/smolvla_libero 四 suite 全量评测（进行中）

- **设备归属：不在 gpupro6000d 本机跑**（8-25 追注：在另一台设备上运行，与本机经 NFS 共享工作区）；结果出来后回填本日志

- 脚本：`scripts/run_hfvla_libero_suites.sh`（新建）+ `compare_hfvla_vs_baselines.py`
- 无需 rename_map（config 即 8D state + image/image2，与 LIBERO processor 一致）
- 运行：PID 2277367，nohup，16:19 启动 libero_spatial，预期 8-12h
- 输出：`outputs/hfvla_libero_<suite>/eval_info.json` + `outputs/hfvla_vs_baselines.md`
- 日志：`outputs/hfvla_libero_multisuite_logs/<suite>.log`

## 自微调（论文 recipe）踩坑记录

目标命令（lerobot-train，冒烟 500 steps）：

```bash
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.repo_id=smolvla_libero_smoke500 --batch_size=8 --steps=500 \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.train_state_proj=true \
  --accelerator.mixed_precision=bf16 --output_dir=outputs/train_smolvla_libero_smoke500 \
  --save_freq=500 --seed=1000
```

已解决/待解决的坑（按遇到顺序）：

1. `Cannot specify both --policy.path and --policy.type` → 去掉 `--policy.type`（path 已隐含类型）；
2. `'accelerate' is required` → `pip install -e ".[training]"`；
3. `'repo_id' argument missing` → 加 `--policy.repo_id=smolvla_libero_smoke500`（本地训练不推 hub 也强制要求）；
4. torchcodec 加载失败（缺 ffmpeg lib）→ WARNING，自动 fallback pyav，可忽略；
5. 下一次预期坑：dataset key 对齐（`image/image2/8D state` vs policy `camera1/2/3/6D state`）。

---

## Meta-World 第二基准接入

### 运行进程
```bash
conda activate smolvla_eval
cd ~/VLA_tcs2/lerobot_current
nohup bash -c '
lerobot-eval \
    --policy.path=/home/lfwang/.cache/huggingface/lerobot_smolvla_metaworld \
    --policy.n_action_steps=1 --policy.num_steps=10 \
    --policy.empty_cameras=2 \
    --rename_map='\''{"observation.image":"observation.images.camera1"}'\'' \
    --env.type=metaworld --env.task=easy,medium,hard,very_hard \
    --eval.n_episodes=10 --eval.batch_size=1 --seed=1000 \
    --output_dir=$HOME/VLA_tcs2/outputs/metaworld_mt50_lerobot_smolvla
' > $HOME/VLA_tcs2/outputs/metaworld_mt50_run.log 2>&1 &
```
- **PID: 2100978**，启动 20:59，日志 `outputs/metaworld_mt50_run.log`
- 预期 3-6h；看进度 `tail -f outputs/metaworld_mt50_run.log`

### 环境重建验证（libero）
- `outputs/libero_spatial_rerun_v2/eval_info.json`：pc_success **81.0%**（100 集），与历史 A 类 spatial 81% 一致 → 重建环境复现成功。
- 注意 torch 版本漂移：yml 锁定 2.7.1+cu118，实际装的是 **2.11.0+cu130**（PyPI 默认），不影响结果。

### Meta-World 接口审计（重要发现）
| 项 | 事实 | 结论 |
|---|---|---|
| 模型架构 | 16 层 VLM / 0.75 宽 / SmolVLM2-500M-Video-Instruct / action 4D | 论文同款结构 |
| config state=[6] | normalizer 权重里 state 统计是 **[4]** 维（与数据集 4D agent_pos 一致） | **实测无需改 config**，lerobot state 通道有 pad 兜底；保持官方文件原样 |
| 相机 | env 单相机 `observation.image` 480×480；模型要 camera1/2/3 | 评估必须 `--rename_map='{"observation.image":"observation.images.camera1"}'` + `--policy.empty_cameras=2` |
| 任务数 | 数据集 total_tasks=**49**（2500 eps），工程配置 50 任务 | 缺失任务是 **push-back-v3**；其描述与 push-v3 完全相同，模型训练时见过该指令。全量评估含 50 任务，报告注明即可 |
| metaworld 3.0.0 | 无 `__version__`；MT1 只认 V3 任务名 | 手册 §6/§12.1 的 `push-v2` 示例是 bug，已修手册为 `push-v3` |

### Meta-World 踩坑记录（按遇到顺序）
1. libero 首次运行交互提问（dataset path）→ 交互式回答后写入 `~/.libero/config.yaml`，之后不再问；
2. `num2words` 缺失（SmolVLM processor 需要）→ `pip install num2words==0.5.14`；
3. `huggingface-cli` 已废弃 → 用 `hf download`；
4. env 相机名与 policy 不匹配 → rename_map + empty_cameras=2（见上表）；
5. pip 慢代理 → `env -u ...` + `--proxy ""` + tuna 源（详见 8-20 会话记录）。

## 结论与下一步

### 已定论

1. **无公开 checkpoint 接近论文 87.3%**（最佳 A 类 69.2%，差 18pp）；各 suite 最佳分属不同模型 → 系统性训练差距；
2. **量化 baseline 选 A 类 `lerobot/smolvla_libero`（69.2%）**：成绩最高、四 suite 齐全、架构仍是论文同款 16/0.75；量化退化相对自测 baseline 报告，不对照论文；需注明其 full-finetune（非 expert-only）；
3. 总结报告：`outputs/checkpoint_baselines_summary.md`（结构对照 + 成绩 + 角色定位 + 量化规范）。

### 计划（按优先级）

1. [ ] **自微调（论文 recipe）**：`smolvla_base` + `HuggingFaceVLA/libero` + 100k/batch64/expert-only/frozen VLM/BF16，
   先 500 steps 冒烟（验证 dataset key 对齐 image/image2→camera1/2、8D state 进 6D 模型的处理、显存），再全量；
   训练时长单卡 H100 数天，可考虑新机器 gpupro6000e 并行；
2. [ ] 量化框架不被阻塞：先用 A 类 checkpoint 开发 QuantizedLinear + calibration（src/vla_tcs2/），新 baseline 出来后替换；
3. [ ] 补：刷新 `outputs/new_candidates_vs_baselines.md`（compare 脚本重跑，D 类数据回填）；
4. [ ] `hfvla` 四 suite 跑完后，用 `compare_hfvla_vs_baselines.py` 出对比报告，视结果更新 baseline 选型逻辑（若 hfvla 仍 < 87.3% 则进一步印证论文不可达；若达 87%+ 则说明 16/0.75 结构本身受限）；
5. [ ] 等全量 MT50 结果（~3-6h），用难度分组 per_group 对比论文 82.5/41.8/45.0/60.0（平均 57.3），写汇总脚本输出 `outputs/metaworld_vs_paper.md`；
6. [ ] MT50 结果确认后冻结 Meta-World 浮点基线 → 开始 PTQ 实验（手册 §15 流程）。

## 备查
- 候选发现过程: HF Hub API 搜索 smolvla+libero, 核查 config 架构/train_config recipe;
  hfvla_ckpts100k 需定位子目录 100000/pretrained_model (preflight_policy.py 的 hfsub: 机制);
  两候选相机 key 均为 image/wrist_image, rename_map 由 preflight 自动推导。
- 自微调入口: `lerobot-train`（lerobot_current/src/lerobot/scripts/lerobot_train.py，支持 --policy.path/--dataset.repo_id/bf16/grad_accum）；训练前需 `which lerobot-train` 确认已装（必要时 pip install -e ".[training]"）。
- 学习: `lerobot/smolvla_base`（预训练底座，6D/3 相机，无 LIBERO 成绩）≠ A 类 `lerobot/smolvla_libero`（LIBERO 微调版）。
- Meta-World: checkpoint `lerobot/smolvla_metaworld` sha `cd6778d2cfa724c1bf5fc637490548e54d81dc4c`；metaworld 3.0.0 / mujoco 3.8.1 / gymnasium 1.3.0；EGL `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=2`（重建环境后需重新 export）；论文 Meta-World 0.45B：Easy 82.5 / Medium 41.8 / Hard 45.0 / Very Hard 60.0 / 平均 57.3。
- 手册 `MetaWorld_Testing_Manual_SmolVLA_v2.md` 已中文化并修正 push-v3 / __version__ 两处 bug。
