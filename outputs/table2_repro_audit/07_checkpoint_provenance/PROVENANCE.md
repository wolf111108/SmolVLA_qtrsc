# PHASE 7 — Checkpoint Provenance 审计结果

> 设备：h100 · 2026-08-25 · 脚本 `scripts/audit_phase7_checkpoints.py`
> 归档：`07_checkpoint_provenance/configs.txt`

## 五 checkpoint 关键字段并排对比

| 字段 | smolvla_base | lerobot/smolvla_libero | HuggingFaceVLA/smolvla_libero | tiantianx | k1000dai |
|---|---|---|---|---|---|
| vlm_model_name | 500M-**Video** | 500M-Video | 500M-**Instruct** ❌ | 500M-Video | 500M-Video |
| num_vlm_layers | 16 | 16 | **0(→32)** ❌ | 16 | 16 |
| num_expert_layers | 0(跟随) | 0 | -1(→32) ❌ | 0 | 0 |
| expert_width_multiplier | 0.75 | 0.75 | **0.5** ❌ | 0.75 | 0.75 |
| n_action_steps(config) | 50 | 50 | **1** | 50 | 50 |
| freeze_vision_encoder | True | **False** ❌ | True | True | True |
| train_expert_only | True | **False** ❌ | True | True | True |
| pad_language_to | max_length | max_length | **longest** | max_length | max_length |
| train.steps | (无train_config) | **25000** ❌ | (无) | 100000 | 100000 |
| train.batch_size | — | **32** ❌ | — | 32 | **64** ✅ |

## 关键判定（回应手册 7.2）

### 1. `HuggingFaceVLA/smolvla_libero` ≠ Table-2 exact checkpoint（硬证据）

三处架构漂移，全部实测确认：
- `vlm_model_name` = Instruct（论文是 Video）
- `num_vlm_layers=0` → 不截断 = 32 层（论文 16）
- `expert_width_multiplier=0.5` → hidden 480（论文 0.75→720）

### 2. `lerobot/smolvla_libero`（A 类）recipe 严重偏离论文

之前只知道"~25k / full finetune"，现在精确确认：
- `train.steps=25000`（论文 100k）
- `train.batch_size=32`（论文 64）
- `freeze_vision_encoder=False` + `train_expert_only=False`（论文两者都 True）

→ 它是"架构像论文、recipe 完全不像"的典型。

### 3. `tiantianx`（C 类）recipe 最接近，但 batch 存疑

- steps=100k ✅、freeze_vision_encoder=True ✅、train_expert_only=True ✅、架构 16/0.75 ✅
- 但 `train.batch_size=32`（论文 64），且 `train.pretrained_path=<missing>`——README handoff 里"2 GPU×32=64"的推测无法从 train_config 证实（无 world_size 字段）

### 4. `k1000dai`（D 类）是唯一 batch=64 的

- steps=100k ✅、batch=64 ✅、freeze_vision_encoder=True ✅、train_expert_only=True ✅
- 架构 16/0.75 ✅ → 这是**recipe 上最贴论文**的 checkpoint（比 tiantianx 还多一项 batch=64）
- 但它成绩只有 65.5%（低于 A 类 69.2%），说明 recipe 吻合 ≠ 成绩吻合，其他未记录因素（数据顺序/scheduler/epoch）仍在起作用

### 5. `smolvla_base` 无 train_config

符合预期（预训练底座，非微调产物）。

## 对 PHASE 8（strict init）的直接启示

strict Table-2 要求"仅 pretrained VLM + 随机 expert"。而：
- `smolvla_base` 的 `load_vlm_weights=True` + `freeze_vision_encoder=True` + `train_expert_only=True` 说明它本身就是"VLM 权重 + 可训练 expert"结构；
- 但它的 expert 已经过 robotics pretraining（不是随机）→ 这就是为何手册坚持用 `--policy.type=smolvla --policy.load_vlm_weights=true`（重建随机 expert）而非 `--policy.path=smolvla_base`。

## 判定

**PASS: CHECKPOINT_PROVENANCE**（五 checkpoint 来源与架构/recipe 差异已全部记录在案）

## 回答手册第 19 节 Q1/Q2

- Q1 Table 2 原始 checkpoint 是否公开？→ **否**，未找到任何同时满足"官方+16层+0.75+100k+batch64+expert-only"的公开 checkpoint
- Q2 current official 与论文架构是否相同？→ **否**，HuggingFaceVLA 是 32层/0.5/Instruct，明确漂移
