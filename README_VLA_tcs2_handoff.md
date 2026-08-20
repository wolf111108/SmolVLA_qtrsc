# VLA_tcs2 项目交接 README

> 更新时间：2026-08-20  
> 当前阶段：**优先完成 SmolVLA + LIBERO 的可复现仿真基线，再继续量化与硬件建模。**  
> 交接原则：后续实验尽量“一次只改变一个变量”，每一步都保留命令、commit、checkpoint revision、seed 和输出目录。

---

## 1. 项目目标

本项目希望搭建一套可扩展的 VLA（Vision-Language-Action）实验框架，以 **SmolVLA** 作为第一类 workload。长期目标包括：

1. 在 LIBERO 等机器人仿真环境中完成闭环 VLA evaluation，并尽可能复现论文结果；
2. 对 VLA 中的 Linear / MatMul 等核心计算引入 PTQ（Post-Training Quantization）；
3. 统计运行时 activation / weight / output 的数值范围、zero ratio、bit sparsity 等特征；
4. 对量化前后的成功率（Success Rate）进行对比；
5. 最终将模型 workload 映射到硬件模型，分析 latency、memory traffic、throughput、sparsity speedup 等指标；
6. 形成一套可复用到其他 VLA / VLM policy 的测试框架。

当前最重要的原则是：**先把仿真与 baseline 复现搞清楚，再做量化。**

---

## 2. 论文基准：SmolVLA

主要参考论文：

- `SmolVLA: A vision-language-action model for affordable and efficient robotics`
- arXiv:2506.01844v1
- 论文日期：2025-06-02

### 2.1 论文主模型结构

论文 0.45B SmolVLA 主模型描述：

- VLM backbone：SmolVLM-2
- 只保留 VLM 前 **16 层**
- Action Expert hidden width：`0.75 × VLM hidden size`
- action chunk size：`50`
- Flow Matching inference steps：`10`
- 总参数约 `450M`
- Action Expert 参数约 `100M`
- VLM frozen
- 主要训练 Action Expert

### 2.2 LIBERO evaluation

论文使用四个标准 LIBERO suite：

- Spatial
- Object
- Goal
- Long

每个 suite 10 个 task，每个 task evaluation 10 次，因此：

```text
1 suite = 10 tasks × 10 episodes = 100 episodes
4 suites = 400 episodes
```

成功率为 binary success rate。

论文中特别说明：

```text
simulation:
每执行 1 个 action 后重新采样 observation，并重新预测 action chunk
```

因此论文式 simulation evaluation 应使用：

```text
n_action_steps = 1
num_steps      = 10
chunk_size     = 50
```

论文 Table 2 中 0.45B SmolVLA：

| Suite | Success Rate |
|---|---:|
| Spatial | 90% |
| Object | 96% |
| Goal | 92% |
| Long | 71% |
| Avg. | 87.3% |

### 2.3 Simulation fine-tuning

论文 Sec. 4.3 描述：

```text
simulation fine-tune steps = 100,000
batch size                 = 64
VLM                        = frozen
Action Expert              = train
BF16
torch.compile()
```

需要注意：论文没有完整给出以下 reproduction 信息：

- exact LeRobot commit
- exact LIBERO / robosuite / MuJoCo version
- exact checkpoint revision
- exact evaluation seed
- exact episode horizon
- reset settle steps
- 完整 simulation fine-tuning command
- fine-tuning scheduler 是否完全复用 pretraining scheduler

因此“严格复现 Table 2”和“复现官方公开 checkpoint 的 LIBERO 性能”必须区分开。

---

## 3. 项目目录与环境

项目根目录：

```text
~/VLA_tcs2
```

目前存在两套主要环境。

---

## 4. 旧环境：论文时期 SmolVLA / 架构研究

Conda：

```text
smolvla
```

LeRobot：

```text
~/VLA_tcs2/lerobot
commit:
5c87365cc160617c45dc5d1bbb3788de010271a7
```

主要版本：

```text
Python       3.10.20
PyTorch      2.6.0+cu124
torchvision  0.21.0+cu124
transformers 4.52.4
accelerate   1.7.0
datasets     3.6.0
```

本地 base checkpoint：

```text
~/VLA_tcs2/checkpoints/smolvla_base
```

曾使用 revision：

```text
d560bdd24ed1230588eac43aeec23fe5beb73089
```

这一环境主要用于：

- 理解 SmolVLA 原始架构；
- 对照论文时期代码；
- 分析 VLM / Action Expert / Flow Matching；
- 不建议作为当前 LIBERO benchmark 主环境。

旧 LeRobot 没有当前完善的 LIBERO 官方 integration。

---

## 5. 当前主环境：LIBERO evaluation

Conda：

```text
smolvla_eval
```

LeRobot：

```text
~/VLA_tcs2/lerobot_current
```

固定 commit：

```text
6adf51511b7625090eade8d82d9f61a1846ebe56
```

LeRobot：

```text
0.6.2
```

主要环境：

```text
Python       3.12.13
torch        2.7.1+cu118
torchvision  0.22.1+cu118
GPU          NVIDIA H100 NVL
```

GPU/CUDA 已验证可正常执行。

### 5.1 LeRobot + LIBERO 安装

使用：

```bash
pip install -e ".[libero]"
```

安装过程中遇到 CMake compatibility 问题：

```text
Compatibility with CMake < 3.5 has been removed
```

解决方法：

1. 使用 conda 安装较新的 CMake（已使用 3.31.8）；
2. 从当前 `lerobot_current/pyproject.toml` 中移除会导致冲突的 Python CMake dependency；
3. 再完成 editable install。

当前已安装并验证：

- hf-libero
- robosuite
- robomimic
- bddl
- mujoco
- egl_probe

`pip check` 已通过。

---

## 6. MuJoCo / EGL 问题

这是目前环境搭建中最关键的问题之一。

### 6.1 OSMesa 不可用

当前服务器使用：

```text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

OSMesa 路径不可用，因此不要切回 OSMesa。

### 6.2 CUDA + EGL 崩溃

早期出现过：

```text
PyTorch H100 CUDA computation
+
LIBERO / MuJoCo EGL rendering
```

运行几十 step 后崩溃。

进一步枚举 EGL device：

```text
device 0 -> NVIDIA
device 1 -> permission/problem
device 2 -> Mesa
```

最终强制：

```text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
MUJOCO_EGL_DEVICE_ID=2
```

后，CUDA + rendering 稳定。

当前环境变量已经存入 conda 环境。

每次恢复实验建议确认：

```bash
printenv MUJOCO_GL
printenv PYOPENGL_PLATFORM
printenv MUJOCO_EGL_DEVICE_ID
```

预期：

```text
egl
egl
2
```

### 6.3 robosuite log 权限

曾出现 `/tmp/robosuite.log` 权限问题。

解决方式：

```text
FILE_LOGGING_LEVEL=None
```

当前 LIBERO smoke test 已稳定通过。

---

## 7. 当前 LIBERO 官方环境行为

当前固定 LeRobot commit 中，LIBERO 默认核心行为：

```text
fps / control_freq = 20 Hz
hard_reset         = True
init_states        = True
control_mode       = relative
num_steps_wait     = 10
```

常用相机：

```text
agentview_image
robot0_eye_in_hand_image
```

当前 EnvProcessor 将它们映射为：

```text
observation.images.image
observation.images.image2
```

当前官方 LIBERO state 由：

```text
eef position      3
eef quaternion -> axis-angle 3
gripper qpos      2
```

组成：

```text
observation.state = 8D
```

current EnvProcessor 还会对 LIBERO image 做 180° rotation，以匹配当前 `HuggingFaceVLA/libero` convention。

当前标准 suite 最大 episode steps：

```text
Spatial   280
Object    280
Goal      300
LIBERO-10 520
```

这些值是当前 LeRobot 实现，不应自动视为论文原始实验环境参数。

---

## 8. 已测试 checkpoint 总览

### 8.1 `lerobot/smolvla_base`

定位：

```text
官方 base checkpoint
不是 LIBERO benchmark checkpoint
```

典型结构：

```text
SmolVLM2-500M-Video-Instruct
num_vlm_layers          = 16
expert_width_multiplier = 0.75
chunk_size              = 50
num_steps               = 10

state  = 6D
camera = camera1/camera2/camera3
```

官网教程主要也是从 `lerobot/smolvla_base` 出发，再对具体任务 fine-tune。

它与论文 0.45B 主架构非常接近，但不能直接用于复现 LIBERO Table 2。

### 8.2 `lerobot/smolvla_libero`

Repo SHA：

```text
31d453f7edd78c839a8bbc39744a292686daf0de
```

结构：

```text
VLM                      SmolVLM2-500M-Video-Instruct
num_vlm_layers           16
num_expert_layers        0
expert_width_multiplier  0.75
chunk_size               50
num_steps                10
state                     6D
camera1/camera2/camera3
action                    7D
```

但是公开训练 config 与论文不一致：

```text
freeze_vision_encoder = False
train_expert_only     = False
steps                 ≈ 25k
batch size            ≈ 32
```

因此：

```text
架构像论文
训练 recipe 不像论文
```

#### 实测 Spatial baseline

使用：

```text
seed=1000
n_action_steps=1
num_steps=10
10 tasks × 10 episodes
```

结果：

| Task | SR |
|---|---:|
| 0 | 100% |
| 1 | 90% |
| 2 | 100% |
| 3 | 100% |
| 4 | 60% |
| 5 | 0% |
| 6 | 100% |
| 7 | 90% |
| 8 | 80% |
| 9 | 90% |
| Overall | **81%** |

Task 5 是明显异常点。

视频人工观察：

```text
机器人通常能接近目标 bowl，
但 grasp 阶段容易把 bowl 推倒，
之后 recovery 较差。
```

因此此前 81% 被暂时作为一个 legacy/public-checkpoint baseline，而不是论文 Table 2 baseline。

### 8.3 `HuggingFaceVLA/smolvla_libero`

Repo SHA：

```text
6721902bc4d61e50a3bfdb11dfb4cb626f05d102
```

实际 config：

```text
vlm_model_name           = HuggingFaceTB/SmolVLM2-500M-Instruct
num_vlm_layers           = 0
base VLM layers          = 32
=> effective VLM layers  = 32
num_expert_layers        = -1
=> effective expert      = 32 layers
expert_width_multiplier  = 0.5
VLM hidden               = 960
expert hidden            = 480
state                     = 8D
image / image2            = 2 cameras
action                    = 7D
train_expert_only         = True
freeze_vision_encoder     = True
n_action_steps            = 1
num_steps                 = 10
```

结论：

```text
它不是论文描述的 16-layer / 0.75-width 模型。
```

但它与当前官方 LeRobot LIBERO processor 的 `8D state + image/image2` 接口完全一致。

#### 已验证 performance

Task 0：

```text
1/1 success
```

Task 5：

```text
9/10 success = 90%
```

Task 5 的 10 次结果：

```text
True True True True False True True True True True
```

因此它非常有力地证明：

```text
当前 LIBERO simulator / EGL / CUDA / official eval pipeline 没有根本性问题。
```

因为同样的 Task 5：

```text
lerobot/smolvla_libero         -> 0/10
HuggingFaceVLA/smolvla_libero  -> 9/10
```

差异主要来自 checkpoint / input convention / training recipe，而不是 simulator 本身。

目前把该模型定义为：

```text
current official LIBERO reference checkpoint
```

而不是：

```text
paper Table-2 exact checkpoint
```

### 8.4 `tiantianx/smolvla_libero`

Repo SHA：

```text
98343cf58d6669cad9686c9251e4c33d18a27a76
```

这是目前最值得继续验证的 **paper-like community reproduction checkpoint**。

实际 config：

```text
vlm_model_name              = HuggingFaceTB/SmolVLM2-500M-Video-Instruct
num_vlm_layers              = 16
num_expert_layers           = 0
expert_width_multiplier     = 0.75
freeze_vision_encoder       = True
train_expert_only           = True
train_state_proj            = True
chunk_size                  = 50
num_steps                   = 10
n_action_steps              = 50   # evaluation 时覆盖为 1
state                       = 6D
camera1/camera2/camera3
action                      = 7D
```

其 repo commit 描述：

```text
SmolVLA finetuned on LIBERO,
100k steps,
batch 64,
from smolvla_base
```

保存的 train config：

```text
steps      = 100000
batch_size = 32
seed       = 1000
pretrained_path = lerobot/smolvla_base
freeze_vision_encoder = True
train_expert_only = True
```

`batch_size=32` 与 commit 中“batch64”不一定矛盾：

```text
如果使用 2 GPU / 2 processes:
32 × 2 = global batch 64
```

但 repo 没有保存 world_size / accelerate config，因此仍未证实。

#### 当前最大疑点

metadata 表面上是：

```text
state = 6D
camera1/camera2/camera3
```

而当前 LIBERO official processor 是：

```text
state = 8D
image/image2
```

camera naming 已通过 rename_map 处理：

```text
image  -> camera1
image2 -> camera2
```

目前没有手动补 camera3。

#### 第一次直接 evaluation

不加 rename_map 时失败：

```text
Missing:
camera1 camera2 camera3
Extra:
image image2
```

错误发生在 policy feature validation，模型权重本身已正常加载。

#### 加 rename_map 后

使用：

```bash
--rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
```

之后：

```text
Spatial Task 0
1 episode
success = 100%
```

说明：

1. 当前 LeRobot 0.6.2 可以正确加载该约 907MB checkpoint；
2. 模型实际按 `16 layers + 0.75 width` 加载；
3. 不需要显式补 camera3 也可以完成 inference；
4. 当前 evaluation 没有因为 config 的 `state=6D` 与 LIBERO 的 `state=8D` 直接报错。

这是当前最重要的 unresolved point。

### 8.5 其他 community checkpoint

已经初步检查：

#### `Alkatt/smolvla-libero-100000`

```text
16 layers
expert width 0.75
100k
VLM frozen
expert-only
8D state
2 cameras
```

优点：与当前 official LIBERO interface 兼容。

疑点：

```text
train config batch size = 4
```

优先级低于 tiantianx。

#### `jadechoghari/smolvla-new-libero`

```text
32 layers
expert width 0.5
8D state
2 cameras
```

属于后来的 official-style architecture，不是论文结构。

#### `jadechoghari/smolvla-libero-ckpts`

当前 LeRobot config parser 会因为：

```text
gradient_accumulation_steps
```

字段不兼容而报错。

公开信息显示同样属于后期 32-layer / 0.5-width 路线。

---

## 9. 当前 checkpoint 分类

建议后续统一使用以下命名，避免混淆。

### A. `legacy_public_checkpoint`

```text
lerobot/smolvla_libero
```

特点：

```text
16 / 0.75
但训练 recipe 非论文
Spatial 已测 81%
```

### B. `official_current_reference`

```text
HuggingFaceVLA/smolvla_libero
```

特点：

```text
32 / 0.5
当前 official LIBERO input contract
Task5 已测 90%
用于验证 simulator / evaluator 正确性
```

### C. `paper_like_reproduction_candidate`

```text
tiantianx/smolvla_libero
```

特点：

```text
16 / 0.75
100k
expert-only
VLM frozen
from smolvla_base
目前最接近论文 recipe
但并非论文作者确认的原始 Table-2 checkpoint
```

---

## 10. 当前最重要的结论

### 10.1 Simulator 基本确认正常

此前最大的疑问是：

```text
为什么论文 Spatial 约 90%，本项目只有 81%？
```

现在通过：

```text
HuggingFaceVLA/smolvla_libero
Task5 = 9/10
```

基本排除了以下因素是主要原因：

- EGL
- CUDA/H100
- MuJoCo rendering
- 当前 LIBERO physics 完全错误
- camera 完全颠倒
- action control 完全错误

如果 simulator 有根本错误，不太可能同一 Task 5 从 0/10 恢复到 9/10。

因此此前 81% 的主要问题更可能是：

```text
checkpoint
+
training recipe
+
observation/input convention
```

---

## 11. 当前未解决的核心问题

### P0：论文 Table 2 原始 checkpoint 是否公开？

截至当前，没有找到一个能够同时被确认满足：

```text
官方/作者来源
+
16 VLM layers
+
expert width 0.75
+
100k LIBERO fine-tune
+
global batch64
+
expert-only
+
明确对应论文 Table 2
```

的公开 checkpoint。

因此必须区分：

```text
严格论文 Table-2 reproduction
```

和：

```text
公开 checkpoint / official port reproduction
```

### P0：tiantianx 的 6D / 8D state 问题

当前 official LIBERO processor：

```text
state = 8D
```

tiantian config：

```text
state = 6D
```

但实际 evaluation：

```text
没有报 shape mismatch
Task0 1/1 success
```

可能原因：

1. normalizer safetensors 实际保存的是 8D stats，而 config metadata 仍是 6D；
2. pipeline 中存在尚未确认的 state transformation；
3. 当前 SmolVLA 对 state 只 pad 到 32，因此 policy 主模型可接受 8D，真正需要检查的是 normalizer；
4. checkpoint metadata 部分继承自 `smolvla_base`，没有完全更新。

**下一步必须审计 normalizer safetensors。**

### P1：第三 camera 的真实训练方式

tiantian policy config：

```text
camera1
camera2
camera3
empty_cameras = 0
```

当前 LIBERO：

```text
camera1 <- image
camera2 <- image2
camera3 missing
```

但 evaluation 成功。

因此当前 inference 似乎并不需要真实 camera3。

仍需确认训练时 `camera3` 是否 empty / masked，还是训练环境中有特殊处理。

### P1：论文环境版本不明确

论文未明确 pin：

- LeRobot commit
- LIBERO fork
- robosuite
- MuJoCo
- reset settle behavior
- exact episode horizon
- exact eval seed

后续如果要写论文级 reproduction，必须将这些作为 limitation 明确记录。

---

## 12. 当前可直接复现的命令

### 12.1 环境检查

```bash
conda activate smolvla_eval
cd ~/VLA_tcs2/lerobot_current

git rev-parse HEAD
python -c "import importlib.metadata; print(importlib.metadata.version('lerobot'))"

printenv MUJOCO_GL
printenv PYOPENGL_PLATFORM
printenv MUJOCO_EGL_DEVICE_ID

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

目标：

```text
commit = 6adf51511b7625090eade8d82d9f61a1846ebe56
lerobot = 0.6.2
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
MUJOCO_EGL_DEVICE_ID=2
```

### 12.2 当前 official reference smoke test

```bash
lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.n_action_steps=1 \
    --policy.num_steps=10 \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids='[0]' \
    --eval.n_episodes=1 \
    --eval.batch_size=1 \
    --env.max_parallel_tasks=1 \
    --seed=1000
```

已经验证成功。

### 12.3 tiantian paper-like checkpoint smoke test

必须加 camera rename：

```bash
lerobot-eval \
    --policy.path=tiantianx/smolvla_libero \
    --policy.n_action_steps=1 \
    --policy.num_steps=10 \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids='[0]' \
    --eval.n_episodes=1 \
    --eval.batch_size=1 \
    --env.max_parallel_tasks=1 \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    --seed=1000
```

已经验证：

```text
Task0 1/1 success
```

---

## 13. 下一步：立即执行的任务

### Step 1：审计 tiantian normalizer

不要马上跑 100 episodes。

先执行：

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

repo = "tiantianx/smolvla_libero"

path = hf_hub_download(
    repo_id=repo,
    filename="policy_preprocessor_step_5_normalizer_processor.safetensors",
)

print("normalizer file =", path)

d = load_file(path)

print("\n=== ALL NORMALIZER TENSORS ===")
for k in sorted(d.keys()):
    v = d[k]
    print(k)
    print("  shape =", tuple(v.shape))
    if v.numel() <= 32:
        print("  value =", v.cpu().tolist())
PY
```

重点确认：

```text
observation.state mean/std
```

到底是 6D 还是 8D。

如果真实 stats 为 8D，则几乎可以确认：

```text
config 的 6D 是 stale/base metadata，
实际 LIBERO training/inference 使用 8D state。
```

### Step 2：对比 LIBERO dataset stats

如果 tiantian normalizer 是 8D：

1. 下载 `HuggingFaceVLA/libero` 的 `meta/stats.json`；
2. 对比 state mean/std；
3. 判断 checkpoint normalization 是否直接来自该 dataset。

如果完全一致，则可以大幅提升该 checkpoint reproduction 的可信度。

### Step 3：验证 Task 5

完成 state/input audit 后：

```text
tiantianx/smolvla_libero
Spatial Task5
10 episodes
seed=1000
```

Task 5 是当前最有区分度的 task：

```text
lerobot/smolvla_libero         -> 0/10
HuggingFaceVLA/smolvla_libero  -> 9/10
```

如果 tiantian 同样达到较高成功率，说明它非常值得作为论文结构 baseline。

### Step 4：跑完整 Spatial

只有前面 audit 通过后再跑：

```text
10 tasks × 10 episodes = 100 episodes
```

目标比较：

```text
paper 0.45B Spatial = 90%
```

需要保存 task-wise SR，而不仅仅保存 overall。

### Step 5：再扩展到 Object / Goal / Long

如果 Spatial 接近论文结果，再依次跑：

```text
Object
Goal
Long
```

最终得到：

```text
Spatial / Object / Goal / Long
```

四套 benchmark。

---

## 14. Reproducibility manifest

从下一轮正式 benchmark 开始，每次实验都建议记录：

```yaml
model_repo:
model_revision:
lerobot_commit:
lerobot_version:
python:
torch:
cuda:
gpu:
libero_version:
robosuite_version:
mujoco_version:
MUJOCO_GL:
PYOPENGL_PLATFORM:
MUJOCO_EGL_DEVICE_ID:
env_task:
task_ids:
episode_length:
fps:
hard_reset:
init_states:
num_steps_wait:
control_mode:
seed:
n_episodes:
batch_size:
max_parallel_tasks:
chunk_size:
n_action_steps:
flow_num_steps:
state_dim:
camera_names:
rename_map:
empty_cameras:
normalizer_revision:
dataset_repo:
dataset_revision:
```

这是后续做量化实验时最重要的基础。

---

## 15. 量化框架：当前设计

量化工作已经有初步架构，但目前暂时暂停，等待 baseline 稳定。

计划文件：

```text
main.py
model_wrapper.py
calibration.py
eval.py
quant_linear.py
quant_matmul.py
stat_manager.py
```

### 15.1 main.py

仅负责 orchestration：

```text
读取 YAML
    ↓
ModelWrapper
    ↓
build model
    ↓
calibration / scale inspection
    ↓
quant_forward or raw
    ↓
evaluate
    ↓
保存结果
```

Main 不应自己：

- 计算 scale；
- 遍历 Linear；
- 操作环境内部逻辑；
- 管理具体 quant formula。

### 15.2 ModelWrapper

负责：

- load pretrained policy；
- module replacement；
- raw / calibration / quant mode switch；
- 选择量化范围：VLM / Action Expert / Vision Encoder / all。

### 15.3 QuantizedLinear

第一阶段优先实现 Linear-only PTQ。

需要支持：

```text
raw
scale_inspection
quant_forward
```

### 15.4 QuantizedMatMul

后续处理：

- Q × K^T
- Attention × V

MatMul 与 Linear 必须分开统计和量化。

### 15.5 StatManager

只作为被动统计器：

- activation min/max
- output min/max
- scale
- zero ratio
- bit sparsity
- tensor shape
- hardware workload metadata

不应负责：env rollout / calibration loop / policy forward orchestration。

---

## 16. 计划中的量化流程

```text
Float checkpoint
      ↓
module replacement
      ↓
scale_inspection
      ↓
calibration
      ↓
保存 scale
      ↓
quant_forward
      ↓
LIBERO evaluation
      ↓
Success Rate degradation
      ↓
activation / bit sparsity statistics
      ↓
hardware model
```

预期实验模式：

```text
raw
scale_inspection
quant_forward
```

第一阶段只量化 Linear，MatMul quantization 暂缓。

---

## 17. 自定义 eval 当前状态

之前已经把官方 `lerobot_eval.py` 逻辑复制到项目 eval 路径，并插入过 identity hook：

```python
policy = apply_model_transform(policy, cfg)
```

hook 当时仅返回原 policy。

已经验证：

```text
复制后的 eval 路径
+
identity transform
```

仍可跑通 Spatial task0。

但当前建议：

> 在 official baseline / paper-like baseline 尚未确定之前，不继续重构自定义 evaluator。

正式量化框架中的 `evaluate()` 最终应该：

```python
evaluate(model, config, output_dir)
```

即 model 从外部传入，而不是 evaluate 内部重新创建模型。

---

## 18. 当前不要做的事情

在 paper-like baseline 确认之前，不建议：

1. 直接修改 checkpoint 的 `num_vlm_layers`；
2. 把 32-layer checkpoint 强行改成 16-layer；
3. 把 expert width 0.5 强行改成 0.75；
4. 人为猜测 8D→6D state conversion；
5. 随意补第三 camera；
6. 修改 control frequency；
7. 修改 reset settle steps；
8. 修改 episode horizon；
9. 同时改变 checkpoint 和 environment；
10. 在 official evaluator 尚未验证时使用自定义 quantized evaluator。

否则一旦成功率变化，很难定位原因。

---

## 19. 推荐实验策略

统一采用：

```text
一次只改变一个因素
```

例如：

```text
checkpoint A
    ↓
固定 environment
    ↓
得到 baseline

只换 checkpoint B
    ↓
比较差异

只改 state adapter
    ↓
比较差异

只改 camera convention
    ↓
比较差异
```

不要一次修改 checkpoint + camera + state + fps + reset + horizon。

---

## 20. 当前项目状态一句话总结

截至 2026-08-20：

> **SmolVLA + LIBERO + H100 + EGL 的当前官方闭环 evaluation 已稳定跑通；此前 81% Spatial 的主要问题已基本定位为 checkpoint / input convention，而非 simulator。当前正在验证 `tiantianx/smolvla_libero` 这一 16-layer / 0.75-width / 100k / expert-only 的 paper-like community checkpoint，它已在 Spatial Task0 上 1/1 成功。下一步优先审计其 normalizer 中实际 state 维度，再决定是否跑完整 LIBERO benchmark。量化工作暂时冻结，待 baseline 固化后恢复。**

---

## 21. 推荐交接顺序

接手后按以下顺序继续：

```text
1. 不更新 LeRobot，保持 commit 6adf515...
2. 检查 EGL 环境变量
3. 审计 tiantian normalizer safetensors
4. 解释 6D metadata vs 8D LIBERO state
5. 确认 camera3 实际行为
6. 跑 tiantian Task5 × 10
7. 跑 tiantian Spatial × 100
8. 若结果合理，再跑 Object / Goal / Long
9. 固化 paper-like FP/BF16 baseline
10. 恢复 QuantizedLinear + calibration 框架
11. 做 PTQ accuracy / sparsity / hardware analysis
```

---

## 22. 关键结果速查

```text
LeRobot current commit:
6adf51511b7625090eade8d82d9f61a1846ebe56

LeRobot version:
0.6.2

GPU:
NVIDIA H100 NVL

EGL:
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
MUJOCO_EGL_DEVICE_ID=2

Paper Spatial:
90%

lerobot/smolvla_libero:
Spatial = 81/100
Task5  = 0/10

HuggingFaceVLA/smolvla_libero:
32-layer / 0.5-width
Task0 = 1/1
Task5 = 9/10

tiantianx/smolvla_libero:
16-layer / 0.75-width
100k / expert-only / frozen VLM
Task0 = 1/1
需要 rename_map
state normalization 维度仍待审计
```

---

## 23. 交接注意事项

- 不要因为 checkpoint 名字中含 `libero` 就默认它对应论文 Table 2；
- 不要只看 `config.json`，还要看 train_config、model tensor shapes、normalizer stats、processor、commit/revision；
- `num_vlm_layers=0` 在当前实现中不是 0 层，而是“不截断”，因此可能实际使用全部 32 层；
- `num_expert_layers<=0` 可能意味着 expert layer 数跟随有效 VLM layer 数；
- 论文模型结构和当前 official LIBERO checkpoint 已经发生明显版本漂移；
- 后续任何量化结果都必须绑定到一个明确固定的 FP/BF16 baseline；
- 成功率是闭环控制指标，不能只用单次 forward numerical error 替代；
- 所有正式 benchmark 建议保留 episode video，方便分析 task-specific failure。
