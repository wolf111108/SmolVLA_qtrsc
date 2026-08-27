# VLA_tcs2 新设备环境重建手册

> 生成时间：2026-08-27
> 目标：在新设备上 1:1 重建 SmolVLA + LIBERO 评测环境，并复现 A 类
> `lerobot/smolvla_libero` 在 mujoco 3.3.2 下的 Spatial 成功率 **~84%**。

---

## 0. 验收标准

跑 `scripts/run_phase6_mj332_A_foursuite_10_10.sh`（或本手册第 5 节的单 suite 验证命令），
`libero_spatial` 的 `pc_success` 应达到 **84% 左右**（历史 3.8.1 下为 81%，3.3.2 复测后目标 84%）。

| 验收项 | 目标 |
|---|---|
| mujoco 版本 | **3.3.2**（决定性变量，3.8.1 会掉到 81% 且 Task5 异常） |
| Spatial pc_success | ~84%（84/100） |
| 评测协议 | `n_action_steps=10`, `num_steps=10`, `seed=1000`, 10 tasks × 10 episodes |

---

## 1. 前置条件（硬件）

| 项 | 要求 |
|---|---|
| GPU | NVIDIA，driver 支持 CUDA 11.8+（torch 用 cu118 wheel） |
| 显存 | 单卡 ≥ 24 GB（H100 / A100 / RTX 4090 均可） |
| 系统 EGL | 已安装 Mesa + glvnd（`ls /usr/share/glvnd/egl_vendor.d/` 有 vendor 文件） |
| 磁盘 | ≥ 50 GB（checkpoint + HF 缓存 + outputs 视频） |
| 网络 | 可访问 HuggingFace Hub（匿名或 HF_TOKEN）与 GitHub |

> ⚠️ **无 sudo 权限的机器**：EGL/Mesa 库必须由管理员预装，否则 MuJoCo 渲染无法工作
> （OSMesa 路径在本项目不可用）。

---

## 2. 所需文件清单

### 2.1 代码（git，必迁）

```text
VLA_tcs2/
├── main.py                  # 量化流水线入口
├── pyproject.toml
├── README.md
├── configs/experiments/     # 实验 YAML
├── scripts/                 # ★ 评测/审计/验证脚本
├── src/vla_tcs2/            # ★ 量化框架源码
└── envs/                    # 环境定义（pip freeze / conda yml）
```

迁移方式：`git clone https://github.com/wolf111108/VLA_tcs2.git`（或用 bundle）。

### 2.2 LeRobot（独立 git 仓库，必迁）

```text
lerobot_current/   # 主评测环境，固定 commit 6adf51511b7625090eade8d82d9f61a1846ebe56
```

迁移方式：git clone / bundle 后 `git checkout 6adf51511b7625090eade8d82d9f61a1846ebe56`。

### 2.3 环境定义（必迁，很小）

```text
envs/smolvla_eval_pip_freeze.txt   # 完整依赖清单（含关键版本）
envs/smolvla_eval.yml              # conda 环境参考（含 cmake=3.31.8）
```

### 2.4 大文件（可选，联网可重新下载）

| 内容 | 位置 | 大小 | 是否必需 |
|---|---|---|---|
| 本地 checkpoint 副本 | `checkpoints/` | ~2 GB | 否（评测走 HF 缓存自动下载） |
| 评测结果 | `outputs/` | 数 GB（含视频） | 否（仅需 `eval_info.json` 作对照） |
| HF 模型缓存 | `~/.cache/huggingface/` | 数 GB | 否（首次加载自动下载） |
| LIBERO assets | `~/.cache/libero/` | ~1 GB | 否（首次运行自动下载） |

> 最小迁移集 = 代码 + lerobot_current + envs 三个目录，联网后其余自动补齐。

---

## 3. 关键版本清单（必须逐项对齐）

| 包 | 版本 | 备注 |
|---|---|---|
| python | 3.12 | |
| cmake | 3.31.8 | conda 安装（pyproject 中已移除冲突的 python cmake 依赖） |
| **mujoco** | **3.3.2** | ★ 决定性变量，务必降到此版本 |
| torch | 2.7.1+cu118 | 从 pytorch cu118 源安装 |
| torchvision | 0.22.1+cu118 | |
| robosuite | 1.4.0 | |
| robomimic | 0.2.0 | |
| bddl | 1.0.1 | |
| hf-libero | 0.1.4 | |
| egl-probe | 1.0.2 | |
| hf-egl-probe | 1.0.2 | |
| gymnasium | 1.3.0 | |
| lerobot | 0.6.2 | editable install，commit 6adf515 |

### 关键 checkpoint revision

| repo | revision |
|---|---|
| `lerobot/smolvla_libero`（A 类） | `31d453f7edd78c839a8bbc39744a292686daf0de` |

---

## 4. 环境重建步骤

### 第 1 步：创建 conda 环境

```bash
conda create -n smolvla_eval python=3.12 -y
conda activate smolvla_eval
conda install -y cmake=3.31.8
```

> 环境名统一用 `smolvla_eval`（脚本注释与文档均已固化此名，改会牵连脚本）。

### 第 2 步：安装 torch（cu118）

```bash
pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 第 3 步：恢复 lerobot_current 并安装

```bash
cd ~/VLA_tcs2/lerobot_current
git fetch && git checkout 6adf51511b7625090eade8d82d9f61a1846ebe56
git rev-parse HEAD    # 确认 commit

pip install -e ".[libero]"
```

> 若遇 `Compatibility with CMake < 3.5 has been removed` 报错：
> 确认 `conda install -y cmake=3.31.8` 已生效；仍失败则从 `pyproject.toml`
> 移除 python cmake 依赖（旧机器即如此处理）。

### 第 4 步：★ 降级 mujoco 到 3.3.2（达成 84% 的关键）

```bash
pip install mujoco==3.3.2

# 确认版本
python -c "import mujoco; print(mujoco.__version__)"
# 预期: 3.3.2
```

> `pip install -e ".[libero]"` 会拉取 mujoco 最新版（3.8.1），
> **必须在安装后显式降级**。版本不对会导致 Spatial 掉到 81% 且 Task5 异常。

### 第 5 步：固化环境变量

```bash
conda env config vars set MUJOCO_GL=egl
conda env config vars set PYOPENGL_PLATFORM=egl
conda env config vars set FILE_LOGGING_LEVEL=None
# MUJOCO_EGL_DEVICE_ID 在枚举后设置（见第 6 步）

conda deactivate && conda activate smolvla_eval   # 重新激活生效
```

### 第 6 步：EGL device 枚举

新机器的 EGL device 编号与旧机器（device 2 = Mesa）很可能不同，**必须重新枚举**：

```bash
conda activate smolvla_eval
for i in 0 1 2 3; do
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$i python - <<'PY' 2>/dev/null && echo "device $i: OK" || echo "device $i: FAIL"
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco/>")
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 64, 64)
r.update_scene(d)
print("render ok")
PY
done
```

选一个 OK 的编号（通常 0 或 2），固化：

```bash
conda env config vars set MUJOCO_EGL_DEVICE_ID=<选中的编号>
conda deactivate && conda activate smolvla_eval
printenv MUJOCO_GL PYOPENGL_PLATFORM MUJOCO_EGL_DEVICE_ID
```

---

## 5. 重建后验证（分阶段）

### 阶段 A：版本核对（快）

```bash
conda activate smolvla_eval
cd ~/VLA_tcs2/lerobot_current
git rev-parse HEAD                                   # 6adf515...
python -c "import mujoco; print(mujoco.__version__)" # 3.3.2
python -c "import torch; print(torch.__version__)"   # 2.7.1+cu118
printenv MUJOCO_GL PYOPENGL_PLATFORM MUJOCO_EGL_DEVICE_ID
pip check
```

### 阶段 B：单任务 smoke（~2 分钟）

```bash
lerobot-eval \
    --policy.path=lerobot/smolvla_libero \
    --policy.n_action_steps=10 --policy.num_steps=10 \
    --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
    --eval.n_episodes=1 --eval.batch_size=1 \
    --env.max_parallel_tasks=1 \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    --seed=1000 --output_dir=/tmp/smoke_rebuild
```

成功 → 环境基本就绪；失败 → 排查 EGL/CUDA/驱动（第 7 节）。

### 阶段 C：完整 Spatial（验收，~1.5h）

```bash
cd ~/VLA_tcs2
bash scripts/run_phase6_mj332_A_foursuite_10_10.sh
# 或只跑 spatial（更快）:
# 见 scripts/verify_rebuild.sh 中的 spatial 单跑命令
```

汇总 `libero_spatial` 的 `pc_success`，**目标 ~84%**。

---

## 6. 一键验证脚本

```bash
cd ~/VLA_tcs2
bash scripts/verify_rebuild.sh            # 阶段 A + B（快，~5 分钟）
bash scripts/verify_rebuild.sh spatial    # 阶段 A + B + C（含完整 spatial，~1.5h）
```

---

## 7. 常见问题排查

| 症状 | 原因 | 解决 |
|---|---|---|
| Spatial 只有 81% 且 Task5 异常 | mujoco 是 3.8.1 | `pip install mujoco==3.3.2` |
| `libavutil.so.*: cannot open` | FFmpeg 缺库 | 非致命（回退 pyav 解码），可忽略 |
| MuJoCo 渲染 crash | EGL device 编号错 | 重新枚举（第 6 步） |
| `No API key configured` | 训练时开 wandb 未登录 | 评测不需 wandb，可忽略 |
| OOM | 显存不足 | 换 ≥24GB 卡，或 `batch_size=1`（已默认） |
| `trust_remote_code` 报错 | hub env 未授权 | LIBERO 本地 env 不需要 |

---

## 8. 与旧机器已知差异

1. **EGL device 编号会变**：`MUJOCO_EGL_DEVICE_ID=2` 是旧机器特定值，新机器必须重枚举；
2. **GPU 型号差异**：非 H100 时 pytorch cu118 wheel 仍可用，但成功率有 ±1-2pp 浮点波动，
   84% 目标建议允许 ±3pp 容差；
3. **conda yml build string**：跨机器 `conda env create -f` 可能失败，以本手册手动重建为准。
