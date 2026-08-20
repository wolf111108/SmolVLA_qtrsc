# Migration Guide — VLA_tcs2 项目迁移方案

> 生成时间：2026-08-20
> 目标：将项目从当前服务器（hankh100，H100 NVL）迁移到新设备，并保证
> SmolVLA + LIBERO 评测环境在新机器上 1:1 复现。

---

## 1. 目录现状与体积评估

```text
~/VLA_tcs2/
├── README_VLA_tcs2_handoff.md     # ★ 必迁：项目交接文档（最重要）
├── main.py / pyproject.toml        # ★ 必迁：量化框架入口
├── .gitignore                      # ★ 必迁：新建
├── configs/experiments/            # 当前为空，可迁可不迁
├── scripts/                        # ★ 必迁：所有评测/审计/汇总脚本（约几十 KB）
│   ├── eval.py                     # 自定义 eval（含 identity transform hook）
│   ├── run_lerobot_libero_suites.sh
│   ├── summarize_lerobot_vs_paper.py
│   ├── audit_tiantianx_normalizer.py
│   ├── compare_libero_stats.py
│   └── test_*.py                   # 环境 smoke test
├── src/vla_tcs2/                   # ★ 必迁：量化框架源码（model_wrapper/quant/stats/hardware）
├── lerobot_current/                # ★ 必迁：主 LeRobot（固定 commit 6adf515，git 仓库）
├── lerobot/                        # 可选：旧环境 LeRobot（commit 5c87365，仅架构研究用）
├── checkpoints/                    # ○ 大文件：smolvla_base 两个版本（约 1-2 GB）
├── outputs/                        # ○ 混合：文本结果小，视频大；见下文拆分策略
└── .vscode/                        # 不迁
```

### outputs/ 拆分建议

| 内容 | 处理 |
|---|---|
| `eval_info.json`（各 suite 评分） | ★ 必迁，很小 |
| `audit_tiantianx_normalizer.txt`、`compare_libero_stats.txt` | ★ 必迁，很小 |
| `.gitignore` 已自动忽略 |
| smoke test 目录（`*_smoke/`） | 不迁 |
| `multisuite_nohup.log` 等 | 不迁（已忽略） |

---

## 2. 版本清单（关键复现信息，新机器必须对齐）

### 2.1 主环境（LIBERO 评测，必须对齐）

| 项 | 值 |
|---|---|
| conda env | `smolvla_eval` |
| Python | 3.12.13 |
| LeRobot | 0.6.2，`lerobot_current`，commit `6adf51511b7625090eade8d82d9f61a1846ebe56`，editable install（`pip install -e ".[libero]"`） |
| torch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| GPU | NVIDIA H100 NVL（新机器至少需要 CUDA 11.8+ 的 NVIDIA GPU） |
| CMake | 3.31.8（conda 安装；pyproject 中移除了冲突的 python cmake 依赖） |
| 关键库 | hf-libero, robosuite, robomimic, bddl, mujoco, egl_probe |
| 环境变量（conda env 内固化） | `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`、`MUJOCO_EGL_DEVICE_ID=2`、`FILE_LOGGING_LEVEL=None` |

### 2.2 旧环境（架构研究，可选）

| 项 | 值 |
|---|---|
| conda env | `smolvla` |
| Python / torch | 3.10.20 / 2.6.0+cu124（torchvision 0.21.0, transformers 4.52.4, accelerate 1.7.0, datasets 3.6.0） |
| LeRobot | `lerobot/`，commit `5c87365cc160617c45dc5d1bbb3788de010271a7` |

### 2.3 关键 checkpoint revision（必须记录到新机器）

| repo | revision SHA |
|---|---|
| `lerobot/smolvla_base`（本地亦有副本） | `d560bdd24ed1230588eac43aeec23fe5beb73089` |
| `lerobot/smolvla_libero` | `31d453f7edd78c839a8bbc39744a292686daf0de` |
| `HuggingFaceVLA/smolvla_libero` | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| `tiantianx/smolvla_libero` | `98343cf58d6669cad9686c9251e4c33d18a27a76` |

---

## 3. 迁移步骤

### 第 1 步：在旧机器导出环境定义

```bash
conda activate smolvla_eval
conda env export --no-builds > ~/VLA_tcs2/envs/smolvla_eval.yml
pip list --format=freeze > ~/VLA_tcs2/envs/smolvla_eval_pip_freeze.txt

# （可选）旧环境
conda activate smolvla
conda env export --no-builds > ~/VLA_tcs2/envs/smolvla.yml
pip list --format=freeze > ~/VLA_tcs2/envs/smolvla_pip_freeze.txt
```

> 注：`conda env export` 的 yml 含 cu118 特定 build string，跨机器恢复
> 以 pip freeze 为准更稳；yml 仅作参考。建议同时在新机器直接重装关键包
> （见第 4 节安装顺序）。

同时导出本机 EGL device 枚举，供新机器对照：

```bash
mkdir -p ~/VLA_tcs2/envs
python - <<'PY' > ~/VLA_tcs2/envs/egl_devices_old_machine.txt
import EGL.core.pyopenegl as egl_stub  # 若失败用下面的方法
PY
# 更稳的方式：直接记录已知事实（旧机器: 0=NVIDIA, 1=problem, 2=Mesa）
cat > ~/VLA_tcs2/envs/egl_devices_old_machine.txt <<'EOF'
device 0 -> NVIDIA
device 1 -> permission problem (不可用)
device 2 -> Mesa (MUJOCO_EGL_DEVICE_ID=2 工作正常)
EOF
ls /usr/share/glvnd/egl_vendor.d/ 10/egl-device-list 2>/dev/null || true
```

### 第 2 步：打包（代码走 git，大文件走 tar）

**代码与轻量结果（git）**：

```bash
cd ~/VLA_tcs2
git init
git add -A
git commit -m "snapshot: 2026-08-20 handoff state"
# 推到远端（自建 git server / GitHub private repo / bundle 文件均可）
# 若用 bundle（无需网络）:
git bundle create ~/vla_tcs2.bundle --all
```

`lerobot/` 与 `lerobot_current/` 本身是独立 git 仓库（嵌套），处理方式二选一：

- **方案 A（推荐，保留 commit 可追溯）**：在各自目录 `git bundle create` 或 push 到远端，新机器 clone 后 `git checkout <commit>`；
- **方案 B（简单）**：直接 `cp -r`（含 `.git`），新机器 `git -C lerobot_current rev-parse HEAD` 验证 commit。

**大文件（tar，不走 git）**：

```bash
cd ~
tar --exclude='VLA_tcs2/outputs/**/videos' \
    --exclude='VLA_tcs2/outputs/*_smoke*' \
    --exclude='VLA_tcs2/**/__pycache__' \
    --exclude='VLA_tcs2/**/.pytest_cache' \
    --exclude='VLA_tcs2/lerobot_current/.venv' \
    -czvf vla_tcs2_large_$(date +%Y%m%d).tar.gz \
    VLA_tcs2/checkpoints \
    VLA_tcs2/outputs \
    VLA_tcs2/envs

# 或用 rsync 直连（同网段更快）:
rsync -avP --exclude='**/__pycache__' --exclude='outputs/**/videos' \
    ~/VLA_tcs2/ user@newhost:~/VLA_tcs2/
```

**HF 缓存（可选，新机器联网可重新下载）**：

```bash
du -sh ~/.cache/huggingface  # 先看体积再决定
# 需要迁移时:
rsync -avP ~/.cache/huggingface user@newhost:~/.cache/
# LIBERO assets:
rsync -avP ~/.cache/libero user@newhost:~/.cache/
```

> checkpoint 本地副本 (`checkpoints/`) 主要用于离线分析；评测时是从
> HF Hub 拉取的缓存，两者都迁最稳妥，只迁其一也能跑（联网时会自动补）。

### 第 3 步：新机器环境重建

```bash
# 1) conda 环境
conda create -n smolvla_eval python=3.12 -y
conda activate smolvla_eval
conda install -y cmake=3.31.8

# 2) CUDA/驱动检查
nvidia-smi   # 需要 >= driver for CUDA 11.8

# 3) LeRobot editable 安装（先恢复 lerobot_current 并锁定 commit）
cd ~/VLA_tcs2/lerobot_current
git fetch && git checkout 6adf51511b7625090eade8d82d9f61a1846ebe56
pip install -e ".[libero]"   # 如遇 cmake 冲突，参考 README 第 5.1 节处理

# 4) 固化环境变量到 conda env
conda env config vars set MUJOCO_GL=egl
conda env config vars set PYOPENGL_PLATFORM=egl
conda env config vars set FILE_LOGGING_LEVEL=None
# MUJOCO_EGL_DEVICE_ID 需要重新枚举（见第 5 步），先不设

# 5) EGL device 枚举（新机器的编号可能不同！）
python - <<'PY'
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import ctypes.util
for i in range(4):
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(i)
    try:
        import mujoco
        e = mujoco.EGLContext()
        print(f"device {i}: OK")
    except Exception as ex:
        print(f"device {i}: FAIL ({type(ex).__name__})")
PY
# 选一个 OK 的（通常 0=NVIDIA 直连或 Mesa 可用），然后:
conda env config vars set MUJOCO_EGL_DEVICE_ID=<n>

# 6) 验证
pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 第 4 步：迁移后验证清单（按序执行）

```bash
conda activate smolvla_eval
cd ~/VLA_tcs2/lerobot_current

# (a) commit 一致
git rev-parse HEAD
# 预期: 6adf51511b7625090eade8d82d9f61a1846ebe56

# (b) EGL 环境变量
printenv MUJOCO_GL PYOPENGL_PLATFORM MUJOCO_EGL_DEVICE_ID
# 预期: egl egl <新机器枚举出的编号>

# (c) CUDA + mujoco
python -c "import mujoco; print(mujoco.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# (d) smoke test: 官方 reference checkpoint 单任务
lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.n_action_steps=1 --policy.num_steps=10 \
    --env.type=libero --env.task=libero_spatial --env.task_ids='[5]' \
    --eval.n_episodes=1 --eval.batch_size=1 \
    --env.max_parallel_tasks=1 --seed=1000
# 预期: 成功（该模型 Task5 曾 9/10，1 集成功概率很高）
# 若失败 -> 新机器 EGL/CUDA 配置问题，先解决再继续

# (e) 已有结果文件齐全
ls ~/VLA_tcs2/outputs/*eval_info.json
python ~/VLA_tcs2/scripts/summarize_lerobot_vs_paper.py  # 应能读出已有数据
```

### 第 5 步：补跑被中断的 multisuite 评测

旧机器上 object/goal/long 评测已被中断且未保存，新机器恢复后直接：

```bash
cd ~/VLA_tcs2 && conda activate smolvla_eval
nohup bash scripts/run_lerobot_libero_suites.sh \
    > outputs/multisuite_nohup.log 2>&1 &
```

脚本自动从头跑三个 suite（无已完成结果可跳过）。

---

## 4. 新旧机器差异风险点

1. **EGL device 编号会变**：`MUJOCO_EGL_DEVICE_ID=2` 是旧机器特定值，新机器必须重新枚举（第 3 步 5）；
2. **GPU 型号差异**：若不是 H100，pytorch cu118 wheel 仍可用，但成功率的微小波动可能来自 GPU 浮点差异——重要结论前建议在新机器重跑一次 Spatial 对比（预期 81% ± 若干 pp）；
3. **没有 sudo 的机器**：EGL/Mesa libraries 需已由管理员装好，否则 MuJoCo 渲染无法工作（ycs 上无法用软件渲染替代，OSMesa 不可用）；
4. **conda yml 的 build strings**：跨机器 `conda env create -f` 可能失败，以第 3 节手动重建 + pip freeze 对照为准；
5. **共享服务器**：注意 GPU 显存竞争（如 hhhuang 的并行任务），启动大评测前 `nvidia-smi` 检查。

---

## 5. 必须人工携带的信息（无法从文件恢复）

- HF Hub 访问：目前使用匿名访问（有 rate limit警告），如有 HF_TOKEN 建议在新机器配置；
- 两个 conda env 的名字（smolvla_eval / smolvla）已固化在所有文档和脚本注释中，新机器建议同名，避免改脚本。

---

## 6. 快速命令总结（旧机器执行）

```bash
mkdir -p ~/VLA_tcs2/envs
# 环境导出
(conda activate smolvla_eval && conda env export --no-builds > ~/VLA_tcs2/envs/smolvla_eval.yml && pip freeze > ~/VLA_tcs2/envs/smolvla_eval_pip_freeze.txt)
(conda activate smolvla && conda env export --no-builds > ~/VLA_tcs2/envs/smolvla.yml && pip freeze > ~/VLA_tcs2/envs/smolvla_pip_freeze.txt)

# git 快照（代码）
cd ~/VLA_tcs2 && git init && git add -A && git commit -m "snapshot 2026-08-20"

# LeRobot 仓库 bundle（可选替代 clone）
(cd lerobot_current && git bundle create ~/lerobot_current.bundle --all)
(cd lerobot && git bundle create ~/lerobot_old.bundle --all)

# 大文件打包
cd ~ && tar --exclude='VLA_tcs2/outputs/**/videos' --exclude='VLA_tcs2/outputs/*_smoke*' \
    -czvf vla_tcs2_large.tar.gz VLA_tcs2/checkpoints VLA_tcs2/outputs VLA_tcs2/envs
```
