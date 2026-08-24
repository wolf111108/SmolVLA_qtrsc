#!/usr/bin/env bash
# ============================================================
# VLA_tcs2 新机器恢复脚本（在 lfwang@10.113.225.67 上执行）
# 前置条件: ~/ 下已有三个迁移文件
#   - vla_tcs2_large.tar.gz
#   - lerobot_current.bundle
#   - lerobot_old.bundle
# 用法: bash restore_on_new_machine.sh [step]
#   不带参数 = 执行 step1+step2（文件恢复，安全，不需要 sudo）
#   step3/step4 需要人工介入（conda 环境 + EGL 枚举），脚本只打印指导
# ============================================================
set -euo pipefail

LEROBOT_CURRENT_COMMIT=6adf51511b7625090eade8d82d9f61a1846ebe56
LEROBOT_OLD_COMMIT=5c87365cc160617c45dc5d1bbb3788de010271a7
REPO=https://github.com/wolf111108/VLA_tcs2.git

step1_verify() {
    echo "=== Step 1: 校验迁移文件完整性 ==="
    ls -lh ~/vla_tcs2_large.tar.gz ~/lerobot_current.bundle ~/lerobot_old.bundle
    echo ""
    echo "请在旧机器(hankh100)上运行 md5sum 对比:"
    echo "  md5sum ~/vla_tcs2_large.tar.gz ~/lerobot_current.bundle ~/lerobot_old.bundle"
    echo "并在本机运行同样命令, 核对一致后继续。"
    echo ""
    echo "tar 内容预览(前10项):"
    tar -tzf ~/vla_tcs2_large.tar.gz | head -10
    echo ""
    echo "磁盘空间:"
    df -h ~ | tail -1
}

step2_restore() {
    echo "=== Step 2: 恢复目录结构 ==="
    # 2.1 解压大文件 -> ~/VLA_tcs2/{checkpoints,outputs,envs}
    cd ~
    if [[ -d ~/VLA_tcs2 ]]; then
        echo "[warn] ~/VLA_tcs2 已存在, tar 将合并覆盖同名文件"
    fi
    tar -xzf ~/vla_tcs2_large.tar.gz -C ~/
    echo "[ok] checkpoints/outputs/envs 已恢复到 ~/VLA_tcs2/"

    # 2.2 恢复 git 代码仓库(与新clone的代码合并)
    cd ~/VLA_tcs2
    if [[ -d .git ]]; then
        echo "[skip] ~/VLA_tcs2 已是 git 仓库, 仅拉取最新"
        git pull --ff-only || true
    else
        git init
        git remote add origin "$REPO"
        git fetch origin
        # tar 解压可能已带 outputs 下的同名结果文件, 强制用 git 版本检出(内容一致)
        git checkout -f -b main origin/main 2>/dev/null || git checkout -f -t origin/main
        echo "[ok] 代码已从 GitHub 恢复并与大文件目录合并"
    fi

    # 2.3 恢复 lerobot_current(主评测环境, 固定 commit)
    if [[ ! -d ~/VLA_tcs2/lerobot_current/.git ]]; then
        git clone ~/lerobot_current.bundle ~/VLA_tcs2/lerobot_current
    fi
    git -C ~/VLA_tcs2/lerobot_current checkout "$LEROBOT_CURRENT_COMMIT"
    echo "[ok] lerobot_current @ $LEROBOT_CURRENT_COMMIT"
    echo "     验证: git -C ~/VLA_tcs2/lerobot_current rev-parse HEAD"

    # 2.4 恢复 lerobot(旧架构研究环境, 可选)
    if [[ ! -d ~/VLA_tcs2/lerobot/.git ]]; then
        git clone ~/lerobot_old.bundle ~/VLA_tcs2/lerobot
    fi
    git -C ~/VLA_tcs2/lerobot checkout "$LEROBOT_OLD_COMMIT"
    echo "[ok] lerobot(旧) @ $LEROBOT_OLD_COMMIT"

    echo ""
    echo "最终结构:"
    ls ~/VLA_tcs2/
}

step3_env_guide() {
    cat <<'EOF'
=== Step 3: conda 环境重建 (手动执行) ===

# 1) 创建环境
conda create -n smolvla_eval python=3.12 -y
conda activate smolvla_eval
conda install -y cmake

# 2) 检查 GPU/驱动 (需支持 CUDA 11.8+)
nvidia-smi

# 3) LeRobot editable 安装
cd ~/VLA_tcs2/lerobot_current
pip install -e ".[libero]"
# 若遇 CMake < 3.5 兼容性报错:
#   conda install -y cmake=3.31.8 后重试;
#   仍失败则从 pyproject.toml 移除 cmake 的 python dependency (旧机器即如此处理)

# 4) 固化环境变量 (EGL_DEVICE_ID 等 step4 枚举后再设)
conda env config vars set MUJOCO_GL=egl
conda env config vars set PYOPENGL_PLATFORM=egl
conda env config vars set FILE_LOGGING_LEVEL=None

# 5) 版本对照 (差异大时逐个 pip install 对齐)
pip list --format=freeze > /tmp/new_pip.txt
diff ~/VLA_tcs2/envs/smolvla_eval_pip_freeze.txt /tmp/new_pip.txt || true
EOF
}

step4_egl_guide() {
    cat <<'EOF'
=== Step 4: EGL device 枚举 + 验证 (手动执行) ===

# 1) 枚举可用 EGL device (新机器编号与旧机器很可能不同!)
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

# 2) 选一个 OK 的编号固化 (假设是 X):
conda env config vars set MUJOCO_EGL_DEVICE_ID=X
# 重新 activate 使生效:
conda deactivate && conda activate smolvla_eval
printenv MUJOCO_GL PYOPENGL_PLATFORM MUJOCO_EGL_DEVICE_ID

# 3) 下载 LIBERO assets (首次运行自动下载, 也可预下载)

# 4) Smoke test: 官方 reference checkpoint 单任务(HuggingFaceVLA/smolvla_libero Task5 曾 9/10)
cd ~/VLA_tcs2/lerobot_current
lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.n_action_steps=1 --policy.num_steps=10 \
    --env.type=libero --env.task=libero_spatial --env.task_ids='[5]' \
    --eval.n_episodes=1 --eval.batch_size=1 \
    --env.max_parallel_tasks=1 --seed=1000
# 成功 -> 环境迁移完成
# 失败 -> 按 MIGRATION_GUIDE.md 第4节排查 (EGL/驱动/CUDA)

# 5) 恢复被中断的 multisuite 评测:
# cd ~/VLA_tcs2 && nohup bash scripts/run_lerobot_libero_suites.sh \
#     > outputs/multisuite_nohup.log 2>&1 &
EOF
}

case "${1:-}" in
    step1) step1_verify ;;
    step2) step2_restore ;;
    step3) step3_env_guide ;;
    step4) step4_egl_guide ;;
    *)
        step1_verify
        echo ""
        step2_restore
        echo ""
        step3_env_guide
        echo ""
        step4_egl_guide
        ;;
esac
