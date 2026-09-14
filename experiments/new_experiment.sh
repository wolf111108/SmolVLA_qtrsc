#!/usr/bin/env bash
# 创建一个新的实验子目录（含标准骨架文件）
# 用法:
#   bash experiments/new_experiment.sh <YYYY-MM-DD> <phase> <topic>   # 顶层实验
#   bash experiments/new_experiment.sh                                 # 交互式
#   bash experiments/new_experiment.sh --task <exp_name> <sub_name>    # 子实验（tasks/ 下，嵌套一层）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/_template"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="top"
if [[ "${1:-}" == "--task" ]]; then
  MODE="task"
  shift
  if [[ $# -ne 2 ]]; then
    echo "用法: $0 --task <exp_name> <sub_name>"; exit 1
  fi
  PARENT_NAME="$1"; SUB_NAME="$2"
fi

if [[ "$MODE" == "top" && $# -eq 3 ]]; then
  DATE="$1"; PHASE="$2"; TOPIC="$3"
elif [[ "$MODE" == "top" ]]; then
  read -r -p "实验启动日期 [YYYY-MM-DD，回车取今天]: " DATE
  DATE="${DATE:-$(date +%F)}"
  read -r -p "阶段标识 [如 phaseG / ablation / repro / pilot]: " PHASE
  read -r -p "实验主题 [短横线小写英文，如 fp8-matmul-pv]: " TOPIC
fi

# --- 校验 ---
if [[ "$MODE" == "top" ]]; then
  [[ "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "日期格式应为 YYYY-MM-DD: $DATE"; exit 1; }
  [[ -n "$PHASE" && "$PHASE" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "phase 非法: $PHASE"; exit 1; }
  [[ -n "$TOPIC" && "$TOPIC" =~ ^[a-z0-9-]+$ ]] || { echo "topic 应为小写英文/数字/短横线: $TOPIC"; exit 1; }
  EXP_NAME="${DATE}_${PHASE}_${TOPIC}"
  EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
  OUT_DIR="${REPO_ROOT}/outputs/${EXP_NAME}"
  DOC_EXP_NAME="${EXP_NAME}"
else
  [[ -n "$PARENT_NAME" && -d "${SCRIPT_DIR}/${PARENT_NAME}" ]] || { echo "父实验不存在: ${SCRIPT_DIR}/${PARENT_NAME}"; exit 1; }
  [[ -n "$SUB_NAME" && "$SUB_NAME" =~ ^[a-z0-9-]+$ ]] || { echo "sub_name 应为小写英文/数字/短横线: $SUB_NAME"; exit 1; }
  [[ -d "${SCRIPT_DIR}/${PARENT_NAME}/tasks" ]] || { echo "父实验目录下无 tasks/（旧结构实验），请先手动创建"; exit 1; }
  EXP_NAME="${PARENT_NAME}/tasks/${SUB_NAME}"
  EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
  # 子实验不建自己的 tasks/（嵌套仅一层）；产出挂父实验名下
  OUT_DIR="${REPO_ROOT}/outputs/${PARENT_NAME}/tasks/${SUB_NAME}"
  DOC_EXP_NAME="${PARENT_NAME} · 子实验 ${SUB_NAME}"
fi

if [[ -e "$EXP_DIR" ]]; then
  echo "已存在: $EXP_DIR"; exit 1
fi

# --- 创建骨架 ---
if [[ "$MODE" == "top" ]]; then
  mkdir -p "${EXP_DIR}/configs" "${EXP_DIR}/scripts" "${EXP_DIR}/docs/figures" "${EXP_DIR}/tasks"
else
  mkdir -p "${EXP_DIR}/configs" "${EXP_DIR}/scripts" "${EXP_DIR}/docs/figures"
fi
# 原始产出目录：与实验同名（子实验挂父实验名下），见 experiments/README.md §5/§7
mkdir -p "$OUT_DIR"

DATE_NOW="$(date +%F)"
sed -e "s/<exp_name>/${DOC_EXP_NAME}/g" \
    -e "s/YYYY-MM-DD/${DATE_NOW}/g" \
    "${TEMPLATE_DIR}/docs/experiment_setup.md" > "${EXP_DIR}/docs/experiment_setup.md"
sed -e "s/<exp_name>/${DOC_EXP_NAME}/g" \
    -e "s/YYYY-MM-DD/${DATE_NOW}/g" \
    "${TEMPLATE_DIR}/docs/results.md" > "${EXP_DIR}/docs/results.md"
sed -e "s/<exp_name>/${DOC_EXP_NAME}/g" \
    -e "s/YYYY-MM-DD/${DATE_NOW}/g" \
    "${TEMPLATE_DIR}/docs/logs.md" > "${EXP_DIR}/docs/logs.md"

echo "已创建实验目录: ${EXP_DIR}"
echo "已创建产出目录: ${OUT_DIR#${REPO_ROOT}/}"
if [[ "$MODE" == "top" ]]; then
  echo "下一步:"
  echo "  1. 复制/编写 configs 到 ${EXP_NAME}/configs/"
  echo "  2. 编写 scripts 到 ${EXP_NAME}/scripts/"
  echo "  3. 开跑前填写 docs/experiment_setup.md"
  echo "  4. 实验全程在 docs/logs.md 记录运行进度与后台任务"
  echo "  5. 跑完后更新 docs/results.md（状态改为 done）"
  echo "  6. （可选）拆分子实验: bash experiments/new_experiment.sh --task ${EXP_NAME} <sub_name>"
else
  echo "下一步:"
  echo "  1. 复制/编写 configs 到 ${EXP_NAME}/configs/"
  echo "  2. 编写 scripts 到 ${EXP_NAME}/scripts/"
  echo "  3. 开跑前填写 docs/experiment_setup.md"
  echo "  4. 实验全程在 docs/logs.md 记录运行进度与后台任务"
  echo "  5. 跑完后更新 docs/results.md，并在父实验 results.md 中链接结论"
fi
