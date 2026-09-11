# 实验设置（Experiment Setup）

> 本文档在实验开跑**前**填写。章节为固定结构，不可删除；无内容写「无」。

- **实验名称**：<exp_name>
- **状态**：draft（draft / running / done / aborted）
- **负责人**：
- **创建日期**：YYYY-MM-DD
- **相关前序实验**：无（链接相关 experiments/ 子目录的 setup 文档）

---

## 1. 实验目的

<!-- 要回答的科学问题 / 要验证的假设，1-3 条 -->

-

## 2. 环境与版本

| 项目 | 值 |
|---|---|
| conda 环境 | smolvla_eval |
| 仓库 revision / commit | `git rev-parse HEAD` 的值 |
| LeRobot 路径与 commit | `lerobot_current/` @ |
| GPU | |
| 关键依赖版本 | torch / mujoco 等（如有非默认版本） |

## 3. 模型与数据

| 项目 | 值 |
|---|---|
| policy checkpoint | （路径 + checkpoint 分类：legacy_public / official / paper_like / 自训） |
| 评测基准 | （LIBERO spatial / goal / object / 10，或 Meta-World 任务集） |
| 校准数据 | （episodes 数、来源，若复用注明 scale 文件路径） |

## 4. 评测协议

| 项目 | 值 |
|---|---|
| episodes × seeds | （如 10 tasks × 10 eps，seed=1000） |
| 采样参数 | （n_action_steps / chunk_size / num_steps 等） |
| 指标 | （success rate 等） |

## 5. 实验变量与分组

<!-- 本实验的核心变量是什么、控制了哪些变量。每个 config 对应一行。 -->

| 组 | config（configs/ 下文件名） | 变量取值 | 固定项 | 说明 |
|---|---|---|---|---|
| baseline | xxx.yaml | 无量化（raw） | | |
| exp1 | xxx.yaml | | | |

## 6. 运行命令

<!-- 每个 config 一条可复制执行的命令，标注预期输出目录 -->

```bash
# baseline
python main.py --config experiments/<exp_name>/configs/xxx.yaml

# 批量（如有）
bash experiments/<exp_name>/scripts/run_all.sh
```

## 7. 输出目录映射

<!-- config → outputs/<exp_name>/ 下的实际产出目录（与实验同名，见 README §5），跑完后逐一登记，便于回溯原始数据 -->

| config | 输出目录（outputs/ 下） | 状态 |
|---|---|---|
| xxx.yaml | outputs/<exp_name>/xxx/ | pending / done / failed |

## 8. 风险与注意事项

<!-- 已知坑：渲染环境变量、checkpoint 特殊处理、随机性来源等 -->

-
