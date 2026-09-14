# experiments/ —— 实验管理规范

本目录是**所有新实验的唯一入口**。每次实验（或一组同主题的消融实验）在此目录下创建一个实验子目录，统一管理 configs、scripts 与文档。

> 历史上散落在 `configs/experiments/`、`scripts/`、`outputs/`、`doc/logs/` 的一次性文件不再新增；旧实验不强制迁移，新实验一律走本规范。

---

## 1. 实验子目录结构（固定）

```text
experiments/
├── README.md                # 本规范
├── new_experiment.sh        # 快速创建新实验子目录的脚手架脚本
└── <exp_name>/              # 一个实验一个子目录
    ├── configs/             # 该实验用到的所有 YAML 配置（自包含，不引用 configs/experiments/）
    │   └── *.yaml
    ├── scripts/             # 该实验的启动/汇总/绘图脚本（可执行 .sh / .py）
    │   └── run_*.sh
    ├── tasks/               # ★ 子实验目录（可嵌套，见 §7；无子实验时保持为空）
    │   └── <sub_name>/      #   子实验：结构与父实验相同（configs/scripts/docs）
    └── docs/
        ├── experiment_setup.md   # 实验设置（固定结构，创建时填写）
        ├── results.md            # 结果记录（固定结构，实验中/后填写）
        ├── logs.md               # 运行日志/进度状态（固定结构，实验全程更新）
        └── figures/              # 结果图片（*.png / *.pdf，results.md 中相对路径引用）
```

## 2. 命名规范

实验子目录名：`<YYYY-MM-DD>_<phase>_<主题>_<可选:变体>`

- 日期为实验**启动日**；
- `phase` 为项目阶段标识（如 `phaseG`、`ablation`、`repro`、`pilot`）；
- 主题用短横线分隔的小写英文，如 `w4-per-layer`、`fp8-matmul`；
- 同一实验的多轮变体在子目录内用 configs/文件名区分，**不再新建顶层目录**。

示例：

```text
2026-09-12_phaseG_fp8-matmul-pv/
2026-09-15_ablation_w4-down-proj/
```

## 3. 目录职责约定

| 子目录 | 放什么 | 不放什么 |
|---|---|---|
| `configs/` | 实验全部 YAML，按变量命名（如 `alpha003.yaml`、`alpha010.yaml`） | 与本实验无关的历史配置 |
| `scripts/` | `run_*.sh`（批量启动）、`summarize_*.py`（汇总）、`plot_*.py`（绘图） | 通用工具（通用工具放仓库级 `scripts/`） |
| `docs/` | `experiment_setup.md`、`results.md`、`logs.md`、`figures/` | 原始评测数据（原始数据放 `outputs/<exp_name>/`，见 §5） |

## 4. 实验生命周期

1. **创建**：`bash experiments/new_experiment.sh 2026-09-12 phaseG fp8-matmul-pv`
   （或 `bash experiments/new_experiment.sh`，按提示交互输入）
2. **配置**：把 YAML 复制进 `configs/`，脚本放进 `scripts/`；
3. **开跑前**：填写完 `docs/experiment_setup.md`（尤其「变量控制」与「输出目录映射」两节）；
4. **实验中**：每完成一个子实验，立即把结果追加进 `docs/results.md`（含日期、命令、输出目录）；
5. **实验中**：每次启动/完成一次运行，在 `docs/logs.md` 追加一条运行记录（命令、状态、输出目录、后台 PID）；
6. **结束**：补全 `results.md` 的「结论与分析」「后续行动」，图片放入 `docs/figures/` 并在文中引用；
7. **可选**：在实验 `results.md` / `logs.md` 头部状态字段标记 `done / running / aborted`。

## 5. 与 outputs/ 的关系

`outputs/` 保存**原始评测产出**（eval_info.json、videos、log 等）。规则：

- **每个实验在 `outputs/` 下创建与实验子目录同名的目录**：`outputs/<exp_name>/`，该实验的全部原始数据（各 config 的产出、运行日志）都放在其中，方便按实验名直接定位；
- `new_experiment.sh` 会自动创建该目录；
- config 中的 `output_dir` 按此命名，如 `outputs/<exp_name>/<组名>_<变体>/`；
- 实验子目录内**不复制**原始数据，只做两件事：
  1. 在 `experiment_setup.md` 的「输出目录映射」表中记录 `config → outputs/<exp_name>/ 子目录`；
  2. 在 `results.md` 中汇总指标（success rate 等）并链接原始目录。

> 历史实验散落在 `outputs/experiments/`、`outputs/` 根部等位置，不强制迁移；新实验一律走本规则。

## 6. 文档结构规范

两个文档的**章节标题固定不可删**（无内容可写"无"），允许在末尾追加自由章节。模板见 `_template/docs/`：

- `experiment_setup.md`：目的 / 环境与版本 / 模型与数据 / 评测协议 / 实验变量与分组 / 运行命令 / 输出目录映射 / 风险与注意
- `results.md`：摘要 / 总结果表 / 分组结果与分析 / 结论 / 问题与后续 / 修订记录
- `logs.md`：当前状态 / 运行日志 / 后台任务 / 异常与处理 / 修订记录

> `results.md` 记录**结果**（指标、SR、结论），`logs.md` 记录**过程**（运行记录、进度状态、后台任务、异常修复）。两者互补，不互相复制内容。

## 7. 子实验（tasks/，实验嵌套）

当一个"实验"实际包含**多个独立但同主题的子问题**（如同一阶段下的若干独立消融），可拆分为子实验放入 `tasks/`：

- **创建**：`bash experiments/new_experiment.sh --task <exp_name> <sub_name>`
  （`sub_name` 用短横线小写英文，可不带日期——归属由父实验的日期表达）；
- **结构**：子实验与父实验同构（`configs/` + `scripts/` + `docs/`），但不建自己的 `tasks/`，**嵌套仅一层**，避免目录深度失控；
- **原始数据**：子实验的产出放 `outputs/<exp_name>/tasks/<sub_name>/`（挂在父实验名下，保持"实验名 → outputs 同名目录"的对应关系）；
- **文档**：子实验的两个文档同样使用固定模板；父实验的 `results.md` 在「分组结果与分析」中链接各子实验结论，形成汇总视图；
- 父实验若自身无直接 config（纯容器），setup 文档的实验变量与输出映射表可写"见各子实验"。
