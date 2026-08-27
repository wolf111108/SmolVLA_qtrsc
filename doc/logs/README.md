# logs/ 每日工作日志约定

本目录存放**每天的进程与实验日志**，与 `outputs/`（评估的原始产出）互补：

- `outputs/`：`lerobot-eval` 等工具的原始输出（eval_info.json / videos / 汇总 md）
- `logs/`：人写的工作日志——今天跑了什么进程、结果如何、结论与下一步

## 文件命名

```text
YYYY-MM-DD_主要任务概况.md
```

示例：

```text
2026-08-22_tiantianx_libero_benchmark.md
2026-08-23_tiantianx_results_analysis.md
```

约定：

- 一天一条主线任务就一个文件，按当天**最主要**的任务命名；
- 一天有多条独立主线（如同时跑 benchmark + 写量化框架），可拆多个文件；
- 同一天对同一任务的续跑/更新，直接追加到当天文件里，用时间戳标注。

## 建议的文件结构

```markdown
# YYYY-MM-DD 主要任务概况

## 今日任务
- [x] 已完成的事
- [ ] 进行中的事

## 运行进程
| 进程 | PID | 命令 | 输出位置 | 状态 |
|---|---|---|---|---|

## 结果
（数据表 / 关键数字 / 指向 outputs/ 的链接）

## 结论与下一步
- ...

## 备查
（环境变量确认、踩坑记录等）
```

## 记录原则

来自 README_VLA_tcs2_handoff.md 的 reproducibility 原则：

1. 每条正式 benchmark 记录：命令、PID、seed、checkpoint revision、输出目录；
2. 一次只改变一个变量，日志中写清楚这次相对上次**改了什么**；
3. 后台长任务记得记录：启动时间、nohup 日志路径、查看进度的命令、预期耗时；
4. 任务结束后回填结果（哪怕失败）；
5. 涉及进程环境的（EGL/conda env），用 `tr '\0' '\n' < /proc/<PID>/environ` 验证后再记录，不要依赖当前 shell 的 printenv。
