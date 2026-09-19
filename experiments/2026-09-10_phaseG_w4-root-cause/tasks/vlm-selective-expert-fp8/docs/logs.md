# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**（区别于 `results.md` 的结果记录）。
> 每次启动/完成一次运行、发现或修复异常时追加一条；章节为固定结构，不可删除；无内容写「无」。
> 运行日志按时间**正序**追加，最新状态反映在「当前状态」与头部字段。

- **实验名称**：2026-09-10_phaseG_w4-root-cause · 子实验 vlm-selective-expert-fp8
- **状态**：done（draft / running / done / aborted）
- **最后更新**：2026-09-19

---

## 1. 当前状态

**实验已完成（done）。** Gate 0-5 全部 PASS（2026-09-14）；Gate 6（Goal ×100 四组）于 2026-09-16 全部完成，SR = **90 / 69 / 41 / 21**（A/B/C/D）。三项预定义判据（|ΔL| ≤ 6pp）全部通过 —— L_attn 21（G5 18，+3）、L_mlp 49（G5 51，−2）、L_all 69（G5 69，0）→ **「VLM MLP > Attention 敏感性」对 Expert-FP8 背景鲁棒**。结果已回填 `results.md`。

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| Gate 0 py_compile | ✅ PASS | 2026-09-14 | — |
| Gate 1 routing resolver 单测 | ✅ PASS | 2026-09-14 | 8 个单测（含 target typo fail-fast） |
| Gate 2 四组 routing count | ✅ PASS | 2026-09-14 / 09-15 复验 | A 224/0、B 160/64、C 176/48、D 112/112，MatMul 64 |
| Gate 3 raw Linear equivalence | ✅ PASS | 2026-09-14 | 224 Linear 与 `F.linear` bit-exact，max\|diff\|=0 |
| Gate 4 calibration-only smoke | ✅ PASS | 2026-09-14 | 四组各 864 scale，无 NaN/Inf |
| Gate 5 task0×1 smoke | ✅ PASS | 2026-09-14 | 四组各 SR=100%（1/1） |
| Gate 6-A Goal ×100 | ✅ done | 2026-09-14 18:11 | **SR = 90.0%**（90/100，eval 2.06h） |
| Gate 6-B Goal ×100 | ✅ done | 2026-09-15 16:13 | **SR = 69.0%**（69/100，eval 5.07h） |
| Gate 6-C Goal ×100 | ✅ done | 2026-09-16 03:25 | **SR = 41.0%**（41/100，eval 11.18h） |
| Gate 6-D Goal ×100 | ✅ done | 2026-09-16 13:57 | **SR = 21.0%**（21/100，eval 10.53h） |
| 结题回填 | ✅ done | 2026-09-19 | results.md 状态改 done + L 值判据 + 逐 task 明细 |

## 2. 运行日志

| 日期 | 阶段 | 命令 / 配置 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-09-14 | 脚手架 | 创建 configs/scripts/docs | ✅ | — | 4 config + 7 scripts |
| 2026-09-14 | Gate 0-1 | `py_compile` + routing resolver 单测 | ✅ | — | 8 单测 PASS |
| 2026-09-14 | Gate 2 | `audit_routing.py --config <4 config>` | ✅ | `routing/` | A 224/0、B 160/64、C 176/48、D 112/112 |
| 2026-09-14 | Gate 3 | `gate3_raw_equiv.py` | ✅ | `generated/` | max\|diff\|=0 |
| 2026-09-14 | Gate 4 | `run_smoke.sh`（calibration-only） | ✅ | `scales/.../<name>/` | 四组各 864 scale，无 NaN/Inf |
| 2026-09-14 | Gate 5 | `run_gate5_smoke.sh`（task0×1） | ✅ | `gate5_task0x1/` | 四组 SR=100% |
| 2026-09-14 15:59–16:04 | Gate 6 (attempt 1) | `run_goal.sh` | ⚠️ 部分 | `g6_goal_run.log` | **仅 G6-A 完成**；G6-B 启动即 `SyntaxError`，`set -e` 中止整条 sweep（见 §4） |
| 2026-09-14 16:07–18:11 | Gate 6-A | `g6a_all_fp8_control.yaml --skip-calibration` | ✅ | `g6a_all_fp8_control/` | **SR = 90.0%**（90/100），eval 7429.7s；逐 task t6=7、t3/t9=8 |
| 2026-09-15 11:01 | 前置复验 | `compileall src/vla_tcs2 main.py` + 入口 import + scale 目录核对 | ✅ | — | 编译 OK、import OK、四组 scale 各 864 文件 |
| 2026-09-15 11:02 | Gate 2 复验 | `audit_routing.py --config g6b/c/d` | ✅ | `routing/` | 三组 ROUTING GATE: PASS |
| 2026-09-15 11:05 → 11:08 | Gate 6 (attempt 2) | `run_goal.sh`（已改 fail-loud + 每组 tee） | ⚠️ 主动中止 | `g6_goal_run2.log` | `conda run` 捕获缓冲致 `<out>/run.log` 为 0 字节；加 `--no-capture-output` 后重启（见 §4） |
| 2026-09-15 11:09 → | Gate 6 (attempt 3) | `run_goal.sh`（流式日志版） | 🔄 running | `g6_goal_run2.log` + 各组 `run.log` | A 自动 `[SKIP]`；G6-B 11:09:14 起跑，流式日志已验证 |
| 2026-09-15 12:37 | 巡检 | G6 + H3 并发状态核查 | ✅ | — | G6-B 3/10 tasks（24/30 = 80.0%）；发现 H3 日志缓冲问题，见 §4.4 |
| 2026-09-15 14:37 | 巡检 | G6 + H3 并发状态核查 | ✅ | — | G6-B 8/10 tasks（55/80 = 68.8%）；H3 s0 7/10（62/70 = 88.6%）；**G6-B 前 8 task 与 G5-B 累计完全相同** |

## 3. 后台任务

| PID | 阶段 | 启动时间 | 状态 | 备注 |
|---|---|---|---|---|
| 51969 | Gate 6-B/C/D | 2026-09-15 11:09 | 🔄 running | `nohup bash .../run_goal.sh > .../g6_goal_run2.log`；串行 B→C→D |
| — | — | — | — | **进度（14:37）**：B 8/10 tasks；单 task 15.7→32.1 min（单调上升） |
| — | — | — | — | **ETA**：B 剩 2 tasks × ~32 min ≈ 15:45；C/D 各 ~5h → 预计 2026-09-16 00:00–02:00 全部完成（受并发挤压，可能更晚） |
| 8120 | Phase H · H3 100ep | 2026-09-15 10:39 | 🔄 running | `run_h3_100ep.sh`（s0 10 tasks → s1 10 tasks）；**非本实验**，但与 G6 争抢同一张 H100 NVL 0 |
| — | — | — | — | **进度（14:37）**：s0 7/10 tasks（62/70 = 88.6%），当前 task07 |
| — | — | — | — | **ETA**：s0 剩 3 tasks ≈ 1.6h → ~16:15；随后 s1 10 tasks ≈ 5.5–9h → 预计 2026-09-15 22:00 – 09-16 01:00 |

> **GPU 竞争**：H100 NVL 0 上同时有 G6 与 H3 两个 `main.py`（显存 8949 MiB / 95830 MiB，利用率 72–99%）。显存充裕，但算力互为稀释——G6-A 独占时 12.4 min/task，并发时升至 15.7–32.1 min/task（约 1.3–2.6×）。两者预计都在 09-16 凌晨收尾，期间单 task 耗时会持续走高。

## 4. 异常与处理

### 4.1 G6-B 启动即 `SyntaxError` → 整条 sweep 静默中止（2026-09-14）

- **现象**：`g6_goal_run.log` 末尾停在 `[RUN] g6b_vlm_attn_w4_expert_fp8`，无任何后续输出；G6-B/C/D 输出目录仅剩 `config.yaml`。
- **根因**：G6-A 长时间 rollout（16:07–18:11）期间，仓库内共享模块 `src/vla_tcs2/quant/stat_manager.py` 被编辑到**中间状态**；G6-B 启动时 `import` 该模块直接失败：
  ```
  File ".../src/vla_tcs2/quant/stat_manager.py", line 727
      def _collect_one_tensor_sparsity(
  SyntaxError: '(' was never closed
  ```
  叠加 `run_goal.sh` 原有的 `set -euo pipefail`，单组失败直接终止整个 for 循环 ⇒ C/D 连坐未跑。
- **判读陷阱**：日志中该 traceback 出现在 G6-A 完成块**之前**，这是 `conda run` 的 stdout 块缓冲导致 stderr 与 stdout 错序。**不能凭文件行号判断崩溃点**，应依据 `[RUN]` 标记 + 各组 `result.json` 是否存在。当时该文件现已由 Phase H 修复（commit `e4034aa`），`py_compile` 通过。
- **修复**：
  1. `run_goal.sh` 由 `set -euo pipefail` 改为 **`set -uo pipefail`（去掉 `-e`）+ 收集 `failed[]` + 末尾非零退出**，实现「单组失败不连坐，但整体仍有非零退出码」的 fail-loud 语义；
  2. 每组输出独立 `tee` 到 `<out>/run.log`，并在标记行附时间戳（`[RUN]/[DONE]/[FAIL] (YYYY-MM-DD HH:MM:SS)`）。
- **预防**：重启前新增 `compileall src/vla_tcs2 main.py` + 入口 import 预检作为标准前置步骤（已执行，PASS）。

### 4.2 每组日志 0 字节 / 输出错序 → 改用 `--no-capture-output`（2026-09-15）

- **现象**：11:05 的 attempt 2 启动后，`g6b_vlm_attn_w4_expert_fp8/run.log` 持续为 **0 字节**，而进程健康、GPU 利用率 99%。
- **根因**：`conda run` 默认**捕获并块缓冲**子进程的 stdout/stderr，仅在结束时整体 flush；这正是 §4.1 中「traceback 与 stdout 错序」的同一个成因。仅设 `PYTHONUNBUFFERED=1` 不足以绕过（它只影响 Python 自身缓冲，不改变 conda 的捕获层）。
- **修复**：脚本改用 `CONDA_RUN=(conda run --no-capture-output -n smolvla_eval)`（等价别名 `--live-stream`），输出直通，配合 `tee` 得到实时日志。因当时 G6-B 仅运行 3 分钟，选择**主动中止并重启**（attempt 3），代价极小。
- **验证**：重启后 1 分钟内 `run.log` 写入 18.8 KB，`STEP 1 → STEP 3` 各阶段与 rollout 进度条实时可见。

### 4.3 重启前置条件核验（2026-09-15 11:01-11:02，全部 PASS）

| 检查项 | 结果 |
|---|---|
| `python -m compileall -q src/vla_tcs2 main.py` | COMPILEALL OK |
| 入口 import（`calibration` / `quant_linear` / `quant.stat_manager`） | IMPORT OK |
| 四组 scale 目录（`--skip-calibration` 前提） | 各 864 文件（`*.p`） |
| Gate 2 路由复验（B/C/D） | 三组 `ROUTING GATE: PASS`（160/64、176/48、112/112，MatMul 64） |
| G6-A 完成后 `run_goal.sh` 的 `[SKIP]` 逻辑 | ✅ 生效（`[SKIP] g6a_all_fp8_control already complete`） |

### 4.4 同类问题在 Phase H 的 H3 脚本中仍然存在（2026-09-15 巡检发现，非本实验改动）

12:37 巡检发现同一成因的残留问题出现在 **Phase H** 的 `run_h3_100ep.sh`（**本实验范围外，未改动**）：

| 问题 | 现象 | 证据 |
|---|---|---|
| `conda run` 块缓冲 | `h3_run.log` **47 分钟未更新**，但进程健康 | G6 日志 mtime = 12:37:46（实时）；H3 日志 mtime = 11:50:16（停滞），而 H3 的 task03 已跑 47 min |
| `set -euo pipefail` | 与 9-14 G6 事故同款，s0 中途失败会连坐丢掉 s1 全部 10 个 task | 脚本第 5 行 |

- **如何判断 H3 仍在推进（无实时日志时的替代手段）**：① `ps -o etime,time,%cpu` 显示 CPU 时间持续增长；② `nvidia-smi` 连续采样为 72–99%；③ `h3_100ep/<group>/taskXX/` 目录与 `result.json` 逐个出现。
- **缓冲机制的实证（14:37 巡检）**：H3 日志在 12:37 时已停滞 47 min，随后于 **14:36:47 突然刷新并出现 task06 的完整结果块** —— 证明 `conda run` 是**按子进程粒度 flush**，即「每个 task 结束时才落一次盘」。因此 H3 的日志滞后上界 ≈ 单 task 耗时（当前 21–53 min）。
- **附带的耗时规律（H3 s0 实测，可作后续排障参考）**：单 task 耗时与 SR 强负相关——SR=100% 时 21.3/22.9/24.6/25.1 min（128–151 s/ep），而 SR=50–80% 时 48.8/37.8/53.4 min（**227–320 s/ep，约 2.2×**）。原因是失败的 episode 会跑满 300 步上限。故「某 task 异常慢」可作为**该 task 失败率偏高**的预警信号。
- **⚠️ 严禁在 H3 运行期间编辑 `run_h3_100ep.sh`**：bash 是按**字节偏移**增量读取脚本的，编辑正在运行的脚本会导致后续循环读入错位内容（未定义行为）。同一条约束适用于本实验正在运行的 `run_goal.sh`——**本次 `run_goal.sh` 的修改发生在 11:08，早于 11:09 的启动，故安全**；此后至 sweep 结束前不得再改。
- **建议**：待 H3 跑完后，把 §4.1/§4.2 的两项修复（`set -uo pipefail` + `failed[]`、`conda run --no-capture-output` + `tee`）同步到 `run_h3_100ep.sh` 与 `run_h2_30ep.sh`，统一长时 sweep 脚本模板。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-15 | 创建文档；回填 Gate 0-5、Gate 6-A（90.0%）与 9-14 中止事故 | |
| 2026-09-15 | 回填 4.2（`--no-capture-output` 修复）、4.3（重启前置核验）、attempt 2/3 记录与后台任务 | |
| 2026-09-15 12:37 | 巡检回填：G6-B 3/10 tasks（80.0%，逐 task `[10,9,5]`）、单 task 耗时与 ETA；新增 §4.4（H3 脚本同类问题 + 运行期禁改脚本约束） | |
| 2026-09-15 14:37 | 巡检回填：G6-B 8/10（68.8%）、H3 s0 7/10（88.6%）与各自 ETA；新增 G6-B↔G5-B 与 G6-A↔H3 s0 两处交叉印证；§4.4 补缓冲机制实证与 SR-耗时相关性 | |
| 2026-09-16 | Gate 6 四组全部完成（B 16:13 / C 03:25 / D 13:57）；A/B/C/D = 90/69/41/21 | |
| 2026-09-19 | **状态改 done**；回填四组 SR、L 值判据（三项全过）与结题说明 | |
