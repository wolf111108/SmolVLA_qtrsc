# 实验日志（Logs）

> 本文档记录实验**运行进度与状态**（区别于 `results.md` 的结果记录）。
> 每次启动/完成一次运行、发现或修复异常时追加一条；章节为固定结构，不可删除；无内容写「无」。
> 运行日志按时间**正序**追加，最新状态反映在「当前状态」与头部字段。

- **实验名称**：2026-09-15_sparsity-ratio-quickscan
- **状态**：draft（draft / running / done / aborted）
- **最后更新**：2026-09-15

---

## 1. 当前状态

**脚手架、4 个 config、runner 与汇总脚本已就绪并通过静态校验；尚未开跑（等待 GPU）。**

静态校验已完成：`bash -n` 通过、`py_compile` 通过、4 个 YAML 可解析、汇总脚本用 Phase H 真实产物验证过聚合正确性（见 §4.3）。当前**唯一阻塞项**是本机仅 1 张 H100 NVL 且被 Phase H 的 H3 占用（见 §3）。

| 阶段 | 状态 | 完成时间 | 备注 |
|---|---|---|---|
| 骨架 + 文档 | ✅ done | 2026-09-15 | configs / scripts / docs |
| scale 准备 | ✅ done | 2026-09-15 | Q2/Q3 本地只读副本 + sha256 存证（864/864 bit-identical） |
| 静态校验 | ✅ done | 2026-09-15 | bash -n / py_compile / YAML / 汇总脚本空跑与真值验证 |
| Q0 INT8 | ⏳ pending | — | 待 GPU |
| Q1 INT16 | ⏳ pending | — | 待 GPU |
| Q2 FP8(A/O)-W4 | ⏳ pending | — | 待 GPU；必须 `--skip-calibration` |
| Q3 FP8 PoT | ⏳ pending | — | 待 GPU；必须 `--skip-calibration` |
| 汇总 | ⏳ pending | — | 依赖 4 组完成 |

## 2. 运行日志

| 日期 | 阶段 | 命令 / 配置 | 状态 | 输出目录 | 备注 |
|---|---|---|---|---|---|
| 2026-09-15 | 建骨架 | `mkdir` + `_template/docs/*.md` 替换 | ✅ | — | 目录名按 `README_EXPERIMENT.md` §7 取 `2026-09-15_sparsity-ratio-quickscan` |
| 2026-09-15 | 写 config | `configs/q0..q3` | ✅ | — | Q2/Q3 量化块逐字复用 G1-B/G1-A |
| 2026-09-15 | scale 准备 | `cp` G1-A/G1-B → 本实验目录 + sha256 校验 | ✅ | `scales/2026-09-15_sparsity-ratio-quickscan/` | 各 864 文件，bit-identical |
| 2026-09-15 | 写脚本 | `run_quickscan.sh` / `summarize_quick_sparsity.py` | ✅ | — | 语法与空跑校验通过 |
| 2026-09-15 | 校验 | 用 Phase H H2-S0 task00 产物验证聚合 | ✅ | `/tmp/qs_validate`（临时） | Expert weight_bit=42.58%，与 Phase H 记录一致 |

## 3. 后台任务

| PID | 阶段 | 启动时间 | 状态 | 备注 |
|---|---|---|---|---|
| — | — | — | ⏳ 未启动 | 待 GPU 空闲后执行 `GPU=0 bash .../run_quickscan.sh` |

### 3.1 GPU 占用情况（2026-09-15 18:0x）

| GPU | 型号 | 占用者 | 说明 |
|---|---|---|---|
| 0 | NVIDIA H100 NVL | Phase H `run_h3_100ep.sh`（s0 → s1，各 10 task × 10ep） | **本实验唯一可用卡** |

> `nvidia-smi -L` 仅列出 GPU 0。README §9 要求「GPU 请选择未被 H3 占用的 GPU」，但本机无第二张卡可选。两种选择：
> 1. **等 H3 跑完再跑本实验**（最干净；H3 预计 2026-09-15 晚间至 09-16 凌晨结束）；
> 2. 与 H3 **共享 GPU 0**（显存充裕：8949 MiB / 95830 MiB，但算力互稀释，单 episode 与校准都会变慢）。
>
> 本实验是 quickscan（预算 30–50 min），共享代价可接受，但需意识到 H3 也会被拖慢。**建议优先方案 1。**

## 4. 异常与处理

### 4.1 脚手架目录名不符合 README 要求

- **现象**：`new_experiment.sh 2026-09-15 sparsity-ratio quickscan` 生成 `2026-09-15_sparsity-ratio_quickscan`；换成 `sparsity ratio-quickscan` 又得到 `2026-09-15_sparsity_ratio-quickscan`。
- **根因**：脚手架固定用 `${DATE}_${PHASE}_${TOPIC}` 拼接，段间分隔符恒为 `_`，无法产出 `README_EXPERIMENT.md` §7 要求的 `2026-09-15_sparsity-ratio-quickscan`（后缀全用连字符）。
- **处理**：按 README 的字面路径**手工建骨架**（`configs/ scripts/ docs/figures/ tasks/`），并用 `sed` 从 `experiments/_template/docs/` 生成三份文档，保证章节结构仍与规范一致。
- **待办**：规范与 README 的命名冲突需要一次性裁决（是放宽脚手架以支持连字符 topic，还是统一改用下划线）。本实验按 README 执行。

### 4.2 Q2/Q3 复用 scale 的覆盖风险

- **风险**：README §4 让 Q2/Q3 的 `scale_dir` 直指 `scales/2026-09-10_phaseG_w4-root-cause/component-localization/{g1a_fp8_all,g1b_w4_all}`。这两个目录**正被 Phase H 的 H1/H2/H3 复用**（Phase H setup §2 明文禁止覆盖，且维护 hash manifest）。若有人不加 `--skip-calibration` 跑 Q2/Q3，框架会原地重写这些 scale。
- **处理**（三重防护）：
  1. 把 scale **复制**到 `scales/2026-09-15_sparsity-ratio-quickscan/{q2_fp8w4_po2,q3_fp8_po2}`，config 的 `scale_dir` 指向副本；
  2. 副本与原始逐文件 sha256 校验 **864/864 bit-identical**，统计等价性成立；
  3. `run_quickscan.sh` 对 Q2/Q3 **硬编码 `--skip-calibration`**，破坏路径不可达。
- **校验记录**：`docs/manifest/{q2_fp8w4_po2,q3_fp8_po2}_scale_hashes.txt`。
- **附注**：原始目录 mtime 保持 2026-09-11（未被触碰）。

### 4.3 汇总脚本的真值验证

为避免「先跑完才发现聚合写错」，用 **Phase H H2-S0 task00** 的既有产物（其量化配置与 Q3 相同：FP8 PoT + G1-A scale）做了端到端验证：

```text
[FP8 PoT] q3_fp8_po2
  [OK] phase 覆盖 prefill+denoise
  [OK] denoise flow_step 覆盖 0..9
  [OK] component 覆盖 vlm+expert
  [OK] tensor_role 覆盖 Linear activation/output + MatMul A/B/O
  VLM prefill      elem=  1.92%  bit= 41.69%  | w_elem=  0.00%  w_bit= 40.86%
  Expert denoise   elem=  3.03%  bit= 42.15%  | w_elem=  0.00%  w_bit= 42.58%
```

**`Expert denoise weight_bit = 42.58%` 与 Phase H `results.md` 记录的「S0 Expert FP8 42.58%」完全一致**，说明本脚本的 sum-then-divide 聚合与 Phase H 独立实现（`build_results_tables.py`）口径相同。Gate 校验四项亦全部通过。

### 4.4 已知的生成期瑕疵（已修）

- `q1_int16.yaml` 的 `batch_size` 曾被批量替换误改为 16（正则 `": 8\n" -> ": 16\n"` 连带命中），已恢复为 README §5 要求的 8。
- `q2/q3` 的 `calibration` 段原为 Phase G 的 `episodes: 8 / frame_stride: 4`，已统一改为 quick 值 `1 / 8 / 16`，使误触发的重校准代价最小。

## 5. 修订记录

| 日期 | 修改内容 | 修改人 |
|---|---|---|
| 2026-09-15 | 创建文档；记录骨架/config/scale/脚本就绪与静态校验结果 | |
| 2026-09-15 | 记录 §4.1 命名冲突、§4.2 scale 覆盖风险与三重防护、§4.3 用 Phase H 真值验证聚合、§4.4 生成期瑕疵修正 | |
