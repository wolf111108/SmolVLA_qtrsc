# outputs/ 目录索引

> 2026-08-26 整理：按类别归入子目录（条目名保持原名）。历史 `logs/*.md` 中记录的旧路径不再回改（归档性质），以本表为准。

## 顶层结构

```
outputs/
├── README.md                  ← 本索引
├── table2_repro_audit/        ← Table-2 严格复现审计（编号体系 00-16，脚本引用，保持原位不动）
├── libero/                    ← LIBERO 四 suite 评测结果
├── metaworld/                 ← Meta-World 评测结果
├── reports/                   ← 汇总/对比/审计报告（md/txt）
├── figures/                   ← 图表（png）
└── misc/                      ← 早期 eval、identity 检查等
```

## 旧路径 → 新路径映射

### LIBERO（`libero/`）

| 旧（outputs/ 根下） | 新 |
|---|---|
| `lerobot_smolvla_libero_libero_{spatial,object,goal,10}/` | `libero/lerobot_smolvla_libero_libero_{spatial,object,goal,10}/`（A 类） |
| `baseline_smolvla450m_libero_spatial/` | `libero/baseline_smolvla450m_libero_spatial/`（A 类早期） |
| `libero_spatial_rerun_v2/` | `libero/libero_spatial_rerun_v2/` |
| `hfvla_libero_libero_{spatial,object,goal,10}/` | `libero/hfvla_libero_libero_{spatial,object,goal,10}/`（官方 ref 32L/0.5） |
| `k1000dai_ft100k_libero_{spatial,object,goal,10}/` | `libero/k1000dai_ft100k_libero_{spatial,object,goal,10}/`（D 类） |
| `tiantianx_smolvla_libero_libero_{spatial,object,goal,10}/` | `libero/tiantianx_smolvla_libero_libero_{spatial,object,goal,10}/`（C 类） |
| `lerobot_smolvla_libero_multisuite_logs/` | `libero/logs/lerobot_smolvla_libero_multisuite_logs/` |
| `multisuite_nohup.log` | `libero/logs/multisuite_nohup.log` |

### Meta-World（`metaworld/`）

| 旧（outputs/ 根下） | 新 |
|---|---|
| `metaworld_mt50_lerobot_smolvla/` | `metaworld/metaworld_mt50_lerobot_smolvla/`（MT50 全量） |
| `metaworld_mt50_run.log` | `metaworld/metaworld_mt50_run.log` |
| `metaworld_push_v3_10ep/` | `metaworld/metaworld_push_v3_10ep/` |
| `metaworld_smoke_push_mj332/` | `metaworld/metaworld_smoke_push_mj332/` |
| `metaworld_smoke_push_origcfg/` | `metaworld/metaworld_smoke_push_origcfg/` |

### 报告 / 图表 / misc

| 旧 | 新 |
|---|---|
| `checkpoint_baselines_summary.md` | `reports/checkpoint_baselines_summary.md` |
| `hfvla_vs_baselines.md` | `reports/hfvla_vs_baselines.md` |
| `new_candidates_vs_baselines.md` | `reports/new_candidates_vs_baselines.md` |
| `lerobot_smolvla_libero_vs_paper.md` | `reports/lerobot_smolvla_libero_vs_paper.md` |
| `tiantianx_vs_lerobot_libero.md` | `reports/tiantianx_vs_lerobot_libero.md` |
| `audit_hfvla_ckpts100k.txt` | `reports/audit_hfvla_ckpts100k.txt` |
| `audit_tiantianx_normalizer.txt` | `reports/audit_tiantianx_normalizer.txt` |
| `compare_libero_stats.txt` | `reports/compare_libero_stats.txt` |
| `benchmark_libero.png` | `figures/benchmark_libero.png` |
| `benchmark_metaworld.png` | `figures/benchmark_metaworld.png` |
| `eval/` | `misc/eval/`（2026-08-19 早期） |
| `vla_tcs2_identity_task0/` | `misc/vla_tcs2_identity_task0/` |

## 已同步更新的引用脚本

- `scripts/compare_hfvla_vs_baselines.py`
- `scripts/compare_new_candidates_vs_baselines.py`
- `scripts/compare_tiantianx_vs_lerobot.py`
- `scripts/summarize_lerobot_vs_paper.py`
- `scripts/plot_benchmarks.py`
- `scripts/run_hfvla_libero_suites.sh`

> 注：`scripts/audit_*.py`、`scripts/compare_libero_stats.py` 内含 h100（zyzhao@hankh100）侧绝对路径，未改动，运行在 h100 侧自行核对。
