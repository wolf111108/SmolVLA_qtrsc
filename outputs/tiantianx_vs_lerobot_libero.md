# tiantianx/smolvla_libero vs lerobot/smolvla_libero (LIBERO 四 suite)

- tiantianx: C: paper-like (16/0.75, 100k, expert-only)
- lerobot:   A: legacy (16/0.75, ~25k, full finetune)
- 论文参照:  SmolVLA Table 2 (arXiv:2506.01844)
- 统一协议:  seed=1000, n_action_steps=1, num_steps=10, chunk_size=50, 10 tasks × 10 episodes

## Suite 级对比

| Suite | 论文 | tiantianx (C) | lerobot (A) | C vs A | C vs 论文 |
|---|---:|---:|---:|---:|---:|
| Spatial | 90% | 76% (76/100) | 81% (81/100) | -5pp | -14pp |
| Object | 96% | 59% (59/100) | 60% (60/100) | -1pp | -37pp |
| Goal | 92% | 63% (63/100) | 77% (77/100) | -14pp | -29pp |
| Long (LIBERO-10) | 71% | 40% (40/100) | 59% (59/100) | -19pp | -31pp |
| **平均** | **87.3%** | **59.5%** | **69.2%** | **-9.8pp** | **-27.8pp** |

## Per-task 成功率

### Spatial

| task | tiantianx (C) | lerobot (A) | diff |
|---|---:|---:|---:|
| task 0 | 90% | 100% | -10pp |
| task 1 | 100% | 90% | +10pp |
| task 2 | 100% | 100% | +0pp |
| task 3 | 90% | 100% | -10pp |
| task 4 | 90% | 60% | +30pp |
| task 5 | 30% | 0% | +30pp |
| task 6 | 100% | 100% | +0pp |
| task 7 | 80% | 90% | -10pp |
| task 8 | 40% | 80% | -40pp |
| task 9 | 40% | 90% | -50pp |

### Object

| task | tiantianx (C) | lerobot (A) | diff |
|---|---:|---:|---:|
| task 0 | 50% | 60% | -10pp |
| task 1 | 40% | 60% | -20pp |
| task 2 | 50% | 70% | -20pp |
| task 3 | 60% | 60% | +0pp |
| task 4 | 60% | 40% | +20pp |
| task 5 | 80% | 40% | +40pp |
| task 6 | 60% | 60% | +0pp |
| task 7 | 90% | 80% | +10pp |
| task 8 | 30% | 70% | -40pp |
| task 9 | 70% | 60% | +10pp |

### Goal

| task | tiantianx (C) | lerobot (A) | diff |
|---|---:|---:|---:|
| task 0 | 20% | 70% | -50pp |
| task 1 | 80% | 90% | -10pp |
| task 2 | 90% | 80% | +10pp |
| task 3 | 50% | 50% | +0pp |
| task 4 | 80% | 100% | -20pp |
| task 5 | 80% | 90% | -10pp |
| task 6 | 60% | 40% | +20pp |
| task 7 | 60% | 90% | -30pp |
| task 8 | 80% | 100% | -20pp |
| task 9 | 30% | 60% | -30pp |

### Long (LIBERO-10)

| task | tiantianx (C) | lerobot (A) | diff |
|---|---:|---:|---:|
| task 0 | 10% | 10% | +0pp |
| task 1 | 40% | 80% | -40pp |
| task 2 | 50% | 90% | -40pp |
| task 3 | 80% | 100% | -20pp |
| task 4 | 0% | 60% | -60pp |
| task 5 | 100% | 90% | +10pp |
| task 6 | 60% | 30% | +30pp |
| task 7 | 10% | 40% | -30pp |
| task 8 | 0% | 20% | -20pp |
| task 9 | 50% | 70% | -20pp |

## 数据来源

- tiantianx/smolvla_libero Spatial: `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_spatial/eval_info.json`
- tiantianx/smolvla_libero Object: `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_object/eval_info.json`
- tiantianx/smolvla_libero Goal: `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_goal/eval_info.json`
- tiantianx/smolvla_libero Long (LIBERO-10): `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_10/eval_info.json`
- lerobot/smolvla_libero Spatial: `/home/zyzhao/VLA_tcs2/outputs/baseline_smolvla450m_libero_spatial/eval_info.json`
- lerobot/smolvla_libero Object: `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_object/eval_info.json`
- lerobot/smolvla_libero Goal: `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_goal/eval_info.json`
- lerobot/smolvla_libero Long (LIBERO-10): `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_10/eval_info.json`
