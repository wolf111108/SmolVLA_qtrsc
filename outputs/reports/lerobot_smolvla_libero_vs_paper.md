# lerobot/smolvla_libero vs 论文 Table 2 (0.45B SmolVLA)

- 论文: SmolVLA (arXiv:2506.01844), 每 suite 10 tasks × 10 episodes
- 本地: seed=1000, n_action_steps=1, num_steps=10, chunk_size=50

## Suite 级对比

| Suite | 论文 SR | 本地 SR | 本地 n | 差值 |
|---|---:|---:|---:|---:|
| Spatial | 90% | 81% (81/100) | 100 | -9pp |
| Object | 96% | 60% (60/100) | 100 | -36pp |
| Goal | 92% | 77% (77/100) | 100 | -15pp |
| Long (LIBERO-10) | 71% | 59% (59/100) | 100 | -12pp |
| **平均** | **87.3%** | **69.2%** | - | **-18.0pp** |

## Per-task 成功率 (本地)

| task | Spatial | Object | Goal | Long |
|---|---|---|---|---|
| task 0 | 100% | 60% | 70% | 10% |
| task 1 | 90% | 60% | 90% | 80% |
| task 2 | 100% | 70% | 80% | 90% |
| task 3 | 100% | 60% | 50% | 100% |
| task 4 | 60% | 40% | 100% | 60% |
| task 5 | 0% | 40% | 90% | 90% |
| task 6 | 100% | 60% | 40% | 30% |
| task 7 | 90% | 80% | 90% | 40% |
| task 8 | 80% | 70% | 100% | 20% |
| task 9 | 90% | 60% | 60% | 70% |

## 数据来源

- Spatial: `/home/zyzhao/VLA_tcs2/outputs/baseline_smolvla450m_libero_spatial/eval_info.json`
- Object: `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_object/eval_info.json`
- Goal: `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_goal/eval_info.json`
- Long (LIBERO-10): `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_10/eval_info.json`
