# 新候选 checkpoint vs 基线 (LIBERO 四 suite)

- HuggingFaceVLA/smolvla_libero_ckpts@100k: 官方 16/0.75, pi/libero 数据, DEPRECIATED 但 recipe 贴论文
- k1000dai/smolvla_libero_finetune: 社区 16/0.75, 100k steps, batch64, from smolvla_base
- tiantianx/smolvla_libero (C): paper-like 社区复现, 100k, expert-only
- lerobot/smolvla_libero (A): legacy 社区, ~25k, full finetune
- 论文参照: SmolVLA Table 2 (arXiv:2506.01844), 平均 87.3%
- 统一协议: seed=1000, n_action_steps=1, num_steps=10, 10 tasks × 10 episodes, EGL(device2)

## Suite 级对比

| Suite | 论文 | HuggingFaceVLA/smolvla_libero_ckpts | k1000dai/smolvla_libero_finetune | tiantianx/smolvla_libero | lerobot/smolvla_libero |
|---|---|---|---|---|---|
| Spatial | 90% | - | - | 76% (76/100) | 81% (81/100) |
| Object | 96% | - | - | 59% (59/100) | 60% (60/100) |
| Goal | 92% | - | - | 63% (63/100) | 77% (77/100) |
| Long | 71% | - | - | 40% (40/100) | 59% (59/100) |
| **平均** | **87.3%** | (0/4) | (0/4) | **59.5%** | **69.2%** |

## Per-task 成功率

### Spatial

| task | tiantianx/smolvla_libero | lerobot/smolvla_libero |
|---|---|---|
| task 0 | 90% | 100% |
| task 1 | 100% | 90% |
| task 2 | 100% | 100% |
| task 3 | 90% | 100% |
| task 4 | 90% | 60% |
| task 5 | 30% | 0% |
| task 6 | 100% | 100% |
| task 7 | 80% | 90% |
| task 8 | 40% | 80% |
| task 9 | 40% | 90% |

### Object

| task | tiantianx/smolvla_libero | lerobot/smolvla_libero |
|---|---|---|
| task 0 | 50% | 60% |
| task 1 | 40% | 60% |
| task 2 | 50% | 70% |
| task 3 | 60% | 60% |
| task 4 | 60% | 40% |
| task 5 | 80% | 40% |
| task 6 | 60% | 60% |
| task 7 | 90% | 80% |
| task 8 | 30% | 70% |
| task 9 | 70% | 60% |

### Goal

| task | tiantianx/smolvla_libero | lerobot/smolvla_libero |
|---|---|---|
| task 0 | 20% | 70% |
| task 1 | 80% | 90% |
| task 2 | 90% | 80% |
| task 3 | 50% | 50% |
| task 4 | 80% | 100% |
| task 5 | 80% | 90% |
| task 6 | 60% | 40% |
| task 7 | 60% | 90% |
| task 8 | 80% | 100% |
| task 9 | 30% | 60% |

### Long

| task | tiantianx/smolvla_libero | lerobot/smolvla_libero |
|---|---|---|
| task 0 | 10% | 10% |
| task 1 | 40% | 80% |
| task 2 | 50% | 90% |
| task 3 | 80% | 100% |
| task 4 | 0% | 60% |
| task 5 | 100% | 90% |
| task 6 | 60% | 30% |
| task 7 | 10% | 40% |
| task 8 | 0% | 20% |
| task 9 | 50% | 70% |

## 数据来源

- HuggingFaceVLA/smolvla_libero_ckpts@100k Spatial: 未找到结果文件
- HuggingFaceVLA/smolvla_libero_ckpts@100k Object: 未找到结果文件
- HuggingFaceVLA/smolvla_libero_ckpts@100k Goal: 未找到结果文件
- HuggingFaceVLA/smolvla_libero_ckpts@100k Long: 未找到结果文件
- k1000dai/smolvla_libero_finetune Spatial: 未找到结果文件
- k1000dai/smolvla_libero_finetune Object: 未找到结果文件
- k1000dai/smolvla_libero_finetune Goal: 未找到结果文件
- k1000dai/smolvla_libero_finetune Long: 未找到结果文件
- tiantianx/smolvla_libero (C) Spatial: `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_spatial/eval_info.json`
- tiantianx/smolvla_libero (C) Object: `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_object/eval_info.json`
- tiantianx/smolvla_libero (C) Goal: `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_goal/eval_info.json`
- tiantianx/smolvla_libero (C) Long: `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_10/eval_info.json`
- lerobot/smolvla_libero (A) Spatial: `/home/zyzhao/VLA_tcs2/outputs/baseline_smolvla450m_libero_spatial/eval_info.json`
- lerobot/smolvla_libero (A) Object: `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_object/eval_info.json`
- lerobot/smolvla_libero (A) Goal: `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_goal/eval_info.json`
- lerobot/smolvla_libero (A) Long: `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_10/eval_info.json`
