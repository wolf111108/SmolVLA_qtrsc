# HuggingFaceVLA/smolvla_libero vs 基线 (LIBERO 四 suite)

- HuggingFaceVLA/smolvla_libero (official ref): 32 层 / 0.5 宽 / 500M-Instruct / 8D state + image/image2
- lerobot/smolvla_libero (A): 16 层 / 0.75 宽 / full finetune ~25k
- k1000dai/smolvla_libero_finetune (D): 16 层 / 0.75 宽 / 100k / batch64
- tiantianx/smolvla_libero (C): 16 层 / 0.75 宽 / 100k / expert-only
- 论文参照: SmolVLA Table 2 (arXiv:2506.01844), 平均 87.3%
- 统一协议: seed=1000, n_action_steps=1, num_steps=10, 10 tasks × 10 episodes, EGL(device2)

## Suite 级对比

| Suite | 论文 | HuggingFaceVLA/smolvla_libero | lerobot/smolvla_libero | k1000dai/smolvla_libero_finetune | tiantianx/smolvla_libero |
|---|---|---|---|---|---|
| Spatial | 90% | 65% (65/100) | 81% (81/100) | 64% (64/100) | 76% (76/100) |
| Object | 96% | 71% (71/100) | 60% (60/100) | 82% (82/100) | 59% (59/100) |
| Goal | 92% | 72% (72/100) | 77% (77/100) | 70% (70/100) | 63% (63/100) |
| Long | 71% | 37% (37/100) | 59% (59/100) | 46% (46/100) | 40% (40/100) |
| **平均** | **87.3%** | **61.2%** | **69.2%** | **65.5%** | **59.5%** |

## Per-task 成功率

### Spatial

| task | HuggingFaceVLA/smolvla_libero | lerobot/smolvla_libero | k1000dai/smolvla_libero_finetune | tiantianx/smolvla_libero |
|---|---|---|---|---|
| task 0 | 60% | 100% | 80% | 90% |
| task 1 | 60% | 90% | 80% | 100% |
| task 2 | 90% | 100% | 100% | 100% |
| task 3 | 50% | 100% | 90% | 90% |
| task 4 | 70% | 60% | 50% | 90% |
| task 5 | 60% | 0% | 40% | 30% |
| task 6 | 70% | 100% | 60% | 100% |
| task 7 | 70% | 90% | 80% | 80% |
| task 8 | 70% | 80% | 20% | 40% |
| task 9 | 50% | 90% | 40% | 40% |

### Object

| task | HuggingFaceVLA/smolvla_libero | lerobot/smolvla_libero | k1000dai/smolvla_libero_finetune | tiantianx/smolvla_libero |
|---|---|---|---|---|
| task 0 | 60% | 60% | 90% | 50% |
| task 1 | 80% | 60% | 90% | 40% |
| task 2 | 90% | 70% | 50% | 50% |
| task 3 | 90% | 60% | 60% | 60% |
| task 4 | 70% | 40% | 90% | 60% |
| task 5 | 100% | 40% | 100% | 80% |
| task 6 | 70% | 60% | 80% | 60% |
| task 7 | 20% | 80% | 90% | 90% |
| task 8 | 70% | 70% | 80% | 30% |
| task 9 | 60% | 60% | 90% | 70% |

### Goal

| task | HuggingFaceVLA/smolvla_libero | lerobot/smolvla_libero | k1000dai/smolvla_libero_finetune | tiantianx/smolvla_libero |
|---|---|---|---|---|
| task 0 | 100% | 70% | 20% | 20% |
| task 1 | 100% | 90% | 90% | 80% |
| task 2 | 80% | 80% | 80% | 90% |
| task 3 | 0% | 50% | 20% | 50% |
| task 4 | 90% | 100% | 100% | 80% |
| task 5 | 70% | 90% | 70% | 80% |
| task 6 | 60% | 40% | 60% | 60% |
| task 7 | 90% | 90% | 90% | 60% |
| task 8 | 80% | 100% | 90% | 80% |
| task 9 | 50% | 60% | 80% | 30% |

### Long

| task | HuggingFaceVLA/smolvla_libero | lerobot/smolvla_libero | k1000dai/smolvla_libero_finetune | tiantianx/smolvla_libero |
|---|---|---|---|---|
| task 0 | 0% | 10% | 0% | 10% |
| task 1 | 40% | 80% | 60% | 40% |
| task 2 | 40% | 90% | 50% | 50% |
| task 3 | 70% | 100% | 80% | 80% |
| task 4 | 0% | 60% | 50% | 0% |
| task 5 | 80% | 90% | 100% | 100% |
| task 6 | 60% | 30% | 10% | 60% |
| task 7 | 20% | 40% | 60% | 10% |
| task 8 | 0% | 20% | 20% | 0% |
| task 9 | 60% | 70% | 30% | 50% |

## 数据来源

- HuggingFaceVLA/smolvla_libero (official ref) Spatial: `/home/zyzhao/VLA_tcs2/outputs/hfvla_libero_libero_spatial/eval_info.json`
- HuggingFaceVLA/smolvla_libero (official ref) Object: `/home/zyzhao/VLA_tcs2/outputs/hfvla_libero_libero_object/eval_info.json`
- HuggingFaceVLA/smolvla_libero (official ref) Goal: `/home/zyzhao/VLA_tcs2/outputs/hfvla_libero_libero_goal/eval_info.json`
- HuggingFaceVLA/smolvla_libero (official ref) Long: `/home/zyzhao/VLA_tcs2/outputs/hfvla_libero_libero_10/eval_info.json`
- lerobot/smolvla_libero (A) Spatial: `/home/zyzhao/VLA_tcs2/outputs/baseline_smolvla450m_libero_spatial/eval_info.json`
- lerobot/smolvla_libero (A) Object: `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_object/eval_info.json`
- lerobot/smolvla_libero (A) Goal: `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_goal/eval_info.json`
- lerobot/smolvla_libero (A) Long: `/home/zyzhao/VLA_tcs2/outputs/lerobot_smolvla_libero_libero_10/eval_info.json`
- k1000dai/smolvla_libero_finetune (D) Spatial: `/home/zyzhao/VLA_tcs2/outputs/k1000dai_ft100k_libero_spatial/eval_info.json`
- k1000dai/smolvla_libero_finetune (D) Object: `/home/zyzhao/VLA_tcs2/outputs/k1000dai_ft100k_libero_object/eval_info.json`
- k1000dai/smolvla_libero_finetune (D) Goal: `/home/zyzhao/VLA_tcs2/outputs/k1000dai_ft100k_libero_goal/eval_info.json`
- k1000dai/smolvla_libero_finetune (D) Long: `/home/zyzhao/VLA_tcs2/outputs/k1000dai_ft100k_libero_10/eval_info.json`
- tiantianx/smolvla_libero (C) Spatial: `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_spatial/eval_info.json`
- tiantianx/smolvla_libero (C) Object: `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_object/eval_info.json`
- tiantianx/smolvla_libero (C) Goal: `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_goal/eval_info.json`
- tiantianx/smolvla_libero (C) Long: `/home/zyzhao/VLA_tcs2/outputs/tiantianx_smolvla_libero_libero_10/eval_info.json`