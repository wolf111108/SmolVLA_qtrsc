# PHASE 2 — Observation Contract 审计结果

> 设备：h100（smolvla_eval，lerobot_current @ 6adf515）
> 日期：2026-08-25
> 方法：静态源码审计（grep 定位 + 关键源码精读）

## 源码定位

| 关注点 | 位置 |
|---|---|
| LiberoProcessorStep（state 组装 + 180° flip） | `src/lerobot/processor/env_processor.py:28-111` |
| _quat2axisangle | `src/lerobot/processor/env_processor.py:114` |
| LIBERO env 设置（control/init/settle） | `src/lerobot/envs/libero.py:123-194, 344-361` |
| LIBERO env config 默认值 | `src/lerobot/envs/configs.py:325-355` |
| TASK_SUITE_MAX_STEPS | `src/lerobot/envs/libero.py:99-106` |
| get_libero_dummy_action | `src/lerobot/envs/libero.py:90-92` |

## 2.2 Observation Contract 矩阵（Runtime 侧已闭环）

| 项目 | Dataset（HuggingFaceVLA/libero） | Runtime（当前 processor） | 一致 |
|---|---|---|---|
| agent view（主相机） | image（key: observation.images.image） | agentview_image → `observation.images.image` | ✅ |
| wrist 相机 | image2（observation.images.image2） | robot0_eye_in_hand_image → `observation.images.image2` | ✅ |
| camera3 | 无（数据集仅 2 相机） | 无（仅 2 相机，无 camera3） | ✅ |
| 图像方向 | — | `torch.flip(img, dims=[2,3])`（180° flip，对齐 HuggingFaceVLA/libero convention） | ✅ |
| state 位置 | 3D | eef_pos(3) | ✅ |
| state 朝向 | axis-angle | `_quat2axisangle` 四元数→轴角(3) | ✅ |
| gripper qpos | 2D | gripper_qpos(2) | ✅ |
| **总 state** | **8D** | eef_pos(3)+axisangle(3)+gripper(2)=**8D** | ✅ |

关键源码（env_processor.py:75）：

```python
state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)  # 8D
```

## env 关键设置（与 README handoff §7 一致）

| 参数 | 值 | 来源 |
|---|---|---|
| fps / control_freq | 20 Hz | configs.py:325 |
| hard_reset | True | configs.py:331 |
| init_states | True | configs.py:330 |
| control_mode | relative（use_delta=True） | configs.py:355 + libero.py:357 |
| num_steps_wait（settle） | 10 | libero.py:127 |
| episode_length | None → 用 TASK_SUITE_MAX_STEPS | libero.py:191-192 |

TASK_SUITE_MAX_STEPS（libero.py:99-106）：

```python
{
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
```

## 判定

**PASS: OBSERVATION_CONTRACT**

- training semantic = runtime semantic（8D state + image/image2 + 180° flip + 2 相机）
- 此前 tiantianx normalizer 与数据集 stats 逐位一致（outputs/compare_libero_stats.txt），
  与本源码审计互相印证，形成完整闭环。
- 唯一遗留：state 组装里 eef_quat 的轴角转换数学实现（_quat2axisangle）未逐行验算，
  但 shape 与语义已确认，且五模型四 suite 评测均正常，判为低风险。

## 备注

- `get_libero_dummy_action()` 返回 `[0,0,0,0,0,0,-1]`，gripper=-1 用于 settle 阶段的 no-op，
  这为 PHASE 3 的 gripper 语义提供了第一条线索（-1 是 settle/关闭方向的基准值）。
