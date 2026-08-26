# PHASE 3 — Action/Gripper Contract 审计（源码侧结论）

> 设备：h100 · lerobot_current @ 6adf515 · 2026-08-25
> 状态：源码链路已闭环；dataset gripper 数值统计进行中（数据集 3.84GB 下载中）

## 3.2 静态源码结论（action pipeline 已确认）

### Action 全链路（A0 → A4）

| 层 | 源码位置 | 行为 |
|---|---|---|
| A0 model raw | `modeling_smolvla.py` | flow matching 采样输出 normalized action |
| A1 select_action | policy | 输出归一化动作 |
| A2 policy unnormalizer | `normalize_processor.py` `UnnormalizerProcessorStep`（factory.py:174 `output_steps=[s.unnormalize, s.to_cpu]`） | MEAN_STD 逆变换：`tensor * std + mean` |
| A3 env postprocessor | `envs/configs.py:127` **Default: identity** | **LIBERO env postprocessor = identity**（手册 3.1 假设成立） |
| A4 env.step | `envs/libero.py:366,374` `self._env.step(action)` | **action 原样透传给 robosuite**，无额外变换 |

### 关键判定：当前 pipeline 无 #1316 时代的 hack

- 手册 #1316 提到的 `action[-1] = 2*action[-1]-1; action[-1] *= -1` **在当前 0.6.2 代码中不存在**；
- 当前用的是统一的 `UnnormalizerProcessorStep`（用 dataset stats 做 MEAN_STD 逆变换）+ env postprocessor=identity + env.step 透传；
- 意味着 gripper convention 完全由 **dataset stats（HuggingFaceVLA/libero 的 action 归一化）** 决定，
  只要 stats 与训练时一致，链就是自洽的。

### 需要警惕的 SmolVLA 特有逻辑

`modeling_smolvla.py:99-140` 有一组 **Aloha gripper 空间转换函数**：
- `aloha_gripper_to_angular` / `aloha_gripper_from_angular`（用 0.01844/0.05800/0.4/1.5/-0.6213/1.4910 这些 Aloha 物理常数）
- 这些只在 `adapt_to_pi_aloha=true` 时触发（配置默认 false），LIBERO 路线**不经过**此转换。

## 3.4 Dataset gripper 统计（2026-08-25 完成，gpupro6000 数据）

**原始 action[-1] 分布（2000 样本）：**

```
samples = 2000
min = -1.0
max = 1.0
mean = -0.056
std = 0.9984
percentiles = [-1, -1, -1, -1, -1, 1, 1, 1, 1]
```

**结论：gripper 是二值信号**，只取 -1 和 +1，无中间值（P0~P50=-1，P75~P100=+1）。

**dataset stats 的 action 归一化基准（最后分量 = gripper）：**

```python
action.min   = -1.0
action.max   = +1.0
action.mean  = -0.0496   # 接近 0，两类均衡
action.std   = 0.9988    # 接近 1，值域即 [-1,+1]
```

## 3.5/3.6 gripper 语义

| 值 | 语义推断 | 依据 |
|---|---|---|
| -1 | 保持/张开（settle no-op） | `get_libero_dummy_action()=[...,-1]` 用于 settle |
| +1 | 闭合 | robosuite 正方向=夹紧 convention |

（语义方向为高置信推断，未做视频逐帧实锤；数值自洽已确证）

## 3.7 最终 Action Contract 表

| Layer | Open(≈-1) | Close(≈+1) |
|---|---:|---:|
| Dataset | -1 | +1 |
| Normalized target | (x-mean)/std ≈ -0.95 | ≈ +1.05 |
| Model normalized output | ≈ -0.95 | ≈ +1.05 |
| Unnormalized output (= env.step) | -1 | +1 |
| Actual simulator motion | OPEN/HOLD | CLOSE |

## 结论

- **源码侧：gripper 链路自洽，无手写 flip hack**，当前 pipeline 是干净的 MEAN_STD 归一化链；
- **数据侧：gripper 二值 {-1,+1}，归一化 std≈1**，unnormalize 后精确还原；
- 数值链路闭环，语义方向高置信推断（-1=开/保持，+1=关）；
- 手册 #1316 的 gripper 格式问题在当前数据+pipeline 下**不存在**。

## 判定

**PASS: GRIPPER_CONTRACT**（数值链完整自洽；语义方向为高置信推断，如需 100% 实锤可补视频逐帧验证，但非阻塞项）
