# SmolVLA + LeRobot 的 Meta-World 评估手册

## 1. 目的

本文档描述将 Meta-World 评估接入当前 SmolVLA 实验框架的完整流程。

目标：

-   复现 SmolVLA 的 Meta-World benchmark 结果；
-   在 LIBERO 之外建立第二个基准；
-   提供稳定的浮点基线；
-   为 PTQ（训练后量化）、激活值统计与硬件建模准备基准。

预期工作流是：

    checkpoint
        |
        v
    LeRobot policy
        |
        v
    Meta-World 环境 / 数据集
        |
        v
    闭环 rollout（滚动执行）
        |
        v
    成功率

------------------------------------------------------------------------

# 2. 背景：SmolVLA 论文中的 Meta-World

SmolVLA 同时评估两个基准：

-   LIBERO
-   Meta-World

Meta-World benchmark：

-   MT50 设定；
-   50 个操作任务；
-   2500 条演示集（episodes）；
-   每个任务 50 条演示；
-   每个任务 10 次评估试验。

评估量：

    50 个任务 × 10 集 = 500 次 rollout

指标：

    成功率（Success Rate）=
    成功集数 / 总集数

论文报告的 SmolVLA 0.45B Meta-World 结果：

  Split         成功率
  ----------- --------------
  Easy                  82.5
  Medium                41.8
  Hard                  45.0
  Very Hard             60.0
  Average               57.3

------------------------------------------------------------------------

# 3. 重要区别：原生 Meta-World 与 LeRobot 版 Meta-World

一个常见误解是：

    原生 Meta-World gym 观测
            |
            v
    直接输入 SmolVLA

这是错误的。

实际流水线是：

    原生 Meta-World 环境
            |
            v
    LeRobot 的 Meta-World 集成
            |
            v
    LeRobot 数据集格式
            |
            v
    SmolVLA processor
            |
            v
    policy 输入

数据集转换层负责对齐：

-   相机格式；
-   状态表示；
-   动作表示；
-   任务指令。

------------------------------------------------------------------------

# 4. 当前项目环境

当前环境：

    conda：
    smolvla_eval

    LeRobot：
    ~/VLA_tcs2/lerobot_current

    commit：
    6adf51511b7625090eade8d82d9f61a1846ebe56

    LeRobot：
    0.6.2

    Python：
    3.12

    PyTorch：
    2.7.1+cu118

    GPU：
    H100 NVL

应复用现有的 LIBERO 环境。

------------------------------------------------------------------------

# 5. 版本对齐

## 5.1 LeRobot 的要求

检查：

``` bash
grep -n "metaworld" pyproject.toml
```

当前结果：

``` toml
metaworld==3.0.0
```

因此安装：

    Meta-World = 3.0.0

不要随意安装最新版本。

------------------------------------------------------------------------

## 5.2 所需安装

使用 LeRobot 的 extra：

``` bash
cd ~/VLA_tcs2/lerobot_current

pip install -e ".[metaworld]"
```

不要使用：

``` bash
pip install metaworld
```

因为 LeRobot 的 extras 同时管理：

-   数据集依赖；
-   scipy 兼容性；
-   相关包。

------------------------------------------------------------------------

## 5.3 当前依赖检查

验证：

``` bash
pip list | grep -E "metaworld|mujoco|gymnasium|scipy"
```

预期：

    metaworld    3.0.0
    gymnasium    >=1.1.1,<2
    mujoco       3.x

当前已知兼容组合：

    gymnasium 1.3.0
    mujoco 3.8.1

在改动前应先测试。

------------------------------------------------------------------------

# 6. Meta-World 环境验证

在加载 SmolVLA 之前：

``` bash
python - <<'PY'
import importlib.metadata
import metaworld

print("metaworld 版本:", importlib.metadata.version("metaworld"))

env = metaworld.MT1(
    "push-v3",
    seed=42
)

print("Meta-World environment created")
PY
```

预期输出：

    metaworld 版本: 3.0.0
    Meta-World environment created

> 注意：metaworld 3.0.0 无 `__version__` 属性，用 `importlib.metadata.version()` 查版本；
> `MT1` 只接受 V3 环境名（如 `push-v3`），`push-v2` 会报 "is not a V3 environment"。

------------------------------------------------------------------------

# 7. EGL / MuJoCo 配置

当前服务器使用无头渲染（headless rendering）。

需要：

``` bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=2
```

验证：

``` bash
echo $MUJOCO_GL
echo $PYOPENGL_PLATFORM
echo $MUJOCO_EGL_DEVICE_ID
```

预期输出：

    egl
    egl
    2

------------------------------------------------------------------------

# 8. 数据格式对齐

## 8.1 原生 Meta-World 格式

原生 Meta-World 通常提供：

图像：

    pixels/top
    480×480 RGB

状态：

    agent_pos
    4 维

动作：

    Meta-World 原生动作空间

该格式不能直接作为 SmolVLA 的输入格式。

------------------------------------------------------------------------

## 8.2 SmolVLA 期望的格式

SmolVLA checkpoint 使用 LeRobot 的 feature 定义：

典型字段：

    observation.images.camera1
    observation.images.camera2
    observation.images.camera3

    observation.state

    action

模型不应直接接收原生 gym 观测。

------------------------------------------------------------------------

## 8.3 数据集转换的职责

转换流水线负责处理：

    相机归一化

    状态提取

    动作转换

    任务指令编码

因此不要手动创建：

    camera1/2/3
    state 填充

除非你在实现一个新的数据集。

------------------------------------------------------------------------

# 9. 官方 Meta-World 数据集

官方 LeRobot 数据集：

    lerobot/metaworld_mt50

包含已经转换为 LeRobot 格式的 Meta-World 演示数据。

下载：

``` bash
huggingface-cli download \
    lerobot/metaworld_mt50 \
    --repo-type dataset
```

验证缓存：

``` bash
ls ~/.cache/huggingface
```

------------------------------------------------------------------------

# 10. 官方 SmolVLA Meta-World Checkpoint

推荐使用的 checkpoint：

    lerobot/smolvla_metaworld

关系：

    lerobot/smolvla_base
              |
              |
              v
    lerobot/metaworld_mt50 训练
              |
              |
              v
    lerobot/smolvla_metaworld

该 checkpoint 专为 Meta-World 评估设计。

下载：

``` bash
huggingface-cli download \
    lerobot/smolvla_metaworld
```

------------------------------------------------------------------------

# 11. 为什么不应使用 LIBERO Checkpoint

例如：

    lerobot/smolvla_libero
    HuggingFaceVLA/smolvla_libero
    tiantianx/smolvla_libero

都是面向 LIBERO 的 checkpoint。

它们的差异在于：

-   训练分布；
-   任务指令分布；
-   观测统计量；
-   环境动力学。

LIBERO checkpoint 不是 Meta-World 基线。

------------------------------------------------------------------------

# 12. 评估流程

## 12.1 单任务冒烟测试

首先验证：

-   环境重置（reset）；
-   渲染；
-   观测；
-   动作执行。

示例：

    push-v3
    1 集

------------------------------------------------------------------------

## 12.2 完整 MT50 评估

目标：

    50 个任务
    每个任务 10 集

总计：

    500 集

保存到：

    results/
        metaworld/
            summary.csv
            task_results.json
            videos/

------------------------------------------------------------------------

# 13. 结果记录

每次实验必须记录：

    checkpoint：
    checkpoint revision：

    LeRobot commit：
    Python：
    PyTorch：
    CUDA：

    Meta-World 版本：
    MuJoCo 版本：

    GPU：

    seed：
    任务列表：
    每任务集数：

    相机配置：
    状态维度：
    动作维度：

------------------------------------------------------------------------

# 14. 调试清单

## Import 报错

检查：

``` bash
pip list | grep metaworld
```

------------------------------------------------------------------------

## 渲染崩溃

检查：

``` bash
echo $MUJOCO_GL
```

------------------------------------------------------------------------

## 观测不匹配

检查：

-   数据集 feature 名称；
-   状态维度；
-   动作维度。

绝不要在未记录转换方式的情况下，比较输入契约（input contract）不同的模型。

------------------------------------------------------------------------

# 15. 与量化框架的集成

在建立浮点 Meta-World 基线之后：

    BF16 模型

        |
        v

    模块替换

        |
        v

    校准（calibration）

        |
        v

    量化推理

        |
        v

    Meta-World 成功率

        |
        v

    激活稀疏性

        |
        v

    硬件建模

------------------------------------------------------------------------

# 16. 推荐的实验顺序

1.  安装 Meta-World 依赖。
2.  验证环境。
3.  验证官方数据集。
4.  下载 `lerobot/smolvla_metaworld`。
5.  运行单任务 rollout。
6.  运行 MT50 benchmark。
7.  冻结基线。
8.  开始 PTQ 实验。

------------------------------------------------------------------------

# 总结

正确的 Meta-World 工作流是：

    lerobot/smolvla_metaworld
                |
                v
    LeRobot Meta-World 评估器
                |
                v
    MT50 闭环 rollout
                |
                v
    成功率

核心要点：

不要将原生 Meta-World 的 gym 观测直接接入 SmolVLA。

应使用官方 LeRobot 的 Meta-World 集成与数据集格式，
从而一致地处理相机/状态/动作对齐。
