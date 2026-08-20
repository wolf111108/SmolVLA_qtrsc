import numpy as np

from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action


def print_nested(obj, prefix=""):
    """递归打印 observation 中每个字段的类型和 shape。"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else key
            print_nested(value, name)
    elif hasattr(obj, "shape"):
        print(
            f"{prefix}: "
            f"shape={obj.shape}, "
            f"dtype={getattr(obj, 'dtype', None)}"
        )
    else:
        print(f"{prefix}: {type(obj).__name__} = {obj}")


def main():
    suite_name = "libero_spatial"
    task_id = 0

    print("=== LIBERO Smoke Test ===")

    # --------------------------------------------------
    # 1. 获取 benchmark 和 task
    # --------------------------------------------------
    benchmark_dict = benchmark.get_benchmark_dict()

    print("available suites:")
    print(sorted(benchmark_dict.keys()))

    suite = benchmark_dict[suite_name]()
    task = suite.get_task(task_id)

    print("\n=== Task ===")
    print("suite:", suite_name)
    print("num_tasks:", len(suite.tasks))
    print("task_id:", task_id)
    print("task_name:", task.name)
    print("language:", task.language)

    # --------------------------------------------------
    # 2. 创建 LeRobot LIBERO 环境
    # --------------------------------------------------
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        obs_type="pixels_agent_pos",
        init_states=True,
        episode_index=0,
        n_envs=1,
    )

    try:
        print("\n=== Environment ===")
        print("observation_space:", env.observation_space)
        print("action_space:", env.action_space)

        # --------------------------------------------------
        # 3. Reset
        # --------------------------------------------------
        print("\nResetting environment ...")

        obs, info = env.reset(seed=0)

        print("\n=== Reset Result ===")
        print("info:", info)

        print("\nObservation:")
        print_nested(obs)

        # --------------------------------------------------
        # 4. Dummy action
        # --------------------------------------------------
        action = np.asarray(
            get_libero_dummy_action(),
            dtype=np.float32,
        )

        print("\n=== Dummy Action ===")
        print("action:", action)
        print("shape:", action.shape)
        print("dtype:", action.dtype)

        # --------------------------------------------------
        # 5. Step
        # --------------------------------------------------
        print("\nExecuting one env.step() ...")

        obs2, reward, terminated, truncated, info2 = env.step(action)

        print("\n=== Step Result ===")
        print("reward:", reward)
        print("terminated:", terminated)
        print("truncated:", truncated)
        print("info:", info2)

        print("\nObservation after step:")
        print_nested(obs2)

        print("\nLIBERO smoke test: PASS")

    finally:
        env.close()


if __name__ == "__main__":
    main()
