import numpy as np

from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action


def main():
    suite_name = "libero_spatial"
    task_id = 0

    print("=== LIBERO Long Render Test ===")

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)

    print("task:", task.name)
    print("language:", task.language)

    # 尽可能复现 lerobot-eval 当前配置：
    # 2 cameras, 360x360, pixels_agent_pos, relative control
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        observation_width=360,
        observation_height=360,
        init_states=True,
        episode_index=0,
        n_envs=1,
        control_mode="relative",
        hard_reset=True,
    )

    try:
        obs, info = env.reset(seed=1000)

        print("reset: PASS")
        print("image shape:", obs["pixels"]["image"].shape)
        print("image2 shape:", obs["pixels"]["image2"].shape)

        action = np.asarray(
            get_libero_dummy_action(),
            dtype=np.float32,
        )

        print("\nStarting 120 env.step() calls ...")

        for step in range(120):
            obs, reward, terminated, truncated, info = env.step(action)

            # 强制真正访问图像数据
            img1 = obs["pixels"]["image"]
            img2 = obs["pixels"]["image2"]

            if (step + 1) % 10 == 0:
                print(
                    f"step={step + 1:3d} "
                    f"img1_mean={img1.mean():.3f} "
                    f"img2_mean={img2.mean():.3f} "
                    f"success={info['is_success']}"
                )

            if terminated or truncated:
                print("environment terminated at step:", step + 1)
                break

        print("\nLIBERO long render test: PASS")

    finally:
        env.close()


if __name__ == "__main__":
    main()
