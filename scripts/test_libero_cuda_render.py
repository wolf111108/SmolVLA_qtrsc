import numpy as np
import torch

from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action


def main():
    print("=== LIBERO + CUDA Render Test ===")

    suite_name = "libero_spatial"
    task_id = 0

    suite = benchmark.get_benchmark_dict()[suite_name]()

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

    # 保持一组 CUDA tensor 常驻 GPU
    a = torch.randn(
        4096, 4096,
        device="cuda",
        dtype=torch.bfloat16,
    )
    b = torch.randn(
        4096, 4096,
        device="cuda",
        dtype=torch.bfloat16,
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "CUDA allocated:",
        f"{torch.cuda.memory_allocated() / 1024**3:.3f} GB",
    )

    try:
        obs, info = env.reset(seed=1000)

        print("reset: PASS")
        print("image:", obs["pixels"]["image"].shape)
        print("image2:", obs["pixels"]["image2"].shape)

        action = np.asarray(
            get_libero_dummy_action(),
            dtype=np.float32,
        )

        print("\nStarting 120 CUDA + env.step() iterations ...")

        for step in range(120):

            # -------------------------------------------------
            # 模拟模型在同一张 GPU 上进行 CUDA 计算
            # -------------------------------------------------
            c = torch.matmul(a, b)

            # 确保 CUDA 计算真正完成
            torch.cuda.synchronize()

            # 防止 Python 优化掉引用
            cuda_value = c[0, 0].item()

            # -------------------------------------------------
            # 随后进行 MuJoCo / EGL rendering
            # -------------------------------------------------
            obs, reward, terminated, truncated, info = env.step(action)

            img1 = obs["pixels"]["image"]
            img2 = obs["pixels"]["image2"]

            if (step + 1) % 10 == 0:
                print(
                    f"step={step + 1:3d} "
                    f"cuda={cuda_value:.4f} "
                    f"img1_mean={img1.mean():.3f} "
                    f"img2_mean={img2.mean():.3f}"
                )

            if terminated or truncated:
                print("environment terminated:", step + 1)
                break

        print("\nLIBERO + CUDA render test: PASS")

    finally:
        env.close()


if __name__ == "__main__":
    main()
