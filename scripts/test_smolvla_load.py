import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


CHECKPOINT = "lerobot/smolvla_libero"


def main():
    print("=== SmolVLA Load Test ===")
    print("checkpoint:", CHECKPOINT)

    print("\nLoading policy ...")

    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT)

    print("\n=== Policy ===")
    print("type:", type(policy))
    print("device before move:", next(policy.parameters()).device)

    # 参数统计
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )

    print("total parameters:", f"{total_params:,}")
    print("trainable parameters:", f"{trainable_params:,}")

    # 配置
    print("\n=== Config ===")
    print(policy.config)

    # 输入 / 输出 feature
    print("\n=== Input Features ===")
    for name, feature in policy.config.input_features.items():
        print(name, "->", feature)

    print("\n=== Output Features ===")
    for name, feature in policy.config.output_features.items():
        print(name, "->", feature)

    # 移到 GPU
    print("\nMoving policy to CUDA ...")
    policy = policy.to("cuda")
    policy.eval()

    print("device:", next(policy.parameters()).device)
    print("dtype:", next(policy.parameters()).dtype)

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"CUDA allocated: {allocated:.3f} GB")
    print(f"CUDA reserved:  {reserved:.3f} GB")

    print("\nSmolVLA load test: PASS")


if __name__ == "__main__":
    main()
