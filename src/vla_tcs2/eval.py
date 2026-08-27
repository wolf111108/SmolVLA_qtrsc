"""
Evaluation for VLA-TCS2.

Responsibilities
----------------
evaluate() runs LIBERO rollouts with the (already wrapped, possibly
quantized) policy in-process and returns the metric dictionary.

This is the in-memory counterpart of `lerobot-eval`: the policy is
passed as an object instead of a checkpoint path, so quantized modules
(QuantizedLinear / QuantizedMatMul) survive evaluation.

Expected top-level config structure:

    evaluation:
        env:
            type: libero
            task: libero_spatial        # one suite, or null for all
        n_episodes: 10
        batch_size: 1
        use_async_envs: false
        seed: 1000
        max_episodes_rendered: 0        # 0 = no videos
        rename_map:                     # optional (e.g. A-class checkpoint)
            observation.images.image: observation.images.camera1
            observation.images.image2: observation.images.camera2
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from lerobot.envs import (
    close_envs,
    make_env,
    make_env_config,
    make_env_pre_post_processors,
)
from lerobot.policies import make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.random_utils import set_seed

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation entry point (called by main.py)
# =============================================================================


def evaluate(
    model: torch.nn.Module,
    config: dict[str, Any],
    output_dir: Path | str,
) -> dict[str, Any]:
    """
    Run LIBERO evaluation with the given (wrapped) policy in-process.

    Returns the aggregated metrics dict, and also persists it as
    `output_dir/eval_info.json` (same schema as lerobot-eval).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config.get(
        "evaluation",
        {},
    )

    env_kwargs = dict(
        eval_cfg.get(
            "env",
            {},
        )
    )
    env_type = str(
        env_kwargs.pop(
            "type",
            "libero",
        )
    )

    n_episodes = int(eval_cfg.get("n_episodes", 10))
    batch_size = int(eval_cfg.get("batch_size", 1))
    use_async_envs = bool(eval_cfg.get("use_async_envs", False))
    seed = eval_cfg.get("seed", 1000)
    max_episodes_rendered = int(eval_cfg.get("max_episodes_rendered", 0))
    rename_map = eval_cfg.get("rename_map", {}) or {}

    device = str(
        config.get("model", {}).get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )

    set_seed(seed)

    # -------------------------------------------------------------------------
    # 1. Environment
    # -------------------------------------------------------------------------

    print(f"[eval] env: {env_type} {env_kwargs}")

    env_cfg = make_env_config(
        env_type,
        **env_kwargs,
    )

    envs = make_env(
        env_cfg,
        n_envs=batch_size,
        use_async_envs=use_async_envs,
    )

    try:
        # ---------------------------------------------------------------------
        # 2. Processors (policy + env)
        # ---------------------------------------------------------------------

        preprocessor_overrides = {
            "device_processor": {
                "device": str(getattr(model, "device", None) or device),
            },
            "rename_observations_processor": {
                "rename_map": rename_map,
            },
        }

        # Reload the processor pipelines from the checkpoint so normalizer
        # stats (mean/std) are preserved — mirrors lerobot-eval's
        # `make_pre_post_processors(..., pretrained_path=cfg.policy.pretrained_path)`.
        pretrained_path = getattr(model.config, "pretrained_path", None)

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=model.config,
            pretrained_path=str(pretrained_path) if pretrained_path else None,
            preprocessor_overrides=preprocessor_overrides,
        )

        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg,
            policy_cfg=model.config,
        )

        # ---------------------------------------------------------------------
        # 3. Rollouts
        # ---------------------------------------------------------------------

        print(
            f"[eval] {n_episodes} episodes "
            f"x {batch_size} envs (async={use_async_envs})"
        )

        model.eval()

        info = eval_policy_all(
            envs=envs,
            policy=model,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=output_dir / "videos",
            return_episode_data=False,
            start_seed=seed,
            max_parallel_tasks=getattr(
                env_cfg, "max_parallel_tasks", 1
            ),
        )

    finally:
        close_envs(envs)

    # -------------------------------------------------------------------------
    # 4. Persist + return
    # -------------------------------------------------------------------------

    info_path = output_dir / "eval_info.json"

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    overall = info.get("overall", {})

    print("[eval] overall:", overall)
    print(f"[eval] results saved to: {info_path}")

    return info
