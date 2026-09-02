"""
Calibration for VLA-TCS2.

Ported to the opt-qt calibration flow (verified working reference:
/home/zyzhao/lfw_opt/opt-qt/main.py -> calibrate()).

Responsibilities
----------------
1. Decide per-layer action: reuse (scale files exist) vs recalibrate.
2. Bind a QuantStatManager to every QuantizedLinear / QuantizedMatMul.
3. Run scale_inspection forward over calibration batches.
4. Save scales (pickle) via QuantStatManager.save_all_scales().

Scale persistence follows opt-qt's pickle scheme:
    {layer_name}_{w|a|o}_scale_{layer_idx}.p
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors

from vla_tcs2.quant_linear import QuantizedLinear
from vla_tcs2.quant_matmul import QuantizedMatMul
from vla_tcs2.quant.stat_manager import QuantStatManager


# =============================================================================
# Action summary (reuse vs recalibrate)
# =============================================================================


def validate_reuse_layers_have_scales(model: torch.nn.Module) -> None:
    """Raise if a layer is set to reuse but its scale files are missing."""
    missing = []
    for m in model.modules():
        if isinstance(m, (QuantizedLinear, QuantizedMatMul)):
            action = m._resolve_calibration_action()
            if action == "reuse" and (not m._scale_files_exist()):
                missing.append(f"{m.layer_name}_{m.layer_idx}")
    if missing:
        raise FileNotFoundError(
            "These layers are set to reuse but scale files are missing:\n"
            + "\n".join(missing)
        )


def calibration_action_summary(model: torch.nn.Module) -> dict[str, int]:
    """Count quantized layers by their resolved calibration action."""
    summary = {"reuse": 0, "recalibrate": 0}
    for m in model.modules():
        if isinstance(m, (QuantizedLinear, QuantizedMatMul)):
            summary[m._resolve_calibration_action()] += 1
    return summary


# =============================================================================
# Calibration entry point (called by main.py)
# =============================================================================


def calibrate(
    model: torch.nn.Module,
    config: dict[str, Any],
    skip_calibration: bool = False,
) -> None:
    """
    Run calibration (opt-qt flow).

    main.py already calls wrapper.set_mode("scale_inspection") before
    invoking this function.
    """

    quant_cfg = config.get("quantization", {})

    scale_dir = Path(
        quant_cfg.get("scale_dir", "scales/default")
    )
    os.makedirs(scale_dir, exist_ok=True)

    device = torch.device(
        config.get("model", {}).get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )

    print(f"[calibration] scale_dir: {scale_dir}")

    # -------------------------------------------------------------------------
    # 1. Bind a stat manager to every quantized module
    # -------------------------------------------------------------------------

    stat_manager = QuantStatManager(str(scale_dir))

    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedMatMul)):
            module._stat_manager = stat_manager

    # -------------------------------------------------------------------------
    # 2. Action plan: reuse vs recalibrate
    # -------------------------------------------------------------------------

    summary = calibration_action_summary(model)
    print(f"Calibration action summary: {summary}")

    validate_reuse_layers_have_scales(model)

    # Only allow truly skipping when every layer already has scales.
    if skip_calibration:
        if summary["recalibrate"] == 0:
            print("✓ --skip-calibration: all quantized layers are reuse; skipping.")
            return
        print(
            "⚠ --skip-calibration is set, but some layers still require "
            "recalibration. Continuing."
        )

    if summary["recalibrate"] == 0:
        print("✓ All quantized layers are reuse; skipping calibration dataloader.")
        return

    # -------------------------------------------------------------------------
    # 3. Prepare calibration data
    # -------------------------------------------------------------------------

    calib_cfg = config.get("calibration", {})

    # The checkpoint's input features use camera1/camera2 keys (training
    # naming). Reuse the evaluation rename_map unless the calibration
    # section overrides it with its own dataset-facing names.
    if "rename_map" not in calib_cfg:
        calib_cfg["rename_map"] = (
            config.get("evaluation", {}).get("rename_map", {}) or {}
        )

    print("Preparing calibration data...")

    batches = prepare_calibration_batches(
        model=model,
        calib_cfg=calib_cfg,
        device=device,
        fallback_rename_map=(
            config.get("evaluation", {}).get("rename_map", {}) or {}
        ),
    )

    print(f"Running calibration on {len(batches)} batches...")

    # -------------------------------------------------------------------------
    # 4. Scale inspection forward
    # -------------------------------------------------------------------------

    model.eval()

    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(batches):
            # Deployment-faithful forward: the inference path exercised
            # during LIBERO rollouts (VLM backbone + action expert).
            model.predict_action_chunk(batch)

            if step % 10 == 0 or step == len(batches) - 1:
                print(f"[calibration] forward {step + 1}/{len(batches)}")

    calibration_time = time.time() - start_time

    # -------------------------------------------------------------------------
    # 5. Summary + save
    # -------------------------------------------------------------------------

    print("\n" + "-" * 80)
    stat_manager.print_summary()

    print("Saving quantization scales...")
    stat_manager.save_all_scales()

    print(f"\n✓ Calibration completed in {calibration_time:.2f}s")
    print(f"✓ Scales saved to: {scale_dir}")


# =============================================================================
# Calibration data (VLA path)
# =============================================================================


def prepare_calibration_batches(
    model: torch.nn.Module,
    calib_cfg: dict[str, Any],
    device: torch.device | None = None,
    fallback_rename_map: dict[str, str] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """
    Build the list of calibration batches (mirrors opt-qt CalibrationDataLoader,
    adapted to LeRobot datasets and SmolVLA's inference path).

    Flow:
        1. Probe total number of episodes via LeRobotDatasetMetadata.
        2. Sample `episodes` episode ids with `seed`.
        3. Load only those episodes into a LeRobotDataset.
        4. Take every `frame_stride`-th frame of each episode.
        5. Push each frame through the SAME preprocessor pipeline as
           evaluation (rename -> batch dim -> language tokenization ->
           to_device -> normalize with checkpoint stats).
        6. Collate into batches of `batch_size` frames.

    Returns a list of policy-forward input dicts consumable by
    `model.predict_action_chunk(batch)`.
    """

    repo_id = calib_cfg.get("dataset_repo_id")
    if not repo_id:
        raise ValueError(
            "Missing config entry: calibration.dataset_repo_id"
        )

    revision = calib_cfg.get("dataset_revision", None)
    n_episodes = int(calib_cfg.get("episodes", 8))
    batch_size = int(calib_cfg.get("batch_size", 8))
    frame_stride = int(calib_cfg.get("frame_stride", 4))
    seed = int(calib_cfg.get("seed", 42))
    rename_map = calib_cfg.get("rename_map", None)
    if rename_map is None:
        rename_map = fallback_rename_map or {}

    if device is None:
        device = next(model.parameters()).device

    # -------------------------------------------------------------------------
    # 1. Probe dataset metadata and sample episode ids
    # -------------------------------------------------------------------------

    meta = LeRobotDatasetMetadata(
        repo_id,
        revision=revision,
    )

    total_episodes = int(meta.total_episodes)

    if n_episodes > total_episodes:
        raise ValueError(
            f"Requested {n_episodes} calibration episodes, but dataset "
            f"'{repo_id}' only has {total_episodes}."
        )

    rng = random.Random(seed)
    episode_ids = sorted(
        rng.sample(range(total_episodes), n_episodes)
    )

    print(
        f"[calibration] dataset: {repo_id} (rev={revision}), "
        f"{total_episodes} episodes total, sampled: {episode_ids}"
    )

    # -------------------------------------------------------------------------
    # 2. Load only the sampled episodes
    # -------------------------------------------------------------------------

    dataset = LeRobotDataset(
        repo_id,
        revision=revision,
        episodes=episode_ids,
    )

    camera_keys = list(dataset.meta.camera_keys)

    # NOTE: with the episodes filter, dataset[i] expects a RELATIVE index
    # into the filtered dataset, not the absolute frame index. meta.episodes
    # holds ALL episodes of the repo, so filter rows to the selected ids
    # first, then walk lengths cumulatively (selected ids stay sorted).
    eps_table = dataset.meta.episodes
    selected = set(int(e) for e in dataset.episodes)

    rows = [
        (int(ep_idx), int(length))
        for ep_idx, length in zip(
            eps_table["episode_index"], eps_table["length"]
        )
        if int(ep_idx) in selected
    ]
    rows.sort(key=lambda r: r[0])

    frame_indices: list[int] = []
    offset = 0
    for _, length in rows:
        frame_indices.extend(
            range(offset, offset + length, frame_stride)
        )
        offset += length

    print(
        f"[calibration] {len(episode_ids)} episodes -> "
        f"{len(frame_indices)} frames (frame_stride={frame_stride})"
    )

    # -------------------------------------------------------------------------
    # 3. Preprocessor: identical pipeline to evaluation
    # -------------------------------------------------------------------------

    policy_cfg = model.config
    pretrained_path = getattr(policy_cfg, "pretrained_path")

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_path) if pretrained_path else None,
        preprocessor_overrides={
            "rename_observations_processor": {
                "rename_map": rename_map,
            },
        },
    )

    # -------------------------------------------------------------------------
    # 4. Per-frame preprocessing + collating
    # -------------------------------------------------------------------------

    batches: list[dict[str, torch.Tensor]] = []
    current: list[dict[str, torch.Tensor]] = []

    for idx in frame_indices:
        frame = dataset[idx]

        # Keep only observation fields the policy consumes.
        obs = {
            key: frame[key]
            for key in list(frame.keys())
            if key.startswith("observation.")
        }
        obs["task"] = frame.get("task", "")

        # (batch=1) forward through eval-identical preprocessing:
        # rename -> add batch dim -> tokenize language -> to_device -> normalize
        processed = preprocessor(obs)

        # Keep tensor fields only; the processor may passthrough unrelated
        # dataset columns (task string, info dicts, next.* flags).
        processed = {
            k: v
            for k, v in processed.items()
            if torch.is_tensor(v)
        }

        current.append(processed)

        if len(current) == batch_size:
            batches.append(_collate(current))
            current = []

    if current:
        batches.append(_collate(current))

    print(
        f"[calibration] prepared {len(batches)} batches "
        f"(batch_size={batch_size}, cameras={camera_keys})"
    )

    return batches


def _collate(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Stack per-frame (1, ...) tensors into (B, ...) batches.

    Language tensors (1, seq_i) may differ in sequence length across frames
    (pad_language_to="longest" pads per-frame). Pad them to the batch-wide
    longest so they can stack.
    """

    def pad_to(t: torch.Tensor, length: int) -> torch.Tensor:
        pad_len = length - t.shape[-1]
        if pad_len == 0:
            return t
        return torch.nn.functional.pad(t, (0, pad_len))

    keys = set(samples[0])
    for s in samples[1:]:
        keys &= set(s)

    batch: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [s[key] for s in samples]
        if not torch.is_tensor(values[0]):
            batch[key] = values
            continue

        if values[0].dim() >= 2 and key.startswith(
            "observation.language"
        ):
            max_len = max(v.shape[-1] for v in values)
            values = [pad_to(v, max_len) for v in values]

        batch[key] = torch.cat(values, dim=0)

    return batch