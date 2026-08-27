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
import time
from pathlib import Path
from typing import Any

import torch

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

    print("Preparing calibration data...")

    batches = prepare_calibration_batches(calib_cfg=calib_cfg)

    print(f"Running calibration on {len(batches)} batches...")

    # -------------------------------------------------------------------------
    # 4. Scale inspection forward
    # -------------------------------------------------------------------------

    model.eval()

    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(batches):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            model(batch)

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
    calib_cfg: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    """
    Build the list of calibration batches.

    TODO: implement. Suggested approach (mirrors opt-qt CalibrationDataLoader):
      - load a LeRobotDataset (calibration.dataset_repo_id / revision)
      - sample `episodes` episodes with `seed`
      - take every frame_stride-th frame of each episode
      - batch them into policy-forward input dicts
        (images, state, language prompt) matching SmolVLAPolicy format
    """

    n_episodes = calib_cfg.get("episodes", 8)
    batch_size = calib_cfg.get("batch_size", 8)
    frame_stride = calib_cfg.get("frame_stride", 4)
    seed = calib_cfg.get("seed", 42)

    print(
        f"[calibration] TODO prepare_calibration_batches "
        f"(episodes={n_episodes}, batch_size={batch_size}, "
        f"frame_stride={frame_stride}, seed={seed})"
    )

    # placeholder: no batches -> no new scales collected
    return []
