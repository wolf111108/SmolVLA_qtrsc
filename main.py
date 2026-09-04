#!/usr/bin/env python

"""
VLA-TCS2 main pipeline.

Responsibilities
----------------
main.py only manages the experiment flow:

    load config
        ↓
    build / wrap model
        ↓
    calibration (optional)
        ↓
    switch execution mode
        ↓
    evaluation (optional)
        ↓
    save results

Detailed implementations belong to:

model_wrapper.py:
    - Load the original VLA model
    - Replace selected Linear / MatMul operations
    - Switch quantization modes

calibration.py:
    - Read calibration settings from config
    - Prepare calibration data
    - Run calibration forward
    - Collect quantization scales
    - Save scales to the configured directory

quant_linear.py / quant_matmul.py:
    - Raw forward
    - Scale inspection
    - Quantized forward

eval.py:
    - Read evaluation settings from config
    - Run LIBERO evaluation
    - Return evaluation results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from vla_tcs2.calibration import calibrate
from vla_tcs2.eval import evaluate
from vla_tcs2.model_wrapper import (
    ModelWrapper,
    apply_sensitivity_target,
)


# =============================================================================
# Configuration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLA-TCS2 quantization and evaluation pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )

    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration and reuse existing scale files.",
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation.",
    )

    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load experiment configuration from YAML.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid config file: {path}"
        )

    return config


# =============================================================================
# Utilities
# =============================================================================


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)
    print()


def prepare_output_dir(
    config: dict[str, Any],
) -> Path:
    """
    Create output directory for this experiment.
    """

    output_dir = Path(
        config.get(
            "output_dir",
            "outputs/default_run",
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def save_config_snapshot(
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save the exact configuration used for this run.
    """

    output_path = output_dir / "config.yaml"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def save_results(
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save evaluation results in JSON format.
    """

    output_path = output_dir / "result.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1. Load experiment configuration
    # -------------------------------------------------------------------------

    config = load_config(
        args.config
    )

    output_dir = prepare_output_dir(
        config
    )

    save_config_snapshot(
        config=config,
        output_dir=output_dir,
    )

    quant_cfg = config.get(
        "quantization",
        {},
    )

    quant_enabled = quant_cfg.get(
        "enabled",
        False,
    )

    print_header("VLA-TCS2")

    print(f"Config          : {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Quantization    : {quant_enabled}")

    # -------------------------------------------------------------------------
    # 2. Build model
    # -------------------------------------------------------------------------

    print_header("STEP 1: BUILD MODEL")

    wrapper = ModelWrapper(
        config=config,
    )

    model = wrapper.build()

    model.eval()

    print("Model build completed.")

    # -------------------------------------------------------------------------
    # 3. Prepare model execution mode
    # -------------------------------------------------------------------------

    if quant_enabled:

        # ---------------------------------------------------------------------
        # 3.1 Calibration
        # ---------------------------------------------------------------------

        if not args.skip_calibration:

            print_header("STEP 2: CALIBRATION")

            wrapper.set_mode(
                "scale_inspection"
            )

            calibrate(
                model=model,
                config=config,
            )

            print("Calibration completed.")

        else:

            print_header("STEP 2: CALIBRATION")

            print(
                "Calibration skipped. "
                "Existing scale files will be reused."
            )

        # ---------------------------------------------------------------------
        # 3.2 Quantized inference
        # ---------------------------------------------------------------------

        test_cfg = config.get("test", {})
        if test_cfg.get("enabled", False):
            # Sensitivity experiment: isolate a single physical site with
            # test_forward (noise injection), everything else raw. Requires
            # no calibration for gaussian methods; quant_residual methods
            # reuse the calibrated scale files above.
            print_header("SENSITIVITY TARGET")

            n_target = apply_sensitivity_target(model, config)
            if n_target == 0:
                raise RuntimeError(
                    "test.enabled=true but no module matched test.target. "
                    "Check the module_id / component / layer / operator "
                    "selectors."
                )
        else:
            wrapper.set_mode(
                "quant_forward"
            )

            print(
                "Quantized inference mode enabled."
            )

    else:

        # ---------------------------------------------------------------------
        # Floating-point baseline
        # ---------------------------------------------------------------------

        wrapper.set_mode(
            "raw"
        )

        print(
            "Quantization disabled. "
            "Running floating-point baseline."
        )

    # -------------------------------------------------------------------------
    # 4. Evaluation
    # -------------------------------------------------------------------------

    if args.skip_evaluation:

        print_header("PIPELINE COMPLETED")

        print("Evaluation skipped.")

        return

    print_header("STEP 3: EVALUATION")

    results = evaluate(
        model=model,
        config=config,
        output_dir=output_dir,
    )

    # -------------------------------------------------------------------------
    # 5. Save results
    # -------------------------------------------------------------------------

    save_results(
        results=results,
        output_dir=output_dir,
    )

    # -------------------------------------------------------------------------
    # 6. Final summary
    # -------------------------------------------------------------------------

    print_header("PIPELINE COMPLETED")

    print(
        f"Results saved to: "
        f"{output_dir / 'result.json'}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()