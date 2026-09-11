#!/usr/bin/env python
"""
Offline deployment-faithful sparsity / workload profile runner
(research manual §40-§45).

Flow per experiment config:
    build wrapped model -> reuse existing scales (skip calibration)
    -> enable sparsity -> N frames of predict_action_chunk (batch=1,
       i.e. real prefill x1 + denoise x10 per frame)
    -> export module_sparsity / workload / weight / unit CSVs
    -> print phase / flow-step summary

Usage:
    conda run -n smolvla_eval python scripts/run_sparsity_profile.py \
        --config configs/experiments/smolvla_int8_outlier_quant_test.yaml \
        --frames 8 \
        --out outputs/sparsity/int8_outlier

Scale files are REUSED as-is (calibration_policy forced to reuse when
files exist); add --recalibrate to force a fresh calibration first.
"""

import argparse
import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import torch  # noqa: E402
import yaml  # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def profile_one(config_path: str, frames: int, out_dir: str,
                recalibrate: bool = False, unit: bool = False) -> dict:
    from vla_tcs2.calibration import (
        calibrate,
        prepare_calibration_batches,
    )
    from vla_tcs2.model_wrapper import ModelWrapper
    from vla_tcs2.runtime_context import get_runtime_context

    config = load_config(config_path)
    os.makedirs(out_dir, exist_ok=True)

    # Snapshot config + git metadata for reproducibility (manual §81).
    meta = {
        "config": os.path.relpath(config_path, _REPO_ROOT),
        "frames": frames,
        "recalibrate": recalibrate,
    }
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        meta["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", _REPO_ROOT, "status", "--porcelain"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
        )
    except Exception:
        pass
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- build (runtime hooks auto-installed) -----------------------------
    wrapper = ModelWrapper(config=config)
    model = wrapper.build()
    model.eval()

    device = torch.device(
        config.get("model", {}).get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # --- scales: reuse existing (or recalibrate on demand) ----------------
    if not recalibrate:
        # Force every layer to REUSE existing scale files: profile runs
        # must not silently trigger a full recalibration (the experiment
        # configs' layer_policy often says "recalibrate").
        for m in model.modules():
            if type(m).__name__ in ("QuantizedLinear", "QuantizedMatMul"):
                m.calibration_policy = "reuse"
        wrapper.set_mode("scale_inspection")
        calibrate(model=model, config=config, skip_calibration=True)
    else:
        wrapper.set_mode("scale_inspection")
        calibrate(model=model, config=config)

    # calibrate() creates a NEW QuantStatManager and rebinds it onto every
    # quantized module — wrapper.stat_manager is stale afterwards. Always
    # re-fetch the live manager from the modules.
    sm = None
    for m in model.modules():
        if hasattr(m, "_stat_manager") and m._stat_manager is not None:
            sm = m._stat_manager
            break
    if sm is None:
        raise RuntimeError("no stat manager bound to quantized modules")
    wrapper.stat_manager = sm
    sm.enable_sparsity()
    # Unit sparsity is the most expensive metric (per-bit x per-dim grouping
    # on every tensor); keep it OFF for quick bit-level baselines. Enable
    # with --unit when needed (manual §36).
    if not unit:
        sm.configure_unit_sparsity(enable=False)

    # Weight sparsity: static, once per module (manual §24, §63).
    wrapper.set_mode("quant_forward")
    # Scales are loaded lazily on first forward; force-load so the weight
    # collector sees intervals.
    for m in model.modules():
        if type(m).__name__ in ("QuantizedLinear", "QuantizedMatMul"):
            if m.mode == "quant_forward":
                try:
                    m._load_scales()
                except FileNotFoundError:
                    pass
    n_weight = sm.collect_model_weight_sparsity(model)
    print(f"[profile] weight sparsity collected: {n_weight} modules")

    # --- runtime profile: deployment-faithful batch=1 forwards ------------
    calib_cfg = dict(config.get("calibration", {}))
    calib_cfg["batch_size"] = 1  # deployment batch (manual §43)
    calib_cfg.setdefault(
        "rename_map",
        config.get("evaluation", {}).get("rename_map", {}) or {},
    )

    # Take the first `frames` batches of the standard calibration sampling.
    calib_cfg["episodes"] = max(2, frames // 4 + 1)
    calib_cfg["frame_stride"] = max(8, 64 // max(frames, 1))

    batches = prepare_calibration_batches(
        model=model, calib_cfg=calib_cfg, device=device,
    )[:frames]
    print(f"[profile] running {len(batches)} batch=1 forwards "
          f"(each = prefill x1 + denoise x10)")

    torch.manual_seed(0)  # action-noise seed control (manual §44)
    with torch.no_grad():
        for i, batch in enumerate(batches):
            ctx_before = get_runtime_context()
            model.predict_action_chunk(batch)
            print(f"[profile] frame {i + 1}/{len(batches)} done "
                  f"(gen_id={get_runtime_context()['generation_id']}, "
                  f"phase_before={ctx_before['phase']})")

    # --- export ------------------------------------------------------------
    tag = os.path.basename(config_path).replace(".yaml", "")
    sm.export_module_sparsity_csv(
        os.path.join(out_dir, "module_sparsity.csv"), config_name=tag,
    )
    sm.export_workload_csv(
        model, os.path.join(out_dir, "workload.csv"), config_name=tag,
    )
    sm.export_per_layer_sparsity_csv(
        os.path.join(out_dir, "per_layer_sparsity.csv"),
    )
    sm.export_per_layer_weight_sparsity_csv(
        os.path.join(out_dir, "weight_sparsity.csv"),
    )
    if unit:
        sm.export_unit_sparsity_csv(
            os.path.join(out_dir, "unit_sparsity.csv"), tag,
        )
    sm.export_collected_layers_csv(
        os.path.join(out_dir, "collected_layers.csv"), tag,
    )

    # --- console summary ----------------------------------------------------
    sm.print_global_sparsity()
    sm.print_prefill_decode_sparsity()

    summary = {
        "frames": len(batches),
        "generations": get_runtime_context()["generation_id"],
        "weight_modules": n_weight,
        "prefill_elements": sm.phase_sparsity["prefill"]["total_element_count"],
        "decode_elements": sm.phase_sparsity["decode"]["total_element_count"],
        "flow_steps_seen": sorted(sm.flow_step_sparsity.keys()),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[profile] summary: {json.dumps(summary)}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument(
        "--unit", action="store_true",
        help="Also collect/export unit (2x2 bit x dim) sparsity (slower).",
    )
    args = parser.parse_args()

    profile_one(args.config, args.frames, args.out, args.recalibrate,
                args.unit)


if __name__ == "__main__":
    main()
