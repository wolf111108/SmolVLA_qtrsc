#!/usr/bin/env python3
"""Capture exactly one VLM-prefill + ten-step expert quantized action chunk."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/int8_w4.yaml",
    )
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--checkpoint")
    p.add_argument("--revision")
    p.add_argument("--device")
    p.add_argument(
        "--batch",
        type=Path,
        help="Reuse phaseL input_batch.pt or ONE preprocessed real observation",
    )
    p.add_argument(
        "--reuse-scales",
        type=Path,
        help="Compatible VLM/expert static INT scales; otherwise recalibrate",
    )
    a = p.parse_args()
    os.environ["VLA_TENSOR_DUMP_DIR"] = ""
    import torch
    import yaml
    from vla_tcs2.model_wrapper import ModelWrapper
    from vla_tcs2.quant_linear import QuantizedLinear
    from vla_tcs2.quant_matmul import QuantizedMatMul
    from vla_tcs2.hardware.paper.workload import WorkloadCapture
    from vla_tcs2.hardware.paper.replay import validate_workload

    c = yaml.safe_load(a.config.read_text())
    for key, value in [
        ("path", a.checkpoint),
        ("revision", a.revision),
        ("device", a.device),
    ]:
        if value is not None:
            c["model"][key] = value
    if a.checkpoint and Path(a.checkpoint).exists() and not a.revision:
        c["model"]["revision"] = None
    out = (a.output_dir or ROOT / c["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    c["quantization"]["scale_dir"] = str(
        a.reuse_scales.resolve() if a.reuse_scales else out / "scales"
    )
    c["quantization"]["calibration_policy"] = {
        "default": "reuse" if a.reuse_scales else "recalibrate"
    }
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    versions = {}
    for pkg in ("torch", "numpy", "transformers", "lerobot", "huggingface_hub"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unavailable"
    meta = dict(
        status="running",
        versions=versions,
        checkpoint=c["model"],
        seed=c["experiment"]["seed"],
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        input_source=(
            str(a.batch.resolve())
            if a.batch
            else "seeded synthetic observation; no task-success claim"
        ),
        calibration=(
            "reuse"
            if a.reuse_scales
            else "one uncaptured same-observation calibration chunk"
        ),
        scope="VLM/expert actual quantized GEMMs and observed auxiliary ATen operations; vision/connector FP32 outside target",
    )
    try:
        seed = c["experiment"]["seed"]
        torch.manual_seed(seed)
        wrapper = ModelWrapper(c)
        model = wrapper.build(mode="scale_inspection")
        device = torch.device(c["model"]["device"])
        model.to(device).float().eval()
        modules = [
            m
            for m in model.modules()
            if isinstance(m, (QuantizedLinear, QuantizedMatMul))
        ]
        audit = []
        for m in modules:
            mid = m.module_id
            if not mid.startswith(("vlm.", "expert.")):
                raise ValueError(f"Non-target module quantized: {mid}")
            bits = (
                (m.a_spec.bits, m.w_spec.bits, m.o_spec.bits)
                if isinstance(m, QuantizedLinear)
                else (m.A_spec.bits, m.B_spec.bits, m.O_spec.bits)
            )
            expected = (
                (8, 4, 8)
                if isinstance(m, QuantizedLinear) and mid.startswith("expert.")
                else (8, 8, 8)
            )
            if bits != expected or m.method != "per_tensor" or m.outlier_ratio:
                raise ValueError(f"Unexpected precision {mid}: {bits}")
            audit.append(dict(module_id=mid, bits=bits))
        if not modules:
            raise ValueError("No quantized modules installed")
        (out / "precision_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        torch.manual_seed(seed)
        if a.batch:
            batch = torch.load(a.batch, map_location="cpu", weights_only=True)
        else:
            batch = {
                key: torch.rand((1, *feature.shape))
                for key, feature in model.config.image_features.items()
            }
            from lerobot.utils.constants import (
                OBS_LANGUAGE_ATTENTION_MASK,
                OBS_LANGUAGE_TOKENS,
                OBS_STATE,
            )

            batch[OBS_STATE] = torch.zeros(
                (1, *model.config.input_features[OBS_STATE].shape)
            )
            tokenizer = model.model.vlm_with_expert.processor.tokenizer
            tokens = tokenizer(
                [c["experiment"]["prompt"]],
                padding="max_length",
                truncation=True,
                max_length=model.config.tokenizer_max_length,
                return_tensors="pt",
            )
            batch[OBS_LANGUAGE_TOKENS] = tokens["input_ids"]
            batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].bool()
        if not isinstance(batch, dict) or any(
            not isinstance(v, torch.Tensor) or v.shape[0] != 1 for v in batch.values()
        ):
            raise ValueError("Expected one preprocessed tensor batch")
        batch = {k: v.to(device) for k, v in batch.items()}
        torch.save({k: v.cpu() for k, v in batch.items()}, out / "input_batch.pt")
        meta.update(
            input_shapes={k: list(v.shape) for k, v in batch.items()},
            chunk_size=model.config.chunk_size,
            num_steps=model.config.num_steps,
            wrapped_modules=len(modules),
        )
        if model.config.num_steps != 10 or model.config.chunk_size != 50:
            raise ValueError("Expected 50 actions and 10 denoise steps")
        if not a.reuse_scales:
            model.reset()
            torch.manual_seed(seed)
            print("[calibration] one uncaptured chunk", flush=True)
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type, enabled=False),
            ):
                model.predict_action_chunk(dict(batch))
            wrapper.stat_manager.save_all_scales()
        for m in modules:
            if not m._scale_files_exist():
                raise FileNotFoundError(f"Missing scales: {m.module_id}")
        wrapper.set_mode("quant_forward")
        model.reset()
        torch.manual_seed(seed)
        started = time.perf_counter()
        print(
            "[capture] one complete quantized inference; serializing real integer codes",
            flush=True,
        )
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            with WorkloadCapture(out / "workload")(model):
                actions = model.predict_action_chunk(dict(batch))
        if device.type == "cuda":
            torch.cuda.synchronize()
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite actions")
        torch.save(actions.cpu(), out / "actions.pt")
        manifest = validate_workload(out / "workload")
        if not manifest["auxiliary_ops"]:
            raise ValueError("Auxiliary capture empty; check model instrumentation")
        meta.update(
            status="complete",
            action_shape=list(actions.shape),
            captured_gemms=manifest["gemms"],
            host_capture_seconds=time.perf_counter() - started,
        )
        print(f"[done] {out}; run replay.py next", flush=True)
    except BaseException as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
