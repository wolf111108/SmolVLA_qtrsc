#!/usr/bin/env python3
"""One calibrated, dense GEMM-only SmolVLA action-chunk capture; no dataset loop."""
import argparse
from copy import deepcopy
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/int8_w4.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", help="Override checkpoint repo ID or local path")
    parser.add_argument("--revision", help="Override checkpoint revision; config pins the public checkpoint")
    parser.add_argument("--device", help="Override model device, e.g. cuda or cpu")
    parser.add_argument("--batch", type=Path, help="Optional trusted torch file containing ONE already-preprocessed model batch")
    parser.add_argument("--reuse-scales", type=Path, help="Use compatible per-site INT scales instead of one-input smoke calibration")
    args = parser.parse_args()
    # Prevent calibration from dumping every weight/activation tensor.
    os.environ["VLA_TENSOR_DUMP_DIR"] = ""
    import torch
    import yaml
    from vla_tcs2.hardware import HardwareManager
    from vla_tcs2.model_wrapper import ModelWrapper
    from vla_tcs2.quant_linear import QuantizedLinear
    from vla_tcs2.quant_matmul import QuantizedMatMul

    config = yaml.safe_load(args.config.read_text())
    config = deepcopy(config)
    for key, value in (("path", args.checkpoint), ("revision", args.revision), ("device", args.device)):
        if value is not None:
            config["model"][key] = value
    if args.checkpoint and Path(args.checkpoint).exists() and not args.revision:
        config["model"]["revision"] = None
    out = args.output_dir or ROOT / config["output_dir"]
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_metadata.json").exists() or (out / "hardware").exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    config["quantization"]["scale_dir"] = str(args.reuse_scales.resolve() if args.reuse_scales else out / "scales")
    config["quantization"]["calibration_policy"] = {"default": "reuse" if args.reuse_scales else "recalibrate"}
    for key in ("vision", "connector"):
        config["quantization"][key]["calibration_policy"] = "reuse" if args.reuse_scales else "recalibrate"
    config["quantization"]["vision"]["matmul"]["calibration_policy"] = "reuse" if args.reuse_scales else "recalibrate"
    hardware = config["hardware"]
    if hardware["backend_config"]["architecture"]["components"]["array"].get("mode", "dense") != "dense":
        raise ValueError("This experiment requires a dense array")
    if hardware.get("max_events", 0) != 0 or hardware.get("mapping_search", {}).get("enabled", False):
        raise ValueError("This baseline requires unlimited capture and a fixed mapping")
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    versions = {}
    for name in ("torch", "torchvision", "transformers", "lerobot", "huggingface_hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    meta = {"status": "running", "config": str(args.config.resolve()), "versions": versions,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "checkpoint": config["model"], "seed": config["experiment"]["seed"],
            "input_source": str(args.batch.resolve()) if args.batch else "seeded synthetic observation; no task-success claim",
            "calibration": "reuse compatible scales" if args.reuse_scales else "one uncaptured scale-inspection action chunk on the same input",
            "scope": "one captured quant_forward action chunk; dense GEMM-only target model; host time is separate"}
    def save_meta():
        (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    save_meta()
    try:
        seed = config["experiment"]["seed"]
        torch.manual_seed(seed)
        wrapper = ModelWrapper(config)
        model = wrapper.build(mode="scale_inspection")
        device = torch.device(config["model"]["device"])
        model.to(device).eval()
        quantized = [m for m in model.modules() if isinstance(m, (QuantizedLinear, QuantizedMatMul))]
        # Assert actual installed specs, rather than trusting config spelling.
        audit = []
        for module in quantized:
            mid = module.module_id
            if isinstance(module, QuantizedLinear):
                bits = (module.a_spec.bits, module.w_spec.bits, module.o_spec.bits)
                expected = (8, 4, 8) if mid.startswith("expert.") else (8, 8, 8)
            else:
                bits = (module.A_spec.bits, module.B_spec.bits, module.O_spec.bits)
                expected = (8, 8, 8)  # attention B is K/V activation, not a learned weight
            if bits != expected or module.method != "per_tensor" or module.outlier_ratio:
                raise ValueError(f"Unexpected precision/method at {mid}: {bits}, {module.method}")
            audit.append({"module_id": mid, "bits": bits, "method": module.method})
        (out / "precision_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        torch.manual_seed(seed)
        if args.batch:
            batch = torch.load(args.batch, map_location="cpu", weights_only=True)
        else:
            batch = {key: torch.rand((1, *feature.shape)) for key, feature in model.config.image_features.items()}
            from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
            batch[OBS_STATE] = torch.zeros((1, *model.config.input_features[OBS_STATE].shape))
            tokenizer = model.model.vlm_with_expert.processor.tokenizer
            tokens = tokenizer([config["experiment"]["prompt"]], padding="max_length", truncation=True,
                               max_length=model.config.tokenizer_max_length, return_tensors="pt")
            batch[OBS_LANGUAGE_TOKENS] = tokens["input_ids"]
            batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].bool()
        if not isinstance(batch, dict) or any(not isinstance(v, torch.Tensor) or v.shape[0] != 1 for v in batch.values()):
            raise ValueError("Batch must be a tensor dictionary with batch size exactly one; preprocess externally first")
        batch = {k: v.to(device) for k, v in batch.items()}
        torch.save({k: v.cpu() for k, v in batch.items()}, out / "input_batch.pt")
        meta["input_shapes"] = {k: list(v.shape) for k, v in batch.items()}
        meta["chunk_size"] = model.config.chunk_size
        meta["num_steps"] = model.config.num_steps
        meta["vlm_model_name"] = model.config.vlm_model_name
        meta["vlm_config_revision"] = getattr(model.model.vlm_with_expert.config, "_commit_hash", None)
        meta["wrapped_modules"] = len(quantized)
        if not args.reuse_scales:
            model.reset()
            torch.manual_seed(seed)
            print("[calibration] one uncaptured action chunk", flush=True)
            with torch.no_grad():
                model.predict_action_chunk(dict(batch))
            wrapper.stat_manager.save_all_scales()
        for module in quantized:
            if not module._scale_files_exist():
                raise FileNotFoundError(f"No complete scales for {module.module_id}")
        wrapper.set_mode("quant_forward")
        # No sparsity provider/collector is enabled. Preserve full non-GEMM execution on host.
        model.reset()
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        options = {k: v for k, v in hardware.items() if k != "enabled"}
        manager = HardwareManager(out / "hardware", provenance={"experiment": str(args.config), "source_commit": meta["source_commit"]},
                                  **options)
        with manager.capture_model(model):
            print("[capture] one quant_forward action chunk", flush=True)
            with torch.no_grad():
                actions = model.predict_action_chunk(dict(batch))
            if device.type == "cuda":
                torch.cuda.synchronize()
        meta["host_seconds_including_capture_and_simulation"] = time.perf_counter() - started
        if not torch.isfinite(actions).all():
            raise ValueError("Inference produced non-finite actions")
        torch.save(actions.cpu(), out / "actions.pt")
        summary = json.loads((out / "hardware/summary.json").read_text())
        from vla_tcs2.hardware.trace import read_trace
        events = list(read_trace(out / "hardware/trace.jsonl"))[1:]
        generations = {e.generation_id for e in events if e.generation_id >= 0}
        steps = sorted({e.flow_step for e in events if e.phase == "denoise"})
        if len(generations) != 1 or steps != list(range(model.config.num_steps)):
            raise ValueError(f"Incomplete inference coverage: generations={generations}, denoise steps={steps}")
        if summary["capture_truncated"] or summary["unsupported_reasons"]:
            raise ValueError(f"Incomplete GEMM simulation: {summary['unsupported_reasons']}")
        for component in ("vision", "connector", "vlm", "expert"):
            if not any(e.component == component for e in events):
                raise ValueError(f"No captured GEMMs from {component}")
        meta.update(status="completed", captured_generations=len(generations), flow_steps=steps,
                    action_shape=list(actions.shape), captured_gemms=len(events))
        print(f"[done] {out}; export with export_timeline.py --run-dir {out / 'hardware'}", flush=True)
    except BaseException as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_meta()


if __name__ == "__main__":
    main()
