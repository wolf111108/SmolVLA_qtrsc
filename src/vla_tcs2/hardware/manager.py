"""Bounded runtime capture and streaming reports, independent of QuantStatManager."""
import csv
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .backends import create_backend
from .capture import make_event
from .report import ASSUMPTIONS, accumulate, new_totals
from .schema import HardwareSpec, MappingSpec, SCHEMA_VERSION


class HardwareManager:
    def __init__(self, output_dir, *, backend="dense_analytic", hardware=None,
                 mapping=None, max_events=10000, provenance=None):
        self.output_dir = Path(output_dir)
        self.backend = create_backend(backend)
        if not self.backend.required_features <= {"metadata"}:
            raise ValueError("This capture version provides metadata only; backend requires unavailable features")
        self.hardware = HardwareSpec(**(hardware or {}))
        self.mapping = MappingSpec(**(mapping or {}))
        if type(max_events) is not int or max_events < 0:
            raise ValueError("max_events must be a nonnegative integer (0 = unlimited)")
        self.max_events = max_events
        self.provenance = provenance or {}
        self.totals = new_totals()
        self.groups = {}
        self.unsupported = {}
        self.attached_modules = {}
        self.module_calls = {}
        self.skipped_modes = {}
        self.skipped_limit = 0
        self.generations = set()
        self.unknown_generation_calls = 0
        self._handles = []
        self._streams = []
        self._opened = False
        self._used = False
        self.manifest = {}

    @classmethod
    def from_config(cls, config, output_dir, provenance=None):
        options = dict(config)
        enabled = options.pop("enabled", False)
        if type(enabled) is not bool:
            raise ValueError("hardware.enabled must be bool")
        if not enabled:
            return None
        return cls(output_dir, provenance=provenance, **options)

    def __enter__(self):
        if self._used:
            raise RuntimeError("Use a fresh manager for each capture/replay session")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # A trace is evidence: never silently overwrite a previous capture.
        for name in ("trace.jsonl", "operators.csv", "summary.json"):
            if (self.output_dir / name).exists():
                raise FileExistsError(self.output_dir / name)
        try:
            self._trace = open(self.output_dir / "trace.jsonl", "x", encoding="utf-8")
            self._streams.append(self._trace)
            table = open(self.output_dir / "operators.csv", "x", newline="", encoding="utf-8")
            self._streams.append(table)
            self._csv = csv.DictWriter(table, fieldnames=[
                "event_id", "module_id", "component", "phase", "flow_step", "generation_id", "op_type",
                "m", "n", "k", "batch", "status", "reason", "dense_macs", "compute_cycles",
                "dram_read_bytes", "dram_write_bytes", "peak_buffer_bytes", "latency_seconds", "pe_utilization",
            ])
            self._csv.writeheader()
            self.manifest = dict(record_type="manifest", schema_version=SCHEMA_VERSION,
                                 backend=self.backend.name, backend_version=self.backend.version,
                                 hardware=asdict(self.hardware), mapping=asdict(self.mapping),
                                 max_events=self.max_events, available_features=["metadata"],
                                 provenance=self.provenance, assumptions=ASSUMPTIONS)
            self._trace.write(json.dumps(self.manifest) + "\n")
            self._opened = self._used = True
            return self
        except BaseException:
            for stream in self._streams:
                stream.close()
            raise

    def submit(self, event):
        if not self._opened:
            raise RuntimeError("Open manager with a context manager before submitting events")
        if self.max_events and self.totals["calls"] >= self.max_events:
            self.skipped_limit += 1
            return
        if event.event_id != self.totals["calls"]:
            raise ValueError("Submitted event IDs must be contiguous from zero")
        estimate = self.backend.estimate(event, self.hardware, self.mapping)
        if estimate.status not in {"estimated", "unsupported"} or estimate.event_id != event.event_id:
            raise ValueError("Backend returned an invalid estimate")
        self._trace.write(json.dumps(dict(record_type="operator", **event.to_dict())) + "\n")
        row = asdict(estimate)
        row.update({name: getattr(event, name) for name in (
            "module_id", "component", "phase", "flow_step", "generation_id", "op_type", "m", "n", "k", "batch")})
        self._csv.writerow(row)
        accumulate(self.totals, event, estimate)
        # Include physical identity; never group by a shared scale filename.
        key = (event.module_id, event.component, event.phase, event.flow_step)
        accumulate(self.groups.setdefault(key, new_totals()), event, estimate)
        self.module_calls[event.module_id] = self.module_calls.get(event.module_id, 0) + 1
        if event.generation_id >= 0:
            self.generations.add(event.generation_id)
        else:
            self.unknown_generation_calls += 1
        if estimate.status == "unsupported":
            self.unsupported[estimate.reason] = self.unsupported.get(estimate.reason, 0) + 1

    def attach(self, model):
        if not self._opened or self._handles:
            raise RuntimeError("Attach once inside an open manager session")
        # These imports are intentionally lazy: offline replay does not need torch/LeRobot.
        import torch.nn as nn
        from vla_tcs2.quant_linear import QuantizedLinear
        from vla_tcs2.quant_matmul import QuantizedMatMul
        from vla_tcs2.runtime_context import get_runtime_context

        def hook(module_id, kind):
            def collect(module, args, kwargs, output):
                mode = getattr(module, "mode", "raw")
                if mode == "scale_inspection":
                    self.skipped_modes[mode] = self.skipped_modes.get(mode, 0) + 1
                    return
                if self.max_events and self.totals["calls"] >= self.max_events:
                    self.skipped_limit += 1
                    return
                event = make_event(self.totals["calls"], module_id, kind, module,
                                   args, kwargs, output, get_runtime_context())
                self.submit(event)
                # Returning None preserves the original output object.
            return collect

        try:
            for name, module in model.named_modules():
                kind = "matmul" if isinstance(module, QuantizedMatMul) else (
                    "linear" if isinstance(module, (nn.Linear, QuantizedLinear)) else None)
                if kind is None:
                    continue
                module_id = getattr(module, "module_id", "") or name or "root"
                if module_id in self.attached_modules:
                    raise ValueError(f"Duplicate physical module_id: {module_id}")
                self.attached_modules[module_id] = kind
                self._handles.append(module.register_forward_hook(hook(module_id, kind), with_kwargs=True))
            if not self._handles:
                raise ValueError("No supported Linear/QuantizedMatMul modules found")
        except BaseException:
            self.detach()
            raise

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def capture_model(self, model):
        with self:
            self.attach(model)
            try:
                yield self
            finally:
                self.detach()

    def __exit__(self, exc_type, exc, tb):
        self.detach()
        try:
            summary = dict(
                schema_version=SCHEMA_VERSION, status="failed" if exc_type else "completed",
                error_type=exc_type.__name__ if exc_type else None,
                manifest=self.manifest, totals=self.totals,
                captured_generation_count=len(self.generations),
                unknown_generation_calls=self.unknown_generation_calls,
                capture_truncated=bool(self.skipped_limit), skipped_limit=self.skipped_limit,
                skipped_modes=self.skipped_modes, unsupported_reasons=self.unsupported,
                attached_modules=self.attached_modules, captured_module_calls=self.module_calls,
                attached_but_unobserved=sorted(set(self.attached_modules) - set(self.module_calls)),
                scope="Captured GEMM cores only; never an end-to-end latency claim",
                groups=[dict(module_id=k[0], component=k[1], phase=k[2], flow_step=k[3], **v)
                        for k, v in sorted(self.groups.items())],
            )
            with open(self.output_dir / "summary.json", "x", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2)
                stream.write("\n")
        finally:
            self._opened = False
            for stream in self._streams:
                stream.close()
        return False
