"""Capture real integer operands and ATen auxiliary work during ONE inference.

Quantized-module internals (FP reference GEMMs, logging and calibration) are
suppressed. Other transformer operators are observed without replacing outputs.
Files are intentionally separate from hardware traces, allowing offline replay.
"""

from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import weakref

import numpy as np
from .sparsity import pack_int4, unpack_int4, effective_mask


def read_records(root):
    with gzip.open(Path(root) / "operators.jsonl.gz", "rt") as f:
        for line in f:
            yield json.loads(line)


def read_code(root, desc):
    with np.load(Path(root) / desc["file"], allow_pickle=False) as z:
        a = z["data"]
        return (
            unpack_int4(a, desc["shape"])
            if desc["bits"] == 4
            else a.reshape(desc["shape"])
        )


def tensor_leaves(value):
    # Avoid importing torch on offline replay.
    if hasattr(value, "untyped_storage") and hasattr(value, "shape"):
        yield value
    elif isinstance(value, (list, tuple)):
        for x in value:
            yield from tensor_leaves(x)
    elif isinstance(value, dict):
        for x in value.values():
            yield from tensor_leaves(x)


class WorkloadCapture:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "codes").mkdir()
        (self.root / "sparsity").mkdir()
        (self.root / "iag").mkdir()
        (self.root / "lookups").mkdir()
        self.heads = 1
        self.stream = gzip.open(self.root / "operators.jsonl.gz", "wt")
        self.records = 0
        self.gemms = 0
        self.stack = []
        self.suppressed = 0
        self.storages = {}
        self.produced = set()
        self.origins = {}
        self.next_storage = 0
        self.code_cache = {}
        self.audit = []
        self.ops = {}
        self.scopes = set()
        self.unknown = set()
        self.output_overflows = 0
        self._handles = []

    def context(self):
        from vla_tcs2.runtime_context import get_runtime_context

        c = dict(get_runtime_context())
        phase = c.get("phase", "unknown")
        c["component"] = (
            "vlm"
            if phase == "prefill"
            else "expert" if phase == "denoise" else "unknown"
        )
        c["module_id"] = self.stack[-1][0] if self.stack else "functional"
        return c

    def active(self):
        if self.suppressed or not self.stack:
            return False
        paths = [name for name, _ in self.stack]
        if any(
            "vision_model" in x
            or "connector" in x
            or x.startswith(("vision.", "connector."))
            for x in paths
        ):
            return False
        return any(
            "vlm_with_expert" in x
            or "text_model" in x
            or "lm_expert" in x
            or x.startswith(("vlm.", "expert."))
            for x in paths
        )

    def tensor(self, t):
        storage = t.untyped_storage()
        key = (str(t.device), storage._cdata)
        entry = self.storages.get(key)
        if entry is None or not any(r() is not None for r in entry[1]):
            entry = (f"s{self.next_storage}", [])
            self.next_storage += 1
            self.storages[key] = entry
        # Bound weakref list for long-lived caches with many temporary views.
        entry[1][:] = [r for r in entry[1] if r() is not None]
        if not any(r() is t for r in entry[1]):
            entry[1].append(weakref.ref(t))
        # Mutating ops are made distinct by explicitly calling tensor before/after
        # and versioning the captured output in dispatch (ATen bumps version later).
        return {
            "id": entry[0],
            "shape": list(t.shape),
            "stride": list(t.stride()),
            "offset": t.storage_offset() * t.element_size(),
            "element_bytes": t.element_size(),
            "storage_bytes": storage.nbytes(),
            "numel": t.numel(),
            "dtype": str(t.dtype),
            "external": entry[0] not in self.produced,
            "version": getattr(t, "_version", 0),
            "source_phases": self.origins.get(entry[0], []),
        }

    def write(self, record):
        record["id"] = self.records
        self.records += 1
        self.stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def code(self, tensor, bits, *, static=False):
        a = tensor.detach().to("cpu").numpy()
        if not np.isfinite(a).all() or not np.equal(a, np.rint(a)).all():
            raise ValueError("Non-integral quantization code")
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        if a.size and (a.min() < lo or a.max() > hi):
            raise ValueError(f"Input/weight INT{bits} overflow")
        a = a.astype(np.int8 if bits <= 8 else np.int32)
        data = pack_int4(a) if bits == 4 else a
        digest = hashlib.sha256(
            data.tobytes() + str((a.shape, bits)).encode()
        ).hexdigest()
        # Deduplication here saves host artifact space only, not simulated EMA.
        file = f"codes/{digest}.npz"
        if not (self.root / file).exists():
            np.savez_compressed(self.root / file, data=data)
        return {"file": file, "sha256": digest, "shape": list(a.shape), "bits": bits}

    def quantized(self, layer, a, b, ac, bc, out, oc, *, linear):
        import torch

        # Serialization must not re-enter auxiliary ATen capture.
        with torch._C._DisableTorchDispatch():
            mid = layer.module_id
            if not mid.startswith(("vlm.", "expert.")):
                return
            if layer.method != "per_tensor" or layer.outlier_ratio:
                raise ValueError("Paper capture requires per_tensor without outliers")
            ab = layer.a_spec.bits if linear else layer.A_spec.bits
            bb = layer.w_spec.bits if linear else layer.B_spec.bits
            ob = layer.o_spec.bits if linear else layer.O_spec.bits
            expected = 4 if linear and mid.startswith("expert.") else 8
            if (ab, bb, ob) != (8, expected, 8):
                raise ValueError(f"Unexpected precision {mid}: {(ab, bb, ob)}")
            ad = self.code(ac, ab)
            # Static weights are stored once, but repeated uses remain in trace.
            wk = (mid, getattr(b, "_version", 0))
            if linear and wk in self.code_cache:
                bd = self.code_cache[wk]
            else:
                bd = self.code(bc, bb, static=linear)
                if linear:
                    self.code_cache[wk] = bd
            rawout = self.tensor(out)
            rawout["external"] = False
            self.produced.add(rawout["id"])
            rawout["source_phases"] = [self.context()["phase"]]
            self.origins[rawout["id"]] = rawout["source_phases"]
            overflow = int(((oc < -128) | (oc > 127)).sum().item())
            self.output_overflows += overflow
            ctx = self.context()
            ctx.update(module_id=mid, component=mid.split(".")[0])
            rec = dict(
                kind="gemm",
                **ctx,
                linear=linear,
                a=ad,
                b=bd,
                inputs=[self.tensor(a)] + ([] if linear else [self.tensor(b)]),
                outputs=[rawout],
                output_nominal_bits=8,
                output_code_storage_bits=32 if overflow else 8,
                output_overflow_elements=overflow,
                scales=[
                    float(getattr(layer, x))
                    for x in (
                        ("a_interval", "w_interval", "o_interval")
                        if linear
                        else ("A_interval", "B_interval", "O_interval")
                    )
                ],
                bias_elements=(
                    layer.bias.numel() if linear and layer.bias is not None else 0
                ),
            )
            # Compact exact p=8 masks for all batch/head and BG coordinates.
            arr = ac.detach().cpu().numpy().astype(np.int8)
            matrices = (
                arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
                if not linear
                else arr.reshape(1, -1, arr.shape[-1])
            )
            masks = np.stack(
                [
                    np.stack([effective_mask(x, lv, 8) for lv in range(4)])
                    for x in matrices
                ]
            )
            sf = f"sparsity/{self.records:08d}.npz"
            np.savez_compressed(
                self.root / sf,
                mask=np.packbits(masks, axis=-1, bitorder="little"),
                shape=np.asarray(masks.shape),
                bits=np.asarray(8),
                p=np.asarray(8),
                t=np.asarray(16),
            )
            rec["sparsity_file"] = sf
            if not linear and ".pv" in mid:
                if arr.ndim >= 4:
                    self.heads = arr.shape[-3]
                scores = np.zeros(matrices.shape[::2], dtype=np.int64)
                for lv in range(4):
                    # Token score uses individual effective flags, before p-token aggregation.
                    scores += np.sum(
                        ((matrices.astype(np.uint8) >> (2 * lv)) & 3) != 0, axis=1
                    ) * (4**lv)
                ig = f"iag/{self.records:08d}.npz"
                by_head = scores.reshape((*arr.shape[:-2], arr.shape[-1]))
                aggregate = by_head.sum(axis=-2) if arr.ndim >= 4 else by_head
                np.savez_compressed(
                    self.root / ig,
                    token_scores=scores,
                    token_scores_aggregate=aggregate,
                    batch_head_shape=np.asarray(arr.shape[:-2]),
                )
                rec["iag_file"] = ig
                rec["iag_kind"] = "token"
            elif linear and ("o_proj" in mid or "out_proj" in mid):
                from .sparsity import digits

                heads = self.heads if arr.shape[-1] % self.heads == 0 else 1
                scores = np.zeros(heads, dtype=np.int64)
                flat = arr.reshape(-1, heads, arr.shape[-1] // heads)
                for lv, d in enumerate(digits(flat)):
                    scores += np.count_nonzero(d, axis=(0, 2)) * (4**lv)
                ig = f"iag/{self.records:08d}.npz"
                np.savez_compressed(self.root / ig, head_scores=scores)
                rec["iag_file"] = ig
                rec["iag_kind"] = "head"
            self.write(rec)
            self.gemms += 1
            self.audit.append({"module_id": mid, "bits": [ab, bb, ob]})

    def dispatch(self, func, args, kwargs):
        if not self.active():
            return func(*args, **kwargs)
        before = [self.tensor(t) for t in tensor_leaves((args, kwargs))]
        out = func(*args, **kwargs)
        after = [self.tensor(t) for t in tensor_leaves(out)]
        name = str(func)
        # True aliases are free; materialized clone/repeat/cat remain data copies.
        before_ids = {d["id"] for d in before}
        alias = (
            bool(after)
            and all(d["id"] in before_ids for d in after)
            and not func._schema.is_mutable
        )
        if alias:
            return out
        if not after:
            return out
        for d in after:
            self.produced.add(d["id"])
            d["external"] = False
            d["source_phases"] = sorted(
                set(p for x in before for p in x.get("source_phases", []))
            ) or [self.context()["phase"]]
            self.origins[d["id"]] = d["source_phases"]
        ctx = self.context()
        self.scopes.add(ctx["component"])
        self.ops[name] = self.ops.get(name, 0) + 1
        record = dict(
            kind="aux",
            op=name,
            **ctx,
            inputs=before,
            outputs=after,
            mutable=func._schema.is_mutable,
        )
        if name.split(".")[1] in {"embedding", "index_select", "gather", "index"}:
            record["lookup_file"] = self.lookup(name.split(".")[1], args, before[0])
        self.write(record)
        return out

    def lookup(self, name, args, desc):
        """Capture selected storage addresses, without reading the whole table."""
        import torch

        with torch._C._DisableTorchDispatch():
            cpu = lambda t: t.detach().cpu().numpy()
            if name == "embedding":
                rows = cpu(args[1]).reshape(-1).astype(np.int64)
                offsets = (
                    desc["offset"] + rows * desc["stride"][0] * desc["element_bytes"]
                )
                lengths = np.full(
                    rows.shape, desc["shape"][1] * desc["element_bytes"], dtype=np.int64
                )
                if desc["stride"][1] != 1:
                    raise ValueError("Embedding table must have contiguous rows")
            else:
                shape = tuple(desc["shape"])
                if name == "index":
                    selection = tuple(
                        slice(None) if x is None else cpu(x) for x in args[1]
                    )
                elif name == "index_select":
                    dim = int(args[1]) % len(shape)
                    selection = tuple(
                        cpu(args[2]) if i == dim else slice(None)
                        for i in range(len(shape))
                    )
                offsets = None
                for dim, size in enumerate(shape):
                    axis = np.arange(size, dtype=np.int64).reshape(
                        tuple(size if i == dim else 1 for i in range(len(shape)))
                    )
                    if name == "gather":
                        indices = cpu(args[2])
                        selected = (
                            indices
                            if dim == int(args[1]) % len(shape)
                            else np.broadcast_to(
                                np.arange(indices.shape[dim]).reshape(
                                    tuple(
                                        indices.shape[dim] if i == dim else 1
                                        for i in range(len(shape))
                                    )
                                ),
                                indices.shape,
                            )
                        )
                    else:
                        selected = np.broadcast_to(axis, shape)[selection]
                    term = selected * desc["stride"][dim] * desc["element_bytes"]
                    offsets = term if offsets is None else offsets + term
                offsets = (
                    np.asarray(offsets, dtype=np.int64).reshape(-1) + desc["offset"]
                )
                lengths = np.full(offsets.shape, desc["element_bytes"], dtype=np.int64)
            # Merge adjacent accesses; retain duplicates so SRAM service remains explicit.
            if len(offsets):
                cuts = np.r_[
                    0,
                    np.flatnonzero(offsets[1:] != offsets[:-1] + lengths[:-1]) + 1,
                    len(offsets),
                ]
                lengths = np.add.reduceat(lengths, cuts[:-1])
                offsets = offsets[cuts[:-1]]
            file = f"lookups/{self.records:08d}.npz"
            np.savez_compressed(self.root / file, offsets=offsets, lengths=lengths)
            return file

    def __call__(self, model):
        self.model = model
        return self

    def __enter__(self):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from vla_tcs2.hardware.quant_observer import observe_quantized_operands
        from vla_tcs2.quant_linear import QuantizedLinear
        from vla_tcs2.quant_matmul import QuantizedMatMul

        owner = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return owner.dispatch(func, args, kwargs or {})

        for path, module in self.model.named_modules():
            if hasattr(module, "num_attention_heads"):
                self.heads = int(module.num_attention_heads)
            quant = isinstance(module, (QuantizedLinear, QuantizedMatMul))

            def pre(m, args, path=path, quant=quant):
                owner.stack.append((getattr(m, "module_id", "") or path, quant))
                owner.suppressed += int(quant)

            def post(m, args, out, quant=quant):
                owner.suppressed -= int(quant)
                owner.stack.pop()

            self._handles.extend(
                (
                    module.register_forward_pre_hook(pre),
                    module.register_forward_hook(post, always_call=True),
                )
            )
        self._exit = ExitStack()
        self._exit.enter_context(observe_quantized_operands(self.quantized))
        self._exit.enter_context(Mode())
        return self

    def __exit__(self, typ, value, tb):
        self._exit.__exit__(typ, value, tb)
        for handle in self._handles:
            handle.remove()
        self.stream.close()
        meta = dict(
            schema_version=1,
            status="failed" if typ else "complete",
            records=self.records,
            gemms=self.gemms,
            auxiliary_ops=self.ops,
            output_overflow_elements=self.output_overflows,
            auxiliary_scopes=sorted(self.scopes),
            assumptions=[
                "ATen materialized GQA repeats are explicit copies; views alias storage",
                "raw functional boundaries retain actual FP container width",
                "packed file deduplication does not imply hardware cache hits",
            ],
        )
        (self.root / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
        (self.root / "precision_audit.json").write_text(
            json.dumps(self.audit, indent=2) + "\n"
        )
