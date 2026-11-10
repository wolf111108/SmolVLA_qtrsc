"""Streaming, exact sign/magnitude activation BG statistics (no model dependency).

p groups consecutive token rows at fixed channel and magnitude-bit position.
Batch/head axes remain separate; neither padding nor FP-protected lanes enter
the native BG denominator. Full-width, wholly native packs are also reported.
"""

from collections import defaultdict
from contextlib import contextmanager
import csv
from pathlib import Path

import numpy as np

P_VALUES = (2, 4, 8, 16, 32, 64, 128)
COUNTERS = (
    "calls", "total_packs", "normal_zero_packs", "full_native_packs",
    "full_native_zero_packs", "native_groups", "native_skipped_groups",
    "protected_groups", "tail_packs", "minus128_elements",
)


def count_bit_groups(code, protected=None, p_values=P_VALUES, token_axis=-2):
    """Count real signed INT8 codes, retaining abs(-128)=128 without overflow.

    Four two-bit magnitude groups use shifts 0/2/4/6. Sign is separate; this is
    lossless sign + 8 magnitude bits, not two's-complement bipolar inversion.
    The 8th magnitude bit is only used by -128 in the original INT8 domain.
    """
    code = np.asarray(code)
    if code.ndim < 2 or not code.size or not np.isfinite(code).all():
        raise ValueError("Expected nonempty, finite [..., tokens, channels] codes")
    if np.any(code != np.rint(code)) or code.min() < -128 or code.max() > 127:
        raise ValueError("Expected integer INT8 codes in [-128, 127]")
    p_values = tuple(p_values)
    if not p_values or len(set(p_values)) != len(p_values) or any(p not in P_VALUES for p in p_values):
        raise ValueError("p must be distinct entries from 2,4,8,16,32,64,128")
    if token_axis not in (-1, -2):
        raise ValueError("token_axis must be -1 (QK B) or -2")
    mask = np.zeros(code.shape, dtype=bool) if protected is None else np.broadcast_to(
        np.asarray(protected, dtype=bool), code.shape
    )
    if np.any(code[mask] != 0):
        raise ValueError("FP-protected positions must be zero in normal-path codes")
    code = np.moveaxis(code, token_axis, -2).astype(np.int16)
    mask = np.moveaxis(mask, token_axis, -2)
    m, k = code.shape[-2:]
    code = code.reshape(-1, m, k)
    mask = mask.reshape(code.shape)
    magnitude = np.abs(code)
    result = []
    for p in p_values:
        n = (m + p - 1) // p
        pad = ((0, 0), (0, n * p - m), (0, 0))
        valid = np.pad(~mask, pad).reshape(-1, n, p, k)
        native_per_pack = valid.sum(axis=2)
        full_native = native_per_pack == p
        base = dict(
            calls=1, total_packs=code.shape[0] * n * k,
            full_native_packs=int(full_native.sum()),
            native_groups=int((~mask).sum()), protected_groups=int(mask.sum()),
            tail_packs=code.shape[0] * k if m % p else 0,
            minus128_elements=int((code == -128).sum()),
        )
        for level in range(4):
            nonzero = ((magnitude >> (2 * level)) & 3) != 0
            active = np.pad(nonzero, pad).reshape(-1, n, p, k).any(axis=2)
            result.append(dict(
                p=p, level=level, **base,
                normal_zero_packs=int((~active).sum()),
                full_native_zero_packs=int((~active & full_native).sum()),
                native_skipped_groups=int(native_per_pack[~active].sum()),
            ))
    return result


def rates(row):
    row = dict(row)
    for name, numerator, denominator in (
        ("native_skip_rate", "native_skipped_groups", "native_groups"),
        ("full_native_pack_zero_rate", "full_native_zero_packs", "full_native_packs"),
        ("normal_pack_zero_rate", "normal_zero_packs", "total_packs"),
        ("tail_pack_fraction", "tail_packs", "total_packs"),
    ):
        row[name] = row[numerator] / row[denominator] if row[denominator] else None
    return row


def emit_activation_code(layer, role, code, protected):
    """Cheap disabled path; called with the real code and already-computed mask."""
    collector = getattr(layer, "_bit_group_collector", None)
    if collector is not None:
        collector(layer, role, code, protected)


@contextmanager
def observe_modules(modules, collector):
    # Module binding works even when LIBERO invokes the policy in a worker thread.
    missing = object()
    old = [(m, getattr(m, "_bit_group_collector", missing)) for m in modules]
    try:
        for m, _ in old:
            m._bit_group_collector = collector
        yield collector
    finally:
        for m, value in old:
            if value is missing:
                del m._bit_group_collector
            else:
                m._bit_group_collector = value


class BitGroupCollector:
    def __init__(self, p_values=P_VALUES):
        self.p_values = tuple(p_values)
        self.counts = defaultdict(lambda: dict.fromkeys(COUNTERS, 0))
        self.shapes = defaultdict(set)
        self.observed_calls = defaultdict(int)

    def __call__(self, layer, role, code, protected):
        import torch
        from vla_tcs2.runtime_context import get_runtime_context

        mid = layer.module_id
        component = mid.split('.')[0]
        if component not in ("vlm", "expert") or role not in ("activation", "A", "B"):
            raise ValueError(f"Unexpected activation site: {mid}/{role}")
        spec = layer.a_spec if role == "activation" else getattr(layer, f"{role}_spec")
        if spec.name() != "int8":
            raise ValueError(f"Expected INT8 activation: {mid}/{role}")
        ctx = get_runtime_context()
        phase, step = ctx["phase"], ctx["flow_step"]
        if (component == "vlm" and (phase, step) != ("prefill", -1)) or (
            component == "expert" and (phase != "denoise" or step not in range(10))
        ):
            raise ValueError(f"Missing/wrong runtime context: {mid}: {ctx}")
        # The methods produce exact INT8-valued floats. Copy without truncating
        # so the independent NumPy validation also detects invalid codes.
        array = code.detach().to(device="cpu", dtype=torch.float32).numpy()
        mask = protected.detach().to(device="cpu").numpy()
        axis = -1 if role == "B" and mid.endswith(".qk") else -2
        key = (component, mid, phase, step, ctx["attention_kind"], role)
        self.shapes[key].add(tuple(code.shape))
        self.observed_calls[key] += 1
        for row in count_bit_groups(array, mask, self.p_values, token_axis=axis):
            dest = self.counts[(*key, row["p"], row["level"])]
            for name in COUNTERS:
                dest[name] += row[name]

    def rows(self):
        keys = ("component", "module_id", "phase", "flow_step", "attention_kind", "role", "p", "level")
        return [rates(dict(zip(keys, key), **value)) for key, value in sorted(self.counts.items())]

    def summary(self):
        buckets = defaultdict(lambda: dict.fromkeys(COUNTERS, 0))
        for row in self.rows():
            scopes = ["all_inputs"]
            if row["role"] in ("activation", "A"):
                scopes.append("bsc_left")
            for component in (row["component"], "all"):
                for scope in scopes:
                    for level in (str(row["level"]), "all"):
                        dest = buckets[(component, scope, row["p"], level)]
                        for name in COUNTERS:
                            dest[name] += row[name]
        return [rates(dict(zip(("component", "scope", "p", "level"), key), **value))
                for key, value in sorted(buckets.items())]

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, rows in (("by_site.csv", self.rows()), ("summary.csv", self.summary())):
            if not rows:
                raise ValueError("No activation codes observed; refusing empty report")
            with (directory / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = self.summary()
        text = ["# Activation 2-bit group sparsity", "",
                "Sign + magnitude; p consecutive tokens at fixed channel. Counts are pooled before division.",
                "Native BG skip rate excludes FP-protected and padded lanes. FP sidepath work remains.",
                "Full native pack rate excludes tails and every pack touching an FP-protected lane.", ""]
        for scope, title in (("bsc_left", "BSC left inputs: Linear activation + QK/PV A"),
                             ("all_inputs", "All activation inputs: additionally QK/PV B")):
            text.extend([f"## {title}", "", "| p | VLM native BG skip | Expert native BG skip | VLM full-pack zero | Expert full-pack zero |",
                         "|---|---:|---:|---:|---:|"])
            for p in self.p_values:
                pair = {r["component"]: r for r in summary
                        if r["scope"] == scope and r["p"] == p and r["level"] == "all"}
                values = [pair[c][metric] for metric in ("native_skip_rate", "full_native_pack_zero_rate")
                          for c in ("vlm", "expert")]
                cells = ["N/A" if x is None else f"{100*x:.4f}%" for x in values]
                text.append(f"| {p} | " + " | ".join(cells) + " |")
            text.append("")
        (directory / "summary.md").write_text("\n".join(text) + "\n")
