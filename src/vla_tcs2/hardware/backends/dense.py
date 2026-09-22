"""Hypothetical dense GEMM core; NOT a calibrated chip or end-to-end model.

Each output tile stays in SRAM for the entire K reduction. A/B are reloaded
for every output tile and every broadcast batch. No inter-op residency.
Bias, scales, quantization, activations, cache management and interconnect
are outside this core model. Report those limitations with every run.
"""
from math import ceil
from ..schema import OpEstimate


def packed_bytes(elements, bits):
    return (elements * bits + 7) // 8


def tile_groups(length, tile):
    full, tail = divmod(length, tile)
    return [(tile, full)] * bool(full) + [(tail, 1)] * bool(tail)


class DenseAnalytic:
    name = "dense_analytic"
    version = "1"
    required_features = frozenset({"metadata"})

    def estimate(self, event, hardware, mapping):
        e, h, t = event, hardware, mapping
        def unsupported(reason):
            return OpEstimate(e.event_id, "unsupported", reason, e.macs)
        if e.metadata.get("is_bitnet") or e.metadata.get("mixed_precision"):
            return unsupported("BitNet/token mixed precision requires a dedicated backend")
        if e.mode not in {"raw", "quant_forward"}:
            return unsupported("Only raw and quant_forward core workloads are modeled")
        if e.mode == "quant_forward" and e.method not in {"per_tensor", "pot_fp8_per_tensor"}:
            return unsupported("Outlier/multi-path or unknown quantization method is not modeled")
        if max(e.a.bits, e.b.bits, e.output.bits) > 32:
            return unsupported("Operand width exceeds this backend's declared 32-bit limit")
        cycles = reads = writes = peak = 0
        for m, cm in tile_groups(e.m, t.tile_m):
            for n, cn in tile_groups(e.n, t.tile_n):
                repeats = e.batch * cm * cn
                acc_bytes = packed_bytes(m * n, h.accumulator_bits)
                writes += repeats * packed_bytes(m * n, e.output.bits)
                for k, ck in tile_groups(e.k, t.tile_k):
                    a_bytes = packed_bytes(m * k, e.a.bits)
                    b_bytes = packed_bytes(k * n, e.b.bits)
                    peak = max(peak, acc_bytes + a_bytes + b_bytes)
                    reads += repeats * ck * (a_bytes + b_bytes)
                    cycles += repeats * ck * ceil(m / h.pe_rows) * ceil(n / h.pe_cols) * (k + h.pipeline_cycles)
        # Explicitly require double buffering when overlap is requested.
        if h.overlap_compute_memory:
            max_m, max_n, max_k = min(e.m, t.tile_m), min(e.n, t.tile_n), min(e.k, t.tile_k)
            peak = (packed_bytes(max_m * max_n, h.accumulator_bits)
                    + 2 * (packed_bytes(max_m * max_k, e.a.bits)
                           + packed_bytes(max_k * max_n, e.b.bits)))
        if peak > h.sram_bytes:
            return unsupported(f"Tile needs {peak} SRAM bytes, capacity is {h.sram_bytes}")
        compute_s = cycles / h.frequency_hz
        memory_s = (reads + writes) / h.dram_bytes_per_second
        latency = max(compute_s, memory_s) if h.overlap_compute_memory else compute_s + memory_s
        return OpEstimate(e.event_id, "estimated", "GEMM core only; see report assumptions", e.macs,
                          cycles, reads, writes, peak, latency,
                          e.macs / (cycles * h.pe_rows * h.pe_cols))
