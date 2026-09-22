"""Scope and assumptions travel with every estimate, including offline reports."""
ASSUMPTIONS = [
    "Hypothetical uniform 1 MAC/PE/cycle; no measured chip timing or precision speedup.",
    "Output-stationary GEMM core only; excludes bias, scale storage/application, quantization and non-GEMM work.",
    "A/B reloaded per output tile and broadcast batch; no inter-operator weight/KV residency.",
    "DRAM bytes are mapping traffic, not unique tensor bytes; SRAM bandwidth/interconnect are not modeled.",
    "Sequential sum across captured calls; not end-to-end policy latency or per-action latency.",
    "Outlier, token mixed precision and BitNet events retain logical MACs but have no latency estimate.",
    "Only registered Linear and QuantizedMatMul modules are captured; functional matmul/SDPA, convolution and other operators are excluded.",
    "No sparse speedup, energy or area estimate in backend version 1.",
]


def new_totals():
    return dict(calls=0, dense_macs=0, estimated_calls=0, unsupported_calls=0,
                estimated_core_macs=0, estimated_core_cycles=0,
                dram_read_bytes=0, dram_write_bytes=0,
                sequential_core_latency_seconds=0.0, peak_buffer_bytes=0)


def accumulate(total, event, estimate):
    total["calls"] += 1
    total["dense_macs"] += event.macs
    if estimate.status != "estimated":
        total["unsupported_calls"] += 1
        return
    total["estimated_calls"] += 1
    total["estimated_core_macs"] += event.macs
    total["estimated_core_cycles"] += estimate.compute_cycles
    total["dram_read_bytes"] += estimate.dram_read_bytes
    total["dram_write_bytes"] += estimate.dram_write_bytes
    total["sequential_core_latency_seconds"] += estimate.latency_seconds
    total["peak_buffer_bytes"] = max(total["peak_buffer_bytes"], estimate.peak_buffer_bytes)
