"""Scope and assumptions travel with every estimate, including offline reports."""
ASSUMPTIONS = [
    "Hypothetical uniform 1 MAC/PE/cycle; no measured chip timing or precision speedup.",
    "GEMM core only; OS/WS/IS describe single-buffer temporal mappings, not PE-level dataflow.",
    "OS retains output over K; WS retains B over M; IS retains A over N. WS/IS include accumulator-precision spills.",
    "No cross-batch/inter-operator weight/KV residency; excludes bias, scale storage/application, quantization and non-GEMM work.",
    "DRAM bytes are mapping traffic, not unique tensor bytes; SRAM bandwidth/interconnect are not modeled.",
    "Sequential sum across captured calls; not end-to-end policy latency or per-action latency.",
    "Outlier, token mixed precision and BitNet events retain logical MACs but have no latency estimate.",
    "Only registered Linear and QuantizedMatMul modules are captured; functional matmul/SDPA, convolution and other operators are excluded.",
    "No sparse speedup, energy or area estimate. Logical low-bit packing assumes a matching storage format.",
    "Shared DRAM service is serialized; independent streams and compute/memory overlap are ideal lower bounds, not scheduled simulation.",
    "Mapping search is project analytic code, not official Timeloop; optimal only over the recorded candidate sets.",
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
