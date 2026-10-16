"""Aggregate per-operator component simulation estimates."""


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
