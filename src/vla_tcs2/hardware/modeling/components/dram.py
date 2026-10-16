"""Shared aggregate DRAM bus; effective service rate, no DRAM command timing."""
from dataclasses import dataclass
from .storage import Storage, StorageSpec
from ..protocol import positive, integer


@dataclass(frozen=True)
class DRAMSpec:
    bandwidth_bytes_per_second: float = 102.4e9
    efficiency: float = 0.7
    latency_cycles: int = 100
    burst_bytes: int = 64

    def __post_init__(self):
        positive(self.bandwidth_bytes_per_second, "bandwidth_bytes_per_second")
        positive(self.efficiency, "efficiency")
        if self.efficiency > 1:
            raise ValueError("efficiency must be <= 1")
        integer(self.latency_cycles, "latency_cycles")
        integer(self.burst_bytes, "burst_bytes", 1)


class DRAM(Storage):
    def __init__(self, component_id, spec, clock_hz):
        self.dram_spec = spec
        bandwidth = spec.bandwidth_bytes_per_second * spec.efficiency / clock_hz
        super().__init__(component_id, StorageSpec(
            capacity_bytes=1, read_bytes_per_cycle=bandwidth,
            write_bytes_per_cycle=bandwidth, shared_read_write=True,
            latency_cycles=spec.latency_cycles, burst_bytes=spec.burst_bytes))
