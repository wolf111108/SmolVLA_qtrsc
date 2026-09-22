"""Versioned, tensor-free hardware workload contracts (standard library only)."""
from dataclasses import asdict, dataclass, field
from math import isfinite, prod
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TensorDesc:
    shape: tuple[int, ...]
    storage_dtype: str
    logical_format: str
    bits: int

    def __post_init__(self):
        if not self.shape or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("Only nonempty positive tensor shapes are supported")
        if type(self.bits) is not int or self.bits <= 0:
            raise ValueError("bits must be a positive integer")

    @property
    def elements(self):
        return prod(self.shape)


@dataclass(frozen=True)
class OperatorEvent:
    event_id: int
    module_id: str
    component: str
    op_type: str
    a: TensorDesc
    b: TensorDesc
    output: TensorDesc
    m: int
    n: int
    k: int
    batch: int = 1
    batch_shape: tuple[int, ...] = ()
    phase: str = "unknown"
    flow_step: int = -1
    generation_id: int = -1
    attention_kind: str = "unknown"
    mode: str = "raw"
    method: str = "raw"
    weight_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported event schema: {self.schema_version}")
        if self.op_type not in {"linear", "matmul"}:
            raise ValueError(f"Unsupported op_type: {self.op_type}")
        if any(type(v) is not int or v <= 0 for v in (self.m, self.n, self.k, self.batch)):
            raise ValueError("M/N/K/batch must be positive integers")
        if not self.module_id or self.event_id < 0:
            raise ValueError("Event requires physical module_id and nonnegative event_id")
        if self.output.elements != self.batch * self.m * self.n:
            raise ValueError("Output shape does not match GEMM dimensions")
        if self.op_type == "linear":
            if (self.batch != 1 or self.a.shape[-1] != self.k
                    or self.a.elements != self.m * self.k
                    or self.b.shape != (self.n, self.k)
                    or self.output.shape != self.a.shape[:-1] + (self.n,)):
                raise ValueError("Invalid Linear shapes")
        else:
            if len(self.a.shape) < 2 or len(self.b.shape) < 2:
                raise ValueError("MatMul vectors are not supported")
            batch_shape = broadcast_shape(self.a.shape[:-2], self.b.shape[:-2])
            if (self.a.shape[-2:] != (self.m, self.k)
                    or self.b.shape[-2:] != (self.k, self.n)
                    or self.batch_shape != batch_shape or self.batch != prod(batch_shape)
                    or self.output.shape != batch_shape + (self.m, self.n)):
                raise ValueError("Invalid batched MatMul shapes")

    @property
    def macs(self):
        return self.batch * self.m * self.n * self.k

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Missing or unsupported event schema_version")
        for name in ("a", "b", "output"):
            desc = dict(data[name])
            desc["shape"] = tuple(desc["shape"])
            data[name] = TensorDesc(**desc)
        data["batch_shape"] = tuple(data.get("batch_shape", ()))
        return cls(**data)


def broadcast_shape(a, b):
    result = []
    for i in range(1, max(len(a), len(b)) + 1):
        x = a[-i] if i <= len(a) else 1
        y = b[-i] if i <= len(b) else 1
        if x != y and x != 1 and y != 1:
            raise ValueError(f"Incompatible batch dimensions: {a}, {b}")
        result.append(max(x, y))
    return tuple(reversed(result))


@dataclass(frozen=True)
class HardwareSpec:
    pe_rows: int = 16
    pe_cols: int = 16
    frequency_hz: float = 1e9
    dram_bytes_per_second: float = 32e9
    sram_bytes: int = 262144
    accumulator_bits: int = 32
    # Uniform 1 MAC/PE/cycle is an explicit hypothetical hardware assumption.
    # No automatic INT8-vs-FP32 throughput multiplier.
    pipeline_cycles: int = 0
    overlap_compute_memory: bool = False

    def __post_init__(self):
        for name in ("pe_rows", "pe_cols", "sram_bytes", "accumulator_bits"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("frequency_hz", "dram_bytes_per_second"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.pipeline_cycles) is not int or self.pipeline_cycles < 0:
            raise ValueError("pipeline_cycles must be a nonnegative integer")
        if type(self.overlap_compute_memory) is not bool:
            raise ValueError("overlap_compute_memory must be bool")


@dataclass(frozen=True)
class MappingSpec:
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 64
    dataflow: str = "output_stationary"

    def __post_init__(self):
        if self.dataflow != "output_stationary":
            raise ValueError("Only output_stationary is implemented")
        if any(type(v) is not int or v <= 0 for v in (self.tile_m, self.tile_n, self.tile_k)):
            raise ValueError("Tile sizes must be positive integers")


@dataclass(frozen=True)
class OpEstimate:
    event_id: int
    status: str
    reason: str
    dense_macs: int
    compute_cycles: int | None = None
    dram_read_bytes: int | None = None
    dram_write_bytes: int | None = None
    peak_buffer_bytes: int | None = None
    latency_seconds: float | None = None
    pe_utilization: float | None = None
