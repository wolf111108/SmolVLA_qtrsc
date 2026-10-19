"""Paper facts and configurable system assumptions, lowered via common Actions."""

from dataclasses import dataclass, asdict
from math import ceil
from ..modeling.protocol import (
    ActionModel,
    Phase,
    PortSpec,
    ResourceDemand,
    ResourceSpec,
)
from ..modeling.components import register_component
from ..modeling.architecture import Architecture


@dataclass(frozen=True)
class PaperConfig:
    clock_hz: float = 390e6
    dram_bytes_per_second: float = 12e9
    dram_latency_ns: float = 100
    dram_burst_bytes: int = 64
    dma_outstanding: int = 8  # multi-burst tile requests, NOT eight 64-byte bursts
    input_bytes: int = 16 * 1024
    weight_bytes: int = 128 * 1024
    metadata_bytes: int = 8 * 1024
    accumulator_bytes: int = 2048 * 4
    input_bytes_per_cycle: int = 32
    weight_bytes_per_cycle: int = 128
    metadata_bytes_per_cycle: int = 16
    auxiliary_elements_per_cycle: int = 32
    auxiliary_special_latency: int = 8
    auxiliary_scale: float = 1.0
    memory_block_bytes: int = 1024
    k_tile: int = 128
    t: int = 16
    sss_threshold: float = 2 / 3
    group: str = "D3"
    ideal_memory: bool = False
    ideal_bsc: bool = False

    def __post_init__(self):
        if self.group not in {"D0", "D1", "D2", "D3"}:
            raise ValueError("Unknown ablation group")
        for k, v in asdict(self).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v <= 0:
                raise ValueError(f"{k} must be positive")
        if self.metadata_bytes < 2048 or self.memory_block_bytes > min(
            self.input_bytes // 2, self.weight_bytes // 2, self.metadata_bytes - 1024
        ):
            raise ValueError("Block and metadata capacities are incompatible")
        if not 0 < self.sss_threshold <= 1:
            raise ValueError("Invalid SSS threshold")
        if self.t != 16 or self.accumulator_bytes != 8192 or self.k_tile % 16:
            raise ValueError(
                "Base architecture requires t16, 2048 INT32 accumulators, aligned K tiles"
            )
        if self.k_tile * 64 > self.input_bytes // 2:
            raise ValueError("Dense input tile exceeds input ping-pong bank")
        if self.k_tile * 256 > self.weight_bytes // 2:
            raise ValueError("A W8 weight tile exceeds weight ping-pong bank")


class PaperUnit:
    """Named pipeline stages with finite bandwidth, sharing the existing protocol."""

    def __init__(self, name, rates):
        self.id, self.rates = name, rates

    def ports(self):
        return (PortSpec(),)

    def resources(self):
        return tuple(
            ResourceSpec(f"{self.id}.{k}", unit, rate)
            for k, (unit, rate) in self.rates.items()
        )

    def validate(self, action):
        for key, amount in action.params.get("work", {}).items():
            if key not in self.rates or type(amount) is not int or amount < 0:
                raise ValueError(f"Invalid {self.id} work: {key}={amount}")

    def lower(self, action):
        self.validate(action)
        ds = tuple(
            ResourceDemand(f"{self.id}.{k}", n, self.rates[k][0])
            for k, n in action.params.get("work", {}).items()
            if n
        )
        return ActionModel(
            (Phase(ds, action.params.get("fixed_cycles", 0)),),
            {
                f"{self.id}.{action.kind}.{k}": n
                for k, n in action.params.get("work", {}).items()
            },
        )


register_component(
    "paper_unit", lambda name, params, clock: PaperUnit(name, params["rates"])
)


def paper_architecture(c):
    def unit(**rates):
        return dict(type="paper_unit", rates=rates)

    components = {
        "dma": unit(**{f"request{i}": ("cycles", 1) for i in range(c.dma_outstanding)}),
        "dram": unit(bus=("bytes", c.dram_bytes_per_second / c.clock_hz)),
        "input": unit(
            read=("bytes", c.input_bytes_per_cycle),
            write=("bytes", c.input_bytes_per_cycle),
            bank0=("cycles", 1),
            bank1=("cycles", 1),
        ),
        "weight": unit(
            read=("bytes", c.weight_bytes_per_cycle),
            write=("bytes", c.weight_bytes_per_cycle),
            bank0=("cycles", 1),
            bank1=("cycles", 1),
        ),
        "metadata": unit(
            read=("bytes", c.metadata_bytes_per_cycle),
            write=("bytes", c.metadata_bytes_per_cycle),
        ),
        "bsc0": unit(
            load=("cycles", 1),
            detect=("cycles", 1),
            compact=("cycles", 1),
            emit=("cycles", 1),
        ),
        "bsc1": unit(
            load=("cycles", 1),
            detect=("cycles", 1),
            compact=("cycles", 1),
            emit=("cycles", 1),
        ),
        "array": unit(issue=("cycles", 1)),
        "sss": unit(control=("cycles", 1)),
        "iag": unit(score=("cycles", 1)),
        "aux": unit(vector=("cycles", 1)),
        "acc": unit(read=("bytes", 128), write=("bytes", 128)),
    }
    return Architecture(
        dict(
            schema_version=1,
            clock_hz=c.clock_hz,
            components=components,
            roles={"array": "array", "dram": "dram"},
            connections=[],
        )
    )
