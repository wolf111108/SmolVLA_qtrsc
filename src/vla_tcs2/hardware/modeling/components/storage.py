"""Aggregate SRAM model; bank conflicts require a more detailed plugin."""
from dataclasses import dataclass
from ..protocol import (ActionModel, Phase, PortSpec, ResourceDemand,
                        ResourceSpec, integer, positive)


@dataclass(frozen=True)
class StorageSpec:
    capacity_bytes: int = 262144
    read_bytes_per_cycle: float = 256
    write_bytes_per_cycle: float = 256
    shared_read_write: bool = False
    latency_cycles: int = 0
    burst_bytes: int = 1

    def __post_init__(self):
        integer(self.capacity_bytes, "capacity_bytes", 1)
        integer(self.latency_cycles, "latency_cycles")
        integer(self.burst_bytes, "burst_bytes", 1)
        positive(self.read_bytes_per_cycle, "read_bytes_per_cycle")
        positive(self.write_bytes_per_cycle, "write_bytes_per_cycle")
        if type(self.shared_read_write) is not bool:
            raise ValueError("shared_read_write must be bool")
        if self.shared_read_write and self.read_bytes_per_cycle != self.write_bytes_per_cycle:
            raise ValueError("A shared bus must have a single read/write service rate")


class Storage:
    def __init__(self, component_id, spec):
        self.id, self.spec = component_id, spec

    def ports(self):
        return (PortSpec(),)

    def resources(self):
        if self.spec.shared_read_write:
            return (ResourceSpec(f"{self.id}.bus", "bytes", self.spec.read_bytes_per_cycle),)
        return (ResourceSpec(f"{self.id}.read", "bytes", self.spec.read_bytes_per_cycle),
                ResourceSpec(f"{self.id}.write", "bytes", self.spec.write_bytes_per_cycle))

    def validate(self, action):
        if action.target != self.id or action.kind not in {"read", "write"}:
            raise ValueError(f"Invalid storage action: {action}")
        if set(action.params) != {"bytes"}:
            raise ValueError("Storage actions require exactly bytes")
        integer(action.params["bytes"], "bytes")

    def lower(self, action):
        self.validate(action)
        logical = action.params["bytes"]
        physical = ((logical + self.spec.burst_bytes - 1) // self.spec.burst_bytes) * self.spec.burst_bytes
        resource = f"{self.id}.bus" if self.spec.shared_read_write else f"{self.id}.{action.kind}"
        return ActionModel((Phase((ResourceDemand(resource, physical, "bytes"),),
                                  self.spec.latency_cycles if physical else 0),),
                           {f"{self.id}.{action.kind}_bytes": physical,
                            f"{self.id}.{action.kind}_logical_bytes": logical})
