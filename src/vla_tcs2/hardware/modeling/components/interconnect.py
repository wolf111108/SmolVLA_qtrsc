"""One serialized link / DMA engine. No routing, multicast or packet buffers."""
from dataclasses import dataclass
from ..protocol import (ActionModel, Phase, PortSpec, ResourceDemand,
                        ResourceSpec, integer, positive)


@dataclass(frozen=True)
class LinkSpec:
    bytes_per_cycle: float = 256
    latency_cycles: int = 0

    def __post_init__(self):
        positive(self.bytes_per_cycle, "bytes_per_cycle")
        integer(self.latency_cycles, "latency_cycles")


class Link:
    def __init__(self, component_id, spec):
        self.id, self.spec = component_id, spec

    def ports(self):
        return (PortSpec(),)

    def resources(self):
        return (ResourceSpec(f"{self.id}.link", "bytes", self.spec.bytes_per_cycle),)

    def validate(self, action):
        if action.target != self.id or action.kind != "transfer" or set(action.params) != {"bytes"}:
            raise ValueError(f"Invalid link action: {action}")
        integer(action.params["bytes"], "bytes")

    def lower(self, action):
        self.validate(action)
        size = action.params["bytes"]
        return ActionModel((Phase((ResourceDemand(f"{self.id}.link", size, "bytes"),),
                                  self.spec.latency_cycles if size else 0),),
                           {f"{self.id}.transfer_bytes": size})
