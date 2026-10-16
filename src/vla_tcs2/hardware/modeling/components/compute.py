"""Dense tiled array and explicitly ideal pooled zero-MAC skipping model."""
from dataclasses import dataclass
from math import ceil
from ..protocol import (ActionModel, Phase, PortSpec, ResourceDemand,
                        ResourceSpec, UnsupportedModel, integer)


@dataclass(frozen=True)
class ArraySpec:
    rows: int = 32
    cols: int = 32
    pipeline_cycles: int = 0
    accumulator_bits: int = 32
    supported_formats: tuple[str, ...] = ("int8",)
    mode: str = "dense"

    def __post_init__(self):
        for name in ("rows", "cols", "accumulator_bits"):
            integer(getattr(self, name), name, 1)
        integer(self.pipeline_cycles, "pipeline_cycles")
        if self.mode not in {"dense", "ideal_mac_skip"}:
            raise ValueError("mode must be dense or ideal_mac_skip")
        if not self.supported_formats or any(not isinstance(x, str) or not x for x in self.supported_formats):
            raise ValueError("supported_formats must contain format names")
        object.__setattr__(self, "supported_formats", tuple(self.supported_formats))


class ComputeArray:
    def __init__(self, component_id, spec):
        self.id, self.spec = component_id, spec

    def ports(self):
        return (PortSpec(supported_formats=self.spec.supported_formats),)

    def resources(self):
        return (ResourceSpec(f"{self.id}.issue", "cycles", 1),)

    def validate(self, action):
        if action.target != self.id or action.kind != "matmul":
            raise ValueError(f"Invalid array action: {action}")
        if set(action.params) != {"m", "n", "k", "a_format", "b_format", "active_macs"}:
            raise ValueError("Invalid matmul action fields")
        p = action.params
        for key in ("m", "n", "k"):
            integer(p[key], key, 1)
        for key in ("a_format", "b_format"):
            if p[key] not in self.spec.supported_formats:
                raise UnsupportedModel(f"{self.id} does not support {p[key]}")
        if p["active_macs"] is not None:
            integer(p["active_macs"], "active_macs")
            if p["active_macs"] > p["m"] * p["n"] * p["k"]:
                raise ValueError("active_macs exceeds dense MACs")
        if self.spec.mode == "ideal_mac_skip" and p["active_macs"] is None:
            raise UnsupportedModel("ideal_mac_skip requires explicit runtime statistics")

    def lower(self, action):
        self.validate(action)
        p, s = action.params, self.spec
        dense = p["m"] * p["n"] * p["k"]
        executed = dense if s.mode == "dense" else p["active_macs"]
        if s.mode == "dense":
            cycles = ceil(p["m"] / s.rows) * ceil(p["n"] / s.cols) * (p["k"] + s.pipeline_cycles)
        else:
            cycles = ceil(executed / (s.rows * s.cols)) + (s.pipeline_cycles if executed else 0)
        return ActionModel((Phase((ResourceDemand(f"{self.id}.issue", cycles, "cycles"),)),),
                           {f"{self.id}.dense_macs": dense, f"{self.id}.executed_macs": executed,
                            f"{self.id}.compute_cycles": cycles})
