"""Version 1 contracts. No tensors, torch, mutable simulator state or I/O."""
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Mapping, Protocol

PROTOCOL_VERSION = 1


def packed_bytes(elements, bits):
    """Bytes needed for a packed logical tile, rounded once per transfer."""
    return (elements * bits + 7) // 8


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


class UnsupportedModel(ValueError):
    """Valid workload/configuration outside the implemented model's scope."""


@dataclass(frozen=True)
class PortSpec:
    name: str = "data"
    direction: str = "bidirectional"
    supported_formats: tuple[str, ...] = ("*",)


@dataclass(frozen=True)
class ResourceSpec:
    id: str
    unit: str
    units_per_cycle: float

    def __post_init__(self):
        if not self.id or not self.unit:
            raise ValueError("Resources require an id and unit")
        positive(self.units_per_cycle, "units_per_cycle")


@dataclass(frozen=True)
class ResourceDemand:
    resource: str
    amount: int
    unit: str

    def __post_init__(self):
        integer(self.amount, "resource demand")


@dataclass(frozen=True)
class Phase:
    demands: tuple[ResourceDemand, ...] = ()
    fixed_cycles: int = 0

    def __post_init__(self):
        integer(self.fixed_cycles, "fixed_cycles")


@dataclass(frozen=True)
class Action:
    target: str
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionModel:
    phases: tuple[Phase, ...]
    counters: Mapping[str, int] = field(default_factory=dict)


class HardwareComponent(Protocol):
    id: str

    def ports(self) -> tuple[PortSpec, ...]: ...
    def resources(self) -> tuple[ResourceSpec, ...]: ...
    def validate(self, action: Action) -> None: ...
    def lower(self, action: Action) -> ActionModel: ...


@dataclass(frozen=True)
class PlanEvent:
    id: str
    # Participants of a single streaming transaction run together. Their
    # demands are combined; each physical resource is occupied only once.
    actions: tuple[Action, ...]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionPlan:
    events: tuple[PlanEvent, ...]
    reservations: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PROTOCOL_VERSION

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported plan schema_version")
        return cls(tuple(PlanEvent(e["id"], tuple(Action(**a) for a in e["actions"]),
                                   tuple(e.get("depends_on", ()))) for e in data["events"]),
                   dict(data.get("reservations", {})), dict(data.get("metadata", {})))


@dataclass(frozen=True)
class EventTiming:
    event_id: str
    phase: int
    ready_cycle: int
    start_cycle: int
    end_cycle: int
    resources: tuple[str, ...]

    @property
    def resource_wait_cycles(self):
        return self.start_cycle - self.ready_cycle


@dataclass(frozen=True)
class SimulationResult:
    total_cycles: int
    latency_seconds: float
    counters: Mapping[str, int]
    resource_busy_cycles: Mapping[str, int]
    reservations: Mapping[str, int]
    timeline: tuple[EventTiming, ...]
    metadata: Mapping[str, Any]
    schema_version: int = PROTOCOL_VERSION

    def to_dict(self):
        result = asdict(self)
        result["resource_utilization"] = {
            k: v / self.total_cycles if self.total_cycles else 0.0
            for k, v in self.resource_busy_cycles.items()}
        result["resource_wait_cycles"] = sum(t.resource_wait_cycles for t in self.timeline)
        return result
