"""A backend owns its model; the manager owns capture, persistence and totals."""
from typing import Protocol
from .schema import HardwareSpec, MappingSpec, OperatorEvent, OpEstimate


class HardwareBackend(Protocol):
    name: str
    version: str
    required_features: frozenset[str]

    def estimate(self, event: OperatorEvent, hardware: HardwareSpec,
                 mapping: MappingSpec) -> OpEstimate: ...
