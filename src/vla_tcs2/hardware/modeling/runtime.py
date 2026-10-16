"""Tensor-free runtime stats; measured tile work or explicit uniform estimates."""
from dataclasses import asdict, dataclass, field
from math import ceil, isfinite
from typing import Mapping
from .protocol import integer, UnsupportedModel


@dataclass(frozen=True)
class RuntimeStats:
    event_id: int
    source: str = "measured"
    scope: str = "tensor"
    sample_count: int = 0
    activation_zero_fraction: float | None = None
    # Key = batch:m_offset:n_offset:k_offset:m_size:n_size:k_size.
    # Shape-qualified keys prevent applying statistics to a different mapping.
    tile_active_macs: Mapping[str, int] = field(default_factory=dict)
    allow_uniform_estimate: bool = False
    schema_version: int = 1

    def __post_init__(self):
        integer(self.event_id, "event_id")
        integer(self.sample_count, "sample_count")
        if self.schema_version != 1 or self.source not in {"measured", "estimated"}:
            raise ValueError("Invalid runtime statistics version/source")
        if self.scope not in {"tensor", "tile"} or type(self.allow_uniform_estimate) is not bool:
            raise ValueError("Invalid runtime statistics scope/policy")
        z = self.activation_zero_fraction
        if z is not None and (isinstance(z, bool) or not isinstance(z, (int, float))
                              or not isfinite(z) or not 0 <= z <= 1):
            raise ValueError("activation_zero_fraction must be in [0, 1]")
        if self.tile_active_macs and self.scope != "tile":
            raise ValueError("Tile work requires scope=tile")
        for key, value in self.tile_active_macs.items():
            if not isinstance(key, str):
                raise ValueError("Tile keys must be strings")
            integer(value, "tile_active_macs")

    def active_macs(self, key, dense):
        if key in self.tile_active_macs:
            work = self.tile_active_macs[key]
            if work > dense:
                raise ValueError("Tile active MACs exceed dense MACs")
            return work, self.source
        if self.allow_uniform_estimate and self.activation_zero_fraction is not None:
            return ceil(dense * (1 - self.activation_zero_fraction)), "estimated_uniform"
        raise UnsupportedModel(f"No compatible tile statistics for {key}; uniform estimation is disabled")

    def to_dict(self):
        return asdict(self)


def capture_raw_activation_zeros(event, module, args, kwargs, output):
    """Optional online provider; synchronizes device, raw mode only.

    Quantized-forward callers must provide statistics of the ACTUAL quantized
    operands; a generic module-input hook cannot infer those from raw tensors.
    """
    if event.mode != "raw":
        raise UnsupportedModel("Raw-input zero provider cannot observe quantized internal operands")
    tensor = args[0] if args else next((kwargs[key] for key in ("input", "x", "A") if key in kwargs), None)
    if tensor is None:
        raise ValueError("Missing activation tensor")
    count = tensor.numel()
    zeros = int((tensor.detach() == 0).sum().item())
    return RuntimeStats(event.event_id, sample_count=count,
                        activation_zero_fraction=zeros / count, allow_uniform_estimate=True)
