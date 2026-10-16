"""Built-in factories. Plugins use register_component without editing the planner."""
from .storage import Storage, StorageSpec
from .dram import DRAM, DRAMSpec
from .interconnect import Link, LinkSpec
from .compute import ComputeArray, ArraySpec

_FACTORIES = {
    "sram": lambda name, params, clock: Storage(name, StorageSpec(**params)),
    "dram": lambda name, params, clock: DRAM(name, DRAMSpec(**params), clock),
    "link": lambda name, params, clock: Link(name, LinkSpec(**params)),
    "dma": lambda name, params, clock: Link(name, LinkSpec(**params)),
    "compute_array": lambda name, params, clock: ComputeArray(name, ArraySpec(**params)),
}


def register_component(name, factory):
    if not name or name in _FACTORIES:
        raise ValueError(f"Empty or duplicate component type: {name}")
    _FACTORIES[name] = factory


def create_component(name, config, clock_hz):
    params = dict(config)
    kind = params.pop("type")
    if kind not in _FACTORIES:
        raise ValueError(f"Unknown component type: {kind}")
    return _FACTORIES[kind](name, params, clock_hz)
