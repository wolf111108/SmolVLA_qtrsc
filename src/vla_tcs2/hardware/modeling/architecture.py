"""Build, validate and connect immutable component configurations."""
from copy import deepcopy
from collections import Counter
from .components import create_component
from .protocol import Action, ActionModel, Phase, ResourceDemand, positive, integer, PROTOCOL_VERSION


class Architecture:
    def __init__(self, config):
        config = deepcopy(config)
        if config.get("schema_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported architecture schema_version")
        if set(config) != {"schema_version", "clock_hz", "components", "connections", "roles"}:
            raise ValueError("Invalid architecture fields")
        positive(config["clock_hz"], "clock_hz")
        self.clock_hz = config["clock_hz"]
        self.config = config
        self.components = {}
        self.resources = {}
        for name, spec in config["components"].items():
            if not name or "." in name:
                raise ValueError("Component IDs must be nonempty and contain no dots")
            component = create_component(name, spec, self.clock_hz)
            if component.id != name:
                raise ValueError("Factory returned a mismatched component ID")
            self.components[name] = component
            for resource in component.resources():
                if resource.id in self.resources:
                    raise ValueError(f"Duplicate physical resource: {resource.id}")
                self.resources[resource.id] = resource
        self.roles = dict(config["roles"])
        for role, name in self.roles.items():
            if name not in self.components:
                raise ValueError(f"Unknown role target: {role}={name}")
        self.connections = {}
        for connection in config["connections"]:
            if set(connection) != {"from", "to", "via"}:
                raise ValueError("Connections require from/to/via")
            source, destination, via = (connection[k] for k in ("from", "to", "via"))
            src = self._port(source)
            dst = self._port(destination)
            if src.direction not in {"out", "bidirectional"} or dst.direction not in {"in", "bidirectional"}:
                raise ValueError("Incompatible port directions")
            if ("*" not in src.supported_formats and "*" not in dst.supported_formats
                    and not set(src.supported_formats).intersection(dst.supported_formats)):
                raise ValueError("Incompatible port formats")
            if via not in self.components:
                raise ValueError(f"Unknown link: {via}")
            self.components[via].validate(Action(via, "transfer", {"bytes": 0}))
            if (source, destination) in self.connections:
                raise ValueError("Duplicate directed connection")
            self.connections[source, destination] = via

    def _port(self, endpoint):
        if not isinstance(endpoint, str) or endpoint.count(".") != 1:
            raise ValueError(f"Expected component.port: {endpoint}")
        name, port = endpoint.split(".")
        if name not in self.components:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        matches = [p for p in self.components[name].ports() if p.name == port]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous port: {endpoint}")
        return matches[0]

    def link_action(self, source, destination, size):
        route = (f"{source}.data", f"{destination}.data")
        if route not in self.connections:
            raise ValueError(f"No route from {route[0]} to {route[1]}")
        return Action(self.connections[route], "transfer", {"bytes": size})

    def transfer(self, source, destination, size, *, destination_bytes=None):
        """Stream one tile; optional width conversion has zero arithmetic cost in v1."""
        src, dst = f"{source}.data", f"{destination}.data"
        if (src, dst) not in self.connections:
            raise ValueError(f"No route from {src} to {dst}")
        via = self.connections[src, dst]
        read = Action(source, "read", {"bytes": size})
        model = self.components[source].lower(read)
        write = Action(destination, "write", {"bytes": size if destination_bytes is None else destination_bytes})
        written = self.components[destination].lower(write).counters[f"{destination}.write_bytes"]
        physical = max(model.counters[f"{source}.read_bytes"], written)
        return (read, Action(via, "transfer", {"bytes": physical}), write)

    def lower_event(self, event):
        if not event.actions:
            raise ValueError("Plan events require at least one action")
        models = []
        for action in event.actions:
            if action.target not in self.components:
                raise ValueError(f"Unknown action target: {action.target}")
            models.append(self.components[action.target].lower(action))
        counters = Counter()
        for model in models:
            for name, value in model.counters.items():
                integer(value, name)
            counters.update(model.counters)
        if len(models) == 1:
            phases = models[0].phases
        else:
            if any(len(m.phases) != 1 for m in models):
                raise ValueError("Coupled transactions require single-phase participants")
            phases = (Phase(tuple(d for m in models for d in m.phases[0].demands),
                            sum(m.phases[0].fixed_cycles for m in models)),)
        if not phases:
            raise ValueError("An action model must contain at least one phase")
        # Sum demands on the same physical resource before computing service time.
        merged = []
        for phase in phases:
            amounts = Counter()
            for demand in phase.demands:
                resource = self.resources.get(demand.resource)
                if resource is None or resource.unit != demand.unit:
                    raise ValueError(f"Unknown resource or unit mismatch: {demand}")
                amounts[demand.resource] += demand.amount
            merged.append(Phase(tuple(ResourceDemand(k, v, self.resources[k].unit)
                                      for k, v in sorted(amounts.items()) if v), phase.fixed_cycles))
        return ActionModel(tuple(merged), dict(counters))

    def validate_reservations(self, reservations):
        for name, size in reservations.items():
            integer(size, "reservation")
            if name not in self.components or self.config["components"][name]["type"] == "dram":
                raise ValueError(f"Invalid on-chip reservation: {name}")
            capacity = getattr(self.components[name].spec, "capacity_bytes", None)
            if capacity is None:
                raise ValueError(f"{name} is not a capacity-bearing component")
            if size > capacity:
                from .protocol import UnsupportedModel
                raise UnsupportedModel(f"{name} needs {size} bytes; capacity is {capacity}")


def default_architecture():
    """Explicit baseline for direct API calls; production runs should configure components."""
    memory = {"type": "sram", "capacity_bytes": 4 * 1024**2,
              "read_bytes_per_cycle": 256, "write_bytes_per_cycle": 256}
    return Architecture({
        "schema_version": 1, "clock_hz": 1e9,
        "components": {
            "dram": {"type": "dram", "bandwidth_bytes_per_second": 102.4e9,
                     "efficiency": 0.7, "latency_cycles": 100, "burst_bytes": 64},
            "dma": {"type": "dma", "bytes_per_cycle": 256},
            "noc": {"type": "link", "bytes_per_cycle": 256},
            "operand_link": {"type": "link", "bytes_per_cycle": 256},
            "accumulator_link": {"type": "link", "bytes_per_cycle": 256},
            "gb": memory,
            "local": {"type": "sram", "capacity_bytes": 128 * 1024,
                      "read_bytes_per_cycle": 256, "write_bytes_per_cycle": 256},
            "acc": {"type": "sram", "capacity_bytes": 64 * 1024,
                    "read_bytes_per_cycle": 256, "write_bytes_per_cycle": 256},
            "array": {"type": "compute_array", "rows": 32, "cols": 32,
                      "pipeline_cycles": 0, "accumulator_bits": 32,
                      "supported_formats": ["int4", "int8", "int12", "int16", "float16", "bfloat16", "float32",
                                            "e2m1", "e4m3", "e4m3fn", "e5m2", "e5m10"]}},
        "roles": {"dram": "dram", "global": "gb", "local": "local", "accumulator": "acc", "array": "array"},
        "connections": [{"from": f"{a}.data", "to": f"{b}.data", "via": link} for a, b, link in
                        [("dram", "gb", "dma"), ("gb", "dram", "dma"),
                         ("gb", "local", "noc"), ("acc", "gb", "noc"),
                         ("gb", "acc", "noc"),
                         ("local", "array", "operand_link"), ("array", "acc", "accumulator_link"),
                         ("acc", "array", "accumulator_link")]]})
