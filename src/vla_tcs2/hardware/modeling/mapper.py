"""One bounded mapping search, scored by the component event simulator."""
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path

from ..schema import MappingSpec, OperatorEvent, TensorDesc
from .planner import MatmulPlanner
from .protocol import UnsupportedModel, integer
from .simulator import EventSimulator

DATAFLOWS = ("output_stationary", "weight_stationary", "input_stationary")


@dataclass(frozen=True)
class SearchSpec:
    enabled: bool = False
    objective: str = "latency"
    candidate_policy: str = "powers_of_two"
    tile_m_candidates: tuple[int, ...] | None = None
    tile_n_candidates: tuple[int, ...] | None = None
    tile_k_candidates: tuple[int, ...] | None = None
    alignment_m: int = 1
    alignment_n: int = 1
    alignment_k: int = 1
    dataflows: tuple[str, ...] = DATAFLOWS

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("mapping_search.enabled must be bool")
        if self.objective not in {"latency", "dram_bytes", "memory_time"}:
            raise ValueError("Invalid mapping_search objective")
        if self.candidate_policy not in {"powers_of_two", "all"}:
            raise ValueError("Invalid mapping_search candidate_policy")
        for axis in ("m", "n", "k"):
            alignment = getattr(self, f"alignment_{axis}")
            integer(alignment, f"alignment_{axis}", 1)
            name = f"tile_{axis}_candidates"
            values = getattr(self, name)
            if values is not None:
                if not isinstance(values, (tuple, list)) or not values:
                    raise ValueError(f"{name} must be a nonempty list")
                for value in values:
                    integer(value, name, 1)
                object.__setattr__(self, name, tuple(sorted(set(values))))
        if not isinstance(self.dataflows, (list, tuple)) or not self.dataflows:
            raise ValueError("dataflows must be a nonempty list")
        if any(flow not in DATAFLOWS for flow in self.dataflows):
            raise ValueError("Unsupported dataflow")
        object.__setattr__(self, "dataflows", tuple(dict.fromkeys(self.dataflows)))


def candidates(limit, explicit, alignment, policy):
    """Enumerate valid tile sizes; an unaligned full dimension is a tail."""
    if explicit is not None:
        values = explicit
    elif policy == "all":
        values = (*range(alignment, limit + 1, alignment), limit)
    else:
        values = [1 << i for i in range(limit.bit_length())]
        values += [((v + alignment - 1) // alignment) * alignment for v in values]
        values.append(limit)
    return tuple(sorted({v for v in values if v <= limit and (v % alignment == 0 or v == limit)}))


@dataclass(frozen=True)
class ComponentMappingResult:
    mapping: MappingSpec
    plan: object
    simulation: object
    evaluated_candidates: int
    feasible_candidates: int
    objective: str
    objective_value: float


class ComponentMapper:
    def __init__(self, architecture, search, planner=None, max_candidates=256):
        integer(max_candidates, "max_candidates", 1)
        self.architecture, self.search = architecture, search
        self.planner = planner or MatmulPlanner()
        self.max_candidates = max_candidates

    def map_event(self, op, stats=None):
        if stats is not None and stats.tile_active_macs:
            raise ValueError("Mapping-specific tile statistics require a fixed mapping; use tensor statistics for search")
        sets = [candidates(getattr(op, axis), getattr(self.search, f"tile_{axis}_candidates"),
                           getattr(self.search, f"alignment_{axis}"), self.search.candidate_policy)
                for axis in ("m", "n", "k")]
        count = len(sets[0]) * len(sets[1]) * len(sets[2]) * len(self.search.dataflows)
        if count > self.max_candidates:
            raise UnsupportedModel(f"Search has {count} candidates; max_candidates={self.max_candidates}; use explicit tile lists")
        best, feasible = None, 0
        for m, n, k, flow in product(*sets, self.search.dataflows):
            mapping = MappingSpec(m, n, k, flow)
            try:
                plan = self.planner.lower(op, mapping, self.architecture, stats)
                simulation = EventSimulator().run(plan, self.architecture)
            except UnsupportedModel:
                continue
            feasible += 1
            dram = self.architecture.roles["dram"]
            if self.search.objective == "latency":
                score = simulation.latency_seconds
            elif self.search.objective == "dram_bytes":
                score = sum(simulation.counters.get(f"{dram}.{kind}_bytes", 0) for kind in ("read", "write"))
            else:
                score = sum(v for key, v in simulation.resource_busy_cycles.items()
                            if key.startswith(dram + ".")) / self.architecture.clock_hz
            if best is None or score < best[0]:
                best = (score, mapping, plan, simulation)
        if best is None:
            raise UnsupportedModel("No feasible component mapping within the declared candidate set and event budget")
        score, mapping, plan, simulation = best
        policy = "oracle_runtime_stats" if stats is not None else "shape_only_search"
        metadata = dict(plan.metadata, mapping_policy=policy,
                        search={"evaluated_candidates": count, "feasible_candidates": feasible,
                                "objective": self.search.objective, "objective_value": score,
                                "candidate_sets": dict(zip(("m", "n", "k"), sets)),
                                "dataflows": self.search.dataflows,
                                "optimality": "Minimum over feasible declared candidates under component model"})
        plan, simulation = replace(plan, metadata=metadata), replace(simulation, metadata=metadata)
        return ComponentMappingResult(mapping, plan, simulation, count, feasible, self.search.objective, score)

    def map_gemm(self, m, k, n, *, activation_bits=16, weight_bits=16, output_bits=16,
                 activation_format="float16", weight_format="float16", output_format="float16",
                 module_id="gemm", save_path=None):
        """Search A[M,K] @ B[K,N] using the same component model as captured events."""
        op = OperatorEvent(0, module_id, "unknown", "linear",
                           TensorDesc((m, k), activation_format, activation_format, activation_bits),
                           TensorDesc((n, k), weight_format, weight_format, weight_bits),
                           TensorDesc((m, n), output_format, output_format, output_bits), m, n, k)
        choice = self.map_event(op)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                json.dump({"operator": op.to_dict(), "architecture": self.architecture.config,
                           "search": asdict(self.search), "mapping": asdict(choice.mapping),
                           "simulation": choice.simulation.to_dict()}, stream, indent=2)
                stream.write("\n")
        return choice
