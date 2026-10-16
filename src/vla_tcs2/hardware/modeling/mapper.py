"""Bounded candidate search scored by the SAME component event simulator."""
from dataclasses import dataclass
from itertools import product
from ..schema import MappingSpec
from ..timeloop import _candidates
from .planner import MatmulPlanner
from .simulator import EventSimulator
from .protocol import UnsupportedModel, integer


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
        if tuple(search.dataflows) != ("output_stationary",):
            raise ValueError("Component search v1 requires dataflows=['output_stationary']")
        integer(max_candidates, "max_candidates", 1)
        self.architecture, self.search = architecture, search
        self.planner = planner or MatmulPlanner()
        self.max_candidates = max_candidates

    def map_event(self, op, stats=None):
        if stats is not None and stats.tile_active_macs:
            raise ValueError("Mapping-specific tile statistics require a fixed mapping; use tensor statistics for search")
        sets = [_candidates(getattr(op, axis), getattr(self.search, f"tile_{axis}_candidates"),
                            getattr(self.search, f"alignment_{axis}"), self.search.candidate_policy)
                for axis in ("m", "n", "k")]
        count = len(sets[0]) * len(sets[1]) * len(sets[2])
        if count > self.max_candidates:
            raise UnsupportedModel(f"Search has {count} candidates; max_candidates={self.max_candidates}; use explicit tile lists")
        best, feasible = None, 0
        for m, n, k in product(*sets):
            mapping = MappingSpec(m, n, k)
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
        # Each returned plan/result owns its metadata; no shared run state.
        metadata = dict(plan.metadata, mapping_policy=policy,
                        search={"evaluated_candidates": count, "feasible_candidates": feasible,
                                "objective": self.search.objective, "objective_value": score,
                                "candidate_sets": dict(zip(("m", "n", "k"), sets)),
                                "optimality": "Minimum over feasible declared candidates under component model"})
        from dataclasses import replace
        plan, simulation = replace(plan, metadata=metadata), replace(simulation, metadata=metadata)
        return ComponentMappingResult(mapping, plan, simulation, count, feasible, self.search.objective, score)
