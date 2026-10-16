"""Dependency-free, Timeloop-style single-buffer GEMM mapping search.

This is a project analytic mapper, NOT a wrapper around official Timeloop.
Search is exhaustive over the declared candidate sets, not all possible
accelerator schedules. It uses DenseAnalytic as its single source of costs.
M/N/K correspond to A/C/B and tile_m/tile_n/tile_k to x/z/y in X[A,B] @ W[B,C].
"""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from .backends.dense import DenseAnalytic, packed_bytes
from .schema import HardwareSpec, MappingSpec, OperatorEvent, OpEstimate

DATAFLOWS = ("output_stationary", "weight_stationary", "input_stationary")
LOOP_ORDERS = {
    "output_stationary": ["m", "n", "k"],
    "weight_stationary": ["n", "k", "m"],
    "input_stationary": ["m", "k", "n"],
}


@dataclass(frozen=True)
class SearchSpec:
    enabled: bool = False
    objective: str = "memory_time"
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
        if self.objective not in {"memory_time", "latency", "dram_bytes"}:
            raise ValueError("objective must be memory_time, latency or dram_bytes")
        if self.candidate_policy not in {"powers_of_two", "all"}:
            raise ValueError("candidate_policy must be powers_of_two or all")
        for axis in ("m", "n", "k"):
            alignment = getattr(self, f"alignment_{axis}")
            if type(alignment) is not int or alignment <= 0:
                raise ValueError("Tile alignments must be positive integers")
            name = f"tile_{axis}_candidates"
            values = getattr(self, name)
            if values is not None:
                if (not isinstance(values, (list, tuple)) or not values
                        or any(type(v) is not int or v <= 0 for v in values)):
                    raise ValueError(f"{name} must be a nonempty list of positive integers")
                object.__setattr__(self, name, tuple(sorted(set(values))))
        if (not isinstance(self.dataflows, (list, tuple)) or not self.dataflows
                or any(v not in DATAFLOWS for v in self.dataflows)):
            raise ValueError(f"dataflows must be a nonempty subset of {DATAFLOWS}")
        object.__setattr__(self, "dataflows", tuple(dict.fromkeys(self.dataflows)))


@dataclass(frozen=True)
class MappingResult:
    mapping: MappingSpec | None
    estimate: OpEstimate
    objective: str
    objective_value: float | int | None
    candidate_space: dict
    evaluated_candidates: int
    feasible_candidates: int
    capacity_pruned_candidates: int = 0
    cache_hit: bool = False

    def to_dict(self):
        result = asdict(self)
        result.update(
            engine="project_analytic", engine_version=DenseAnalytic.version,
            loop_order=LOOP_ORDERS[self.mapping.dataflow] if self.mapping else None,
            optimality_scope="Minimum over declared candidate sets under DenseAnalytic costs",
        )
        return result


def _candidates(limit, explicit, alignment, policy):
    if explicit is not None:
        values = explicit
    elif policy == "all":
        values = (*range(alignment, limit + 1, alignment), limit)
    else:
        values = [1 << i for i in range(limit.bit_length())]
        # Include full dimension and aligned powers for non-power-of-two arrays.
        values += [((v + alignment - 1) // alignment) * alignment for v in values]
        values.append(limit)
    return tuple(sorted({v for v in values if v <= limit and (v % alignment == 0 or v == limit)}))


def _score(estimate, objective):
    if objective == "memory_time":
        return estimate.memory_seconds
    if objective == "latency":
        return estimate.latency_seconds
    return estimate.dram_read_bytes + estimate.dram_write_bytes


class TimeloopMapper:
    """Search once per workload/hardware signature; rebind cached event IDs.

    Standalone usage::

        mapper = TimeloopMapper(HardwareSpec(sram_bytes=1048576))
        result = mapper.map_gemm(256, 1024, 4096, activation_bits=16,
                                 weight_bits=16, output_bits=32,
                                 save_path="mapping.json")

    map_event() also accepts captured Linear and batched MatMul events. Each
    broadcast batch is modeled independently; no reuse across batches or calls.
    """

    def __init__(self, hardware=None, search=None):
        self.hardware = hardware if isinstance(hardware, HardwareSpec) else HardwareSpec(**(hardware or {}))
        self.search = search if isinstance(search, SearchSpec) else SearchSpec(**(search or {}))
        self.backend = DenseAnalytic()
        self._cache = {}
        self.cache_hits = 0

    def map_event(self, event):
        signature = event.to_dict()
        for name in ("event_id", "module_id", "component", "phase", "flow_step", "generation_id",
                     "attention_kind", "weight_id"):
            signature.pop(name)
        key = json.dumps([signature, asdict(self.hardware), asdict(self.search), self.backend.version],
                         sort_keys=True, allow_nan=False)
        if key in self._cache:
            self.cache_hits += 1
            cached = self._cache[key]
            return replace(cached, estimate=replace(cached.estimate, event_id=event.event_id), cache_hit=True)

        axes = {axis: _candidates(getattr(event, axis), getattr(self.search, f"tile_{axis}_candidates"),
                                  getattr(self.search, f"alignment_{axis}"), self.search.candidate_policy)
                for axis in ("m", "n", "k")}
        space = dict(policy=self.search.candidate_policy, tiles=axes, dataflows=self.search.dataflows,
                     alignments={axis: getattr(self.search, f"alignment_{axis}") for axis in axes},
                     full_dimension_tail_allowed=True)
        # Reject unsupported numerical paths before searching thousands of tiles.
        probe = self.backend.estimate(event, replace(self.hardware, sram_bytes=max(self.hardware.sram_bytes, 1024)),
                                      MappingSpec(1, 1, 1))
        if probe.status == "unsupported":
            result = MappingResult(None, probe, self.search.objective, None, space, 0, 0)
            self._cache[key] = result
            return result

        best = rank = None
        evaluated = feasible = pruned = 0
        for m in axes["m"]:
            for n in axes["n"]:
                for k in axes["k"]:
                    # All three flows have the same tile footprint. Prune
                    # capacity violations before the more expensive cost model.
                    copies = 2 if self.hardware.overlap_compute_memory else 1
                    peak = (packed_bytes(m * n, self.hardware.accumulator_bits)
                            + copies * (packed_bytes(m * k, event.a.bits)
                                        + packed_bytes(k * n, event.b.bits)))
                    if peak > self.hardware.sram_bytes:
                        pruned += len(self.search.dataflows)
                        continue
                    for flow in self.search.dataflows:
                        mapping = MappingSpec(m, n, k, flow)
                        estimate = self.backend.estimate(event, self.hardware, mapping)
                        evaluated += 1
                        if estimate.status != "estimated":
                            continue
                        feasible += 1
                        cost = _score(estimate, self.search.objective)
                        # Deterministic ties prefer lower latency, then traffic,
                        # smaller buffer footprint and stable mapping order.
                        candidate_rank = (cost, estimate.latency_seconds,
                                          estimate.dram_read_bytes + estimate.dram_write_bytes,
                                          estimate.peak_buffer_bytes, DATAFLOWS.index(flow), m, n, k)
                        if rank is None or candidate_rank < rank:
                            rank = candidate_rank
                            best = (mapping, estimate, cost)
        if best is None:
            estimate = OpEstimate(event.event_id, "unsupported",
                                  "No feasible mapping in declared candidate sets (SRAM/alignment constraints)", event.macs)
            result = MappingResult(None, estimate, self.search.objective, None, space, evaluated, feasible, pruned)
        else:
            result = MappingResult(best[0], best[1], self.search.objective, best[2], space, evaluated, feasible, pruned)
        self._cache[key] = result
        return result

    def map_gemm(self, a, b, c, *, activation_bits=16, weight_bits=16, output_bits=16,
                 module_id="gemm", save_path=None):
        """Map activation[A,B] @ weight[B,C]; persist inputs and result if requested."""
        from .schema import TensorDesc
        event = OperatorEvent(0, module_id, "unknown", "linear",
                              TensorDesc((a, b), "logical", "logical", activation_bits),
                              TensorDesc((c, b), "logical", "logical", weight_bits),
                              TensorDesc((a, c), "logical", "logical", output_bits), a, c, b)
        result = self.map_event(event)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                json.dump(dict(event=event.to_dict(), hardware=asdict(self.hardware),
                               mapping_search=asdict(self.search), **result.to_dict()), stream, indent=2)
                stream.write("\n")
        return result
