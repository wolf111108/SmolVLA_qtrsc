"""Adapter from captured operators to the component planner and simulator."""
from dataclasses import asdict
from ..schema import OpEstimate
from .architecture import Architecture, default_architecture
from .mapper import ComponentMapper
from .planner import ASSUMPTIONS, MatmulPlanner
from .protocol import UnsupportedModel, integer
from .runtime import RuntimeStats
from .simulator import EventSimulator


class ComponentBackend:
    name = "component_sim"
    version = "2"
    assumptions = ASSUMPTIONS

    def __init__(self):
        self.config = None
        self.architecture = default_architecture()
        self.search = None
        self.last_record = None
        self.last_mapping = None

    def configure(self, *, architecture=None, double_buffer=True, max_plan_events=200000,
                  max_candidates=256, save_plan=False):
        integer(max_plan_events, "max_plan_events", 1)
        integer(max_candidates, "max_candidates", 1)
        if type(save_plan) is not bool or type(double_buffer) is not bool:
            raise ValueError("save_plan/double_buffer must be bool")
        self.architecture = Architecture(architecture) if architecture is not None else default_architecture()
        self.config = dict(architecture=architecture, double_buffer=double_buffer,
                           max_plan_events=max_plan_events, max_candidates=max_candidates, save_plan=save_plan)

    def estimate(self, event, mapping):
        self.last_mapping = mapping
        self.last_record = None
        if not self.config:
            self.configure()
        architecture = self.architecture
        planner = MatmulPlanner(double_buffer=self.config["double_buffer"],
                                max_plan_events=self.config["max_plan_events"])
        data = event.metadata.get("runtime_stats")
        stats = RuntimeStats(**data) if data is not None else None
        try:
            if self.search is not None and self.search.enabled:
                choice = ComponentMapper(architecture, self.search, planner,
                                         self.config["max_candidates"]).map_event(event, stats)
                self.last_mapping, plan, result = choice.mapping, choice.plan, choice.simulation
            else:
                plan = planner.lower(event, mapping, architecture, stats)
                result = EventSimulator().run(plan, architecture)
        except UnsupportedModel as exc:
            self.last_record = {"event_id": event.event_id, "status": "unsupported", "reason": str(exc)}
            return OpEstimate(event.event_id, "unsupported", str(exc), event.macs)
        self.last_record = {"event_id": event.event_id, "status": "estimated",
                            "mapping": asdict(self.last_mapping), "simulation": result.to_dict()}
        if self.config["save_plan"]:
            self.last_record["plan"] = plan.to_dict()
        roles, counts = architecture.roles, result.counters
        dram, array = roles["dram"], roles["array"]
        stream_bytes = {"activation_read_bytes": 0, "weight_read_bytes": 0,
                        "partial_sum_read_bytes": 0, "partial_sum_write_bytes": 0}
        flow = self.last_mapping.dataflow
        stream_kind = {
            "load_a": "activation_read_bytes", "load_b": "weight_read_bytes",
            "load_stationary": "weight_read_bytes" if flow == "weight_stationary" else "activation_read_bytes",
            "load_streamed": "activation_read_bytes" if flow == "weight_stationary" else "weight_read_bytes",
            "restore_psum": "partial_sum_read_bytes", "store_psum": "partial_sum_write_bytes",
        }
        for tile_event in plan.events:
            kind = tile_event.id.rsplit("/", 1)[-1]
            for action in tile_event.actions:
                if action.target != dram:
                    continue
                physical = architecture.components[dram].lower(action).counters[f"{dram}.{action.kind}_bytes"]
                if kind in stream_kind:
                    stream_bytes[stream_kind[kind]] += physical
        cycles = counts.get(f"{array}.compute_cycles", 0)
        executed = counts.get(f"{array}.executed_macs", 0)
        spec = architecture.components[array].spec
        return OpEstimate(
            event_id=event.event_id, status="estimated", reason="Scheduled GEMM core; see simulations.jsonl assumptions",
            dense_macs=event.macs, compute_cycles=cycles,
            dram_read_bytes=counts.get(f"{dram}.read_bytes", 0),
            dram_write_bytes=counts.get(f"{dram}.write_bytes", 0),
            peak_buffer_bytes=sum(plan.reservations.values()), latency_seconds=result.latency_seconds,
            pe_utilization=executed / (cycles * spec.rows * spec.cols) if cycles else 0.0,
            **stream_bytes,
            memory_seconds=sum(v for key, v in result.resource_busy_cycles.items()
                               if key.startswith(dram + ".")) / architecture.clock_hz)
