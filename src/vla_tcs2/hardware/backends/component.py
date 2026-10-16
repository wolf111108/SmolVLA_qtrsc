"""Adapter from existing capture/trace contracts to the component protocol."""
from dataclasses import asdict
from ..schema import OpEstimate
from ..modeling import (Architecture, ComponentMapper, EventSimulator, MatmulPlanner,
                        RuntimeStats, UnsupportedModel, default_architecture)
from ..modeling.planner import ASSUMPTIONS


class ComponentBackend:
    name = "component_sim"
    version = "1"
    required_features = frozenset({"metadata"})
    assumptions = ASSUMPTIONS

    def __init__(self):
        self.config = {}
        self.architecture = None
        self.search = None
        self.last_record = None
        self.last_mapping = None

    def configure(self, *, architecture=None, double_buffer=None, max_plan_events=200000,
                  max_candidates=256, save_plan=False):
        from ..modeling.protocol import integer
        integer(max_plan_events, "max_plan_events", 1)
        integer(max_candidates, "max_candidates", 1)
        if type(save_plan) is not bool or (double_buffer is not None and type(double_buffer) is not bool):
            raise ValueError("save_plan/double_buffer must be bool")
        self.architecture = Architecture(architecture) if architecture is not None else None
        self.config = dict(architecture=architecture, double_buffer=double_buffer,
                           max_plan_events=max_plan_events, max_candidates=max_candidates, save_plan=save_plan)

    def estimate(self, event, hardware, mapping):
        self.last_mapping = mapping
        self.last_record = None
        if not self.config:
            self.configure()
        architecture = self.architecture or default_architecture(hardware)
        planner = MatmulPlanner(
            double_buffer=hardware.overlap_compute_memory if self.config["double_buffer"] is None
                          else self.config["double_buffer"],
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
            partial_sum_read_bytes=0, partial_sum_write_bytes=0,
            memory_seconds=sum(v for key, v in result.resource_busy_cycles.items()
                               if key.startswith(dram + ".")) / architecture.clock_hz)
