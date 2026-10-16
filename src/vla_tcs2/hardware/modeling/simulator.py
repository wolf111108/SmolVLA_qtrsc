"""Deterministic ready-event DES; exclusive nonpreemptive resources, atomic acquire.

Fixed latency is charged while all phase resources are reserved. This is a
conservative transaction model, not flit/DRAM-command or fluid bandwidth timing.
"""
from collections import Counter, deque
import heapq
from math import ceil
from .protocol import EventTiming, SimulationResult, PROTOCOL_VERSION


class EventSimulator:
    def run(self, plan, architecture):
        if plan.schema_version != PROTOCOL_VERSION:
            raise ValueError("Unsupported plan version")
        architecture.validate_reservations(plan.reservations)
        events = {e.id: e for e in plan.events}
        if len(events) != len(plan.events) or any(not e.id for e in plan.events):
            raise ValueError("Event IDs must be nonempty and unique")
        dependents = {key: [] for key in events}
        counts = {}
        for e in plan.events:
            if len(set(e.depends_on)) != len(e.depends_on):
                raise ValueError("Duplicate dependency")
            counts[e.id] = len(e.depends_on)
            for dep in e.depends_on:
                if dep not in events:
                    raise ValueError(f"Missing dependency: {dep}")
                dependents[dep].append(e.id)
        # Reject cycles before executing any action.
        check = dict(counts)
        queue = deque(key for key in events if not check[key])
        visited = 0
        while queue:
            key = queue.popleft()
            visited += 1
            for child in dependents[key]:
                check[child] -= 1
                if not check[child]:
                    queue.append(child)
        if visited != len(events):
            raise ValueError("Execution plan contains a dependency cycle")
        models = {key: architecture.lower_event(event) for key, event in events.items()}
        counters = Counter()
        for model in models.values():
            counters.update(model.counters)
        ordinal = {e.id: i for i, e in enumerate(plan.events)}
        ready = {key: (0, 0) for key in events if not counts[key]}
        running, locked, timeline = [], set(), []
        busy = Counter({key: 0 for key in architecture.resources})
        now = completed = serial = 0
        while completed < len(events):
            # Every completion at a timestamp releases resources before issue.
            while running and running[0][0] == now:
                _, _, key, phase_index, resources = heapq.heappop(running)
                locked.difference_update(resources)
                if phase_index + 1 < len(models[key].phases):
                    ready[key] = (phase_index + 1, now)
                else:
                    completed += 1
                    for child in dependents[key]:
                        counts[child] -= 1
                        if not counts[child]:
                            ready[child] = (0, now)
            for key in sorted(tuple(ready), key=ordinal.__getitem__):
                phase_index, ready_at = ready[key]
                phase = models[key].phases[phase_index]
                resources = tuple(d.resource for d in phase.demands)
                if locked.intersection(resources):
                    continue
                duration = phase.fixed_cycles + max(
                    (ceil(d.amount / architecture.resources[d.resource].units_per_cycle)
                     for d in phase.demands), default=0)
                del ready[key]
                locked.update(resources)
                for resource in resources:
                    busy[resource] += duration
                timeline.append(EventTiming(key, phase_index, ready_at, now, now + duration, resources))
                heapq.heappush(running, (now + duration, serial, key, phase_index, resources))
                serial += 1
            if running:
                now = running[0][0]
            elif completed < len(events):
                raise RuntimeError("No schedulable work remains")
        return SimulationResult(now, now / architecture.clock_hz, dict(counters), dict(busy),
                                dict(plan.reservations), tuple(timeline), dict(plan.metadata))
