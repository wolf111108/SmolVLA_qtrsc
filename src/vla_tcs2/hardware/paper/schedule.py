"""Incremental, gap-filling transaction scheduler using common Action lowering.

Deterministic planner order is the issue priority. Interval calendars allow a
ready transaction to use gaps before previously planned future transactions.
Calendars are pruned at operator barriers; no full-inference DAG in RAM.
"""

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
import csv
import gzip
import json
from math import ceil

from ..modeling.protocol import Action, PlanEvent


@dataclass
class Completion:
    id: int = -1
    end: int = 0
    critical: dict = field(default_factory=dict)


class TimelineSink:
    def __init__(self, root, clock_hz):
        root.mkdir(parents=True, exist_ok=True)
        self.events = gzip.open(root / "events.csv.gz", "wt", newline="")
        self.writer = csv.DictWriter(
            self.events,
            fieldnames=[
                "id",
                "kind",
                "category",
                "ready",
                "start",
                "end",
                "wait",
                "resources",
                "depends_on",
                "critical_predecessor",
                "blockers",
                "metadata",
                "actions",
            ],
        )
        self.writer.writeheader()
        self.trace = gzip.open(root / "chrome_trace.json.gz", "wt")
        self.trace.write('{"traceEvents":[')
        self.first = True
        self.clock = clock_hz
        self.tids = {}

    def emit(self, item):
        if not self.first:
            self.trace.write(",")
        self.first = False
        self.trace.write(json.dumps(item, separators=(",", ":")))

    def write(self, row):
        self.writer.writerow(
            {
                k: (
                    json.dumps(v, separators=(",", ":"))
                    if isinstance(v, (list, dict, tuple))
                    else v
                )
                for k, v in row.items()
            }
        )
        for resource in row["resources"] or ["control"]:
            if resource not in self.tids:
                self.tids[resource] = len(self.tids) + 1
                self.emit(
                    dict(
                        name="thread_name",
                        ph="M",
                        pid=1,
                        tid=self.tids[resource],
                        args={"name": resource},
                    )
                )
            item = dict(
                name=row["kind"],
                ph="X",
                pid=1,
                tid=self.tids[resource],
                ts=row["start"] * 1e6 / self.clock,
                dur=(row["end"] - row["start"]) * 1e6 / self.clock,
                args={
                    **row["metadata"],
                    "event_id": row["id"],
                    "wait_cycles": row["wait"],
                    "critical_predecessor": row["critical_predecessor"],
                    "blockers": row["blockers"],
                },
            )
            self.emit(item)

    def close(self):
        self.events.close()
        self.trace.write("]}")
        self.trace.close()


class StreamingEventSimulator:
    def __init__(self, architecture, sink=None):
        self.architecture, self.sink = architecture, sink
        self.calendars = {r: [] for r in architecture.resources}
        self.serial = 0
        self.busy = Counter()
        self.service = Counter()
        self.waits = Counter()
        self.counters = Counter()
        self.context = {}
        self.floor = Completion()
        self.last = Completion()

    def issue(self, kind, category, actions=(), deps=(), metadata=None):
        ds = [self.floor, *deps]
        ready_token = max(ds, key=lambda t: t.end)
        ready = start = ready_token.end
        pred = ready_token
        blockers = {}
        event = PlanEvent(str(self.serial), tuple(actions))
        if actions:
            model = self.architecture.lower_event(event)
            if len(model.phases) != 1:
                raise ValueError("Split pipelined actions into single-stage events")
            phase = model.phases[0]
            demands = {d.resource: d.amount for d in phase.demands}
            duration = phase.fixed_cycles + max(
                (
                    ceil(n / self.architecture.resources[r].units_per_cycle)
                    for r, n in demands.items()
                ),
                default=0,
            )
            self.counters.update(model.counters)
        else:
            demands, duration = {}, 0
        resources = tuple(demands)
        if duration:
            while True:
                bumped = False
                for r in resources:
                    intervals = self.calendars[r]
                    # Calendars are ordered and never overlapping.
                    idx = bisect_left([x[0] for x in intervals], start)
                    if idx and intervals[idx - 1][1].end > start:
                        idx -= 1
                    while idx < len(intervals) and intervals[idx][0] < start + duration:
                        _, token = intervals[idx]
                        if token.end > start:
                            start, pred = token.end, token
                            blockers[r] = token.id
                            bumped = True
                            break
                        idx += 1
                if not bumped:
                    break
        critical = Counter(pred.critical)
        critical[category] += duration
        if start > pred.end:
            raise AssertionError("Unexplained scheduler gap")
        token = Completion(self.serial, start + duration, dict(critical))
        for r in resources:
            intervals = self.calendars[r]
            idx = bisect_left([x[0] for x in intervals], start)
            if duration:
                intervals.insert(idx, (start, token))
            self.busy[r] += duration
            self.service[r] += ceil(
                demands[r] / self.architecture.resources[r].units_per_cycle
            )
        self.waits[category] += start - ready
        if self.sink:
            self.sink.write(
                dict(
                    id=self.serial,
                    kind=kind,
                    category=category,
                    ready=ready,
                    start=start,
                    end=token.end,
                    wait=start - ready,
                    resources=resources,
                    depends_on=[t.id for t in ds if t.id >= 0],
                    critical_predecessor=pred.id,
                    blockers=blockers,
                    metadata={**self.context, **(metadata or {})},
                    actions=[
                        dict(target=a.target, kind=a.kind, params=dict(a.params))
                        for a in actions
                    ],
                )
            )
        self.serial += 1
        if token.end >= self.last.end:
            self.last = token
        return token

    def barrier(self):
        self.floor = self.last
        for r in self.calendars:
            self.calendars[r] = [
                (s, t) for s, t in self.calendars[r] if t.end > self.floor.end
            ]
        return self.floor


def work(target, kind, **amounts):
    return Action(target, kind, {"work": amounts})
