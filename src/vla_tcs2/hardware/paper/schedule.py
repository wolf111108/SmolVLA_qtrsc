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


def _freeze(value):
    """Recursively convert nested dicts/lists into hashable tuples."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass
class Completion:
    id: int = -1
    end: int = 0
    critical: dict = field(default_factory=dict)


class TimelineSink:
    """Detailed transaction exporter. If window is set, only transactions whose
    interval intersects [window_start, window_end) cycles are written; counters
    and occupancy bins still cover the whole replay via the simulator."""

    def __init__(self, root, clock_hz, window=None):
        root.mkdir(parents=True, exist_ok=True)
        self.window = window  # (start_cycle, end_cycle) or None for everything
        self.written = 0
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
        if self.window is not None:
            lo, hi = self.window
            if row["end"] <= lo or row["start"] >= hi:
                return
        self.written += 1
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
    def __init__(self, architecture, sink=None, occupancy_bins=1000):
        self.architecture, self.sink = architecture, sink
        # Online occupancy bins over an elastic horizon: when an event exceeds
        # the horizon the bins are re-scaled (doubles amortize to O(1)/event).
        self.bin_count = occupancy_bins
        self._horizon = 1
        self._occ = {r: [0] * occupancy_bins for r in architecture.resources}
        self.calendars = {r: [] for r in architecture.resources}
        # Parallel start lists keep interval lookup O(log n) without rebuilding
        # a key list on every issue() call.
        self.calendar_starts = {r: [] for r in architecture.resources}
        self.serial = 0
        self.busy = Counter()
        self.service = Counter()
        self.waits = Counter()
        self.counters = Counter()
        self.context = {}
        self.floor = Completion()
        self.last = Completion()
        # Repeated identical actions (same target/kind/params) dominate large
        # GEMM replays; cache their lowered demands, duration and counters.
        self._lower_cache = {}

    def issue(self, kind, category, actions=(), deps=(), metadata=None):
        ds = [self.floor, *deps]
        ready_token = max(ds, key=lambda t: t.end)
        ready = start = ready_token.end
        pred = ready_token
        blockers = {}
        if actions:
            if len(actions) == 1:
                a = actions[0]
                cache_key = (a.target, a.kind, _freeze(a.params))
            else:
                cache_key = tuple(
                    (a.target, a.kind, _freeze(a.params)) for a in actions
                )
            cached = self._lower_cache.get(cache_key)
            if cached is None:
                event = PlanEvent(str(self.serial), tuple(actions))
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
                cached = (tuple(demands.items()), duration, tuple(model.counters.items()))
                self._lower_cache[cache_key] = cached
            demands = dict(cached[0])
            duration = cached[1]
            self.counters.update(cached[2])
        else:
            demands, duration = {}, 0
        resources = tuple(demands)
        if duration:
            while True:
                bumped = False
                for r in resources:
                    intervals = self.calendars[r]
                    # Calendars are ordered and never overlapping.
                    idx = bisect_left(self.calendar_starts[r], start)
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
            idx = bisect_left(self.calendar_starts[r], start)
            if duration:
                intervals.insert(idx, (start, token))
                self.calendar_starts[r].insert(idx, start)
                for rr in resources:
                    self._bin_add(rr, start, token.end)
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

    def _bin_add(self, resource, start, end):
        """Integrate [start, end) into the bin array, rescaling on overflow.

        The horizon starts at 1 cycle and doubles when exceeded; rescaling
        merges pairs of adjacent bins so total occupancy is preserved exactly.
        """
        if end > self._horizon:
            while end > self._horizon:
                self._horizon *= 2
            half = self.bin_count // 2
            for bins in self._occ.values():
                merged = [bins[i] + bins[i + half] for i in range(half)]
                bins[:] = merged + [0] * half
        bins = self._occ[resource]
        n, h = self.bin_count, self._horizon
        b0 = start * n // h
        if b0 >= n:
            return
        b1 = end * n // h
        if b1 <= b0:
            b1 = b0 + 1
        width = h // n
        for b in range(b0, min(b1, n)):
            s = b * width if b * width > start else start
            e = (b + 1) * width if (b + 1) * width < end else end
            if e > s:
                bins[b] += e - s

    def occupancy_bins(self):
        """Per-resource occupancy cycles per bin over the final horizon."""
        return {r: list(bins) for r, bins in self._occ.items()}

    def barrier(self):
        self.floor = self.last
        for r in self.calendars:
            kept = [(s, t) for s, t in self.calendars[r] if t.end > self.floor.end]
            self.calendars[r] = kept
            self.calendar_starts[r] = [s for s, _ in kept]
        return self.floor


def work(target, kind, **amounts):
    # Cached factory: identical work actions recur millions of times in large
    # GEMM replays; interning them also speeds up the issue() cache key.
    key = (target, kind, tuple(sorted(amounts.items())))
    action = _WORK_CACHE.get(key)
    if action is None:
        action = Action(target, kind, {"work": dict(amounts)})
        _WORK_CACHE[key] = action
    return action


_WORK_CACHE = {}
