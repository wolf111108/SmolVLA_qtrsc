"""Offline full-workload replay and phase/EMA/critical-path reports."""

from collections import Counter
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time

from .architecture import PaperConfig, paper_architecture
from .schedule import StreamingEventSimulator, TimelineSink
from .memory import MemorySystem
from .planner import PaperPlanner
from .workload import read_records


def delta(a, b):
    return {
        k: a.get(k, 0) - b.get(k, 0)
        for k in a.keys() | b.keys()
        if a.get(k, 0) != b.get(k, 0)
    }


def validate_workload(root, *, complete=True):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Workload capture did not complete")
    steps = set()
    generations = set()
    components = set()
    count = 0
    order = []
    for rec in read_records(root):
        if rec["id"] != count:
            raise ValueError("Noncontiguous workload records")
        count += 1
        if rec["component"] not in {"vlm", "expert"}:
            raise ValueError("Non-target component in workload")
        expected_phase = "prefill" if rec["component"] == "vlm" else "denoise"
        if rec["phase"] != expected_phase:
            raise ValueError("Component/phase mismatch")
        stage = -1 if rec["component"] == "vlm" else rec["flow_step"]
        if not order or order[-1] != stage:
            order.append(stage)
        if rec["kind"] == "gemm":
            components.add(rec["component"])
            generations.add(rec["generation_id"])
            if rec["phase"] == "denoise":
                steps.add(rec["flow_step"])
            expected_bits = 4 if rec["linear"] and rec["component"] == "expert" else 8
            if (rec["a"]["bits"], rec["b"]["bits"], rec["output_nominal_bits"]) != (
                8,
                expected_bits,
                8,
            ):
                raise ValueError("Workload precision differs from phaseM configuration")
            for role in ("a", "b"):
                if not (Path(root) / rec[role]["file"]).is_file():
                    raise FileNotFoundError(rec[role]["file"])
    if count != manifest["records"]:
        raise ValueError("Incomplete workload file")
    if complete and (
        components != {"vlm", "expert"}
        or len(generations) != 1
        or steps != set(range(10))
    ):
        raise ValueError(
            f"Incomplete inference coverage: {components}, {generations}, {steps}"
        )
    if complete and order != [-1, *range(10)]:
        raise ValueError(
            f"Expected exactly one prefill followed by ten denoise steps: {order}"
        )
    return manifest


def replay(workload, output, config=None, *, complete=True, timeline=True,
           timeline_window=None, progress_every=25):
    root = Path(workload)
    output = Path(output)
    manifest = validate_workload(root, complete=complete)
    if output.exists():
        raise FileExistsError(f"Choose a fresh replay directory: {output}")
    output.mkdir(parents=True)
    c = config or PaperConfig()
    (output / "resolved_hardware.json").write_text(
        json.dumps(asdict(c), indent=2) + "\n"
    )
    sink = (
        TimelineSink(output / "timeline", c.clock_hz, window=timeline_window)
        if timeline
        else None
    )
    s = StreamingEventSimulator(paper_architecture(c), sink)
    mem = MemorySystem(s, c)
    planner = PaperPlanner(s, mem, c, root)
    groups = {}
    operators = []
    start = time.monotonic()
    previous = None
    summary = {
        "status": "running",
        "assumptions": [
            "System assumptions are explicit in resolved_hardware.json; not a cycle-exact RTL replica.",
            "Operator order is serial; within a GEMM dual BSC and DMA overlap. No inter-operator prefetch.",
            "12 GB/s is effective shared read/write bandwidth; outstanding means multi-burst DMA requests.",
            "BG-major microtiles retain identical accumulator coordinates; displaced partial sums spill; mapping-dependent.",
            "Input/weight SRAM capacity is finite; two equal block-interleaved banks with LRU within each bank; metadata uses its own finite store.",
            "Sparse modes skip weight SRAM service and digit MACs, not external contiguous weight fills.",
            "ATen view spans include strided holes; materialized GQA copies are explicitly charged.",
            "Nominal O8 is NOT saturated in phaseL; actual floating interface and overflow audit are preserved.",
            "Auxiliary costs, accumulator width, buffering and DRAM interface are engineering assumptions.",
            "IAG D3 is score-only; no token/head removal or claimed IAG acceleration.",
        ],
    }
    try:
        for rec in read_records(root):
            group = (
                "vlm_prefill"
                if rec["component"] == "vlm"
                else f'expert_step_{rec["flow_step"]}'
            )
            if previous is not None and previous != group:
                # Handoff completion includes writes; residency itself persists.
                before = s.floor
                cb = Counter(mem.counts)
                bb = Counter(s.busy)
                mem.flush()
                mem.prune()
                g = groups[previous]
                g["cycles"] += s.floor.end - before.end
                g["ema"].update(delta(mem.counts, cb))
                g["busy"].update(delta(s.busy, bb))
                g["critical"].update(delta(s.floor.critical, before.critical))
            previous = group
            s.context = {
                k: rec[k]
                for k in (
                    "id",
                    "module_id",
                    "component",
                    "phase",
                    "flow_step",
                    "generation_id",
                )
            }
            s.context["operator_id"] = s.context.pop("id")
            s.context["attention_kind"] = rec.get("attention_kind", "unknown")
            before = s.floor
            cb = Counter(mem.counts)
            bb = Counter(s.busy)
            wb = Counter(s.waits)
            pb = Counter(planner.stats)
            if rec["kind"] == "gemm":
                planner.gemm(rec)
            elif rec["kind"] == "aux":
                planner.auxiliary(rec)
            else:
                raise ValueError(f'Unknown record {rec["kind"]}')
            s.barrier()
            mem.prune()
            latency = s.floor.end - before.end
            critical = delta(s.floor.critical, before.critical)
            if sum(critical.values()) != latency:
                raise AssertionError("Critical path does not reconcile")
            g = groups.setdefault(
                group,
                dict(
                    cycles=0,
                    operators=0,
                    ema=Counter(),
                    critical=Counter(),
                    busy=Counter(),
                    work=Counter(),
                ),
            )
            g["cycles"] += latency
            g["operators"] += 1
            g["ema"].update(delta(mem.counts, cb))
            g["critical"].update(critical)
            g["busy"].update(delta(s.busy, bb))
            g["work"].update(delta(planner.stats, pb))
            row = {
                k: rec[k]
                for k in ("id", "module_id", "component", "phase", "flow_step", "kind")
            }
            row["attention_kind"] = rec.get("attention_kind", "unknown")
            row.update(
                start_cycle=before.end,
                end_cycle=s.floor.end,
                cycles=latency,
                critical=critical,
                ema=delta(mem.counts, cb),
                work=delta(planner.stats, pb),
            )
            operators.append(row)
            rid = rec["id"]
            if rid % progress_every == 0:
                rate = (rid + 1) / (time.monotonic() - start)
                remain = (manifest["records"] - rid - 1) / rate if rate else 0
                print(
                    f'[replay {c.group}] operator {rid}/{manifest["records"]} '
                    f"({rid * 100 // manifest["records"]}%) "
                    f"events={s.serial} elapsed={time.monotonic()-start:.0f}s "
                    f"rate={rate:.1f}op/s eta={remain/60:.0f}m "
                    f"last={rec['module_id'][:40]}",
                    flush=True,
                )
        before = s.floor
        cb = Counter(mem.counts)
        bb = Counter(s.busy)
        mem.flush()
        if previous:
            groups[previous]["cycles"] += s.floor.end - before.end
            groups[previous]["ema"].update(delta(mem.counts, cb))
            groups[previous]["busy"].update(delta(s.busy, bb))
            groups[previous]["critical"].update(
                delta(s.floor.critical, before.critical)
            )
        expert = dict(
            cycles=0,
            operators=0,
            ema=Counter(),
            critical=Counter(),
            busy=Counter(),
            work=Counter(),
        )
        for name, g in groups.items():
            if name.startswith("expert_step_"):
                for field in ("cycles", "operators"):
                    expert[field] += g[field]
                for field in ("ema", "critical", "busy", "work"):
                    expert[field].update(g[field])
        groups["expert_denoise_total"] = expert
        for g in groups.values():
            g["latency_seconds"] = g["cycles"] / c.clock_hz
            g["digit_slot_utilization"] = g["work"].get("digit_mac_slots", 0) / (
                2048 * max(g["cycles"], 1)
            )
            g["nonzero_digit_utilization"] = g["work"].get("nonzero_digit_macs", 0) / (
                2048 * max(g["cycles"], 1)
            )
            g["resource_utilization"] = {
                r: v / max(g["cycles"], 1) for r, v in g["busy"].items()
            }
            if sum(g["critical"].values()) != g["cycles"]:
                raise AssertionError("Phase critical path does not reconcile")
        summary.update(
            status="complete",
            config=asdict(c),
            total_cycles=s.floor.end,
            latency_seconds=s.floor.end / c.clock_hz,
            groups=groups,
            ema=dict(mem.counts),
            unique_read_payload_bytes=dict(mem.unique_bytes),
            peak_sram_bytes=dict(mem.peak),
            resource_busy_cycles=dict(s.busy),
            resource_service_cycles=dict(s.service),
            overlapping_event_wait_cycles=dict(s.waits),
            work=dict(planner.stats),
            events=s.serial,
            host_replay_seconds=time.monotonic() - start,
            output_overflow_elements=manifest["output_overflow_elements"],
        )
        summary["occupancy"] = {
            "horizon_cycles": s._horizon,
            "bins": s.bin_count,
            "resources": s.occupancy_bins(),
        }
        if sink is not None:
            summary["timeline_events_written"] = sink.written
            if sink.window is not None:
                summary["timeline_window"] = list(sink.window)
        (output / "operators.json").write_text(
            json.dumps(operators, separators=(",", ":")) + "\n"
        )
        with (output / "sparsity.csv").open("w", newline="") as f:
            fields = ["operator", "batch", "m", "bg", "eg", "mode", "prescan"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(planner.sparsity_rows)
        lines = [
            "# Paper bipolar VLM/expert replay",
            "",
            "One captured inference; simulation results under explicit system assumptions.",
            "",
            "| Phase | Latency (ms) | EMA read payload (MiB) | EMA read bus (MiB) |",
            "|---|---:|---:|---:|",
        ]
        for name, g in groups.items():
            reads = (
                lambda suffix: sum(
                    v
                    for k, v in g["ema"].items()
                    if k.startswith("read.") and k.endswith(suffix)
                )
                / 2**20
            )
            lines.append(
                f'| {name} | {g["latency_seconds"]*1000:.4f} | {reads("payload_bytes"):.4f} | {reads("bus_bytes"):.4f} |'
            )
        lines += ["", "Critical-path cycles (additive within a phase):", ""]
        for name, g in groups.items():
            if name in {"vlm_prefill", "expert_denoise_total"}:
                lines.append(
                    f"- {name}: "
                    + ", ".join(
                        f"{k}={v}"
                        for k, v in sorted(g["critical"].items(), key=lambda x: -x[1])
                    )
                )
        lines += ["", "Assumptions:", ""] + ["- " + x for x in summary["assumptions"]]
        (output / "report.md").write_text("\n".join(lines) + "\n")
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if sink:
            sink.close()
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
