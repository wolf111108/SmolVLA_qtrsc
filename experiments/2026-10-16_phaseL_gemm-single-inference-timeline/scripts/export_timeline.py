#!/usr/bin/env python3
"""Stream component simulations to a global GEMM timeline, Chrome Trace and utilization tables."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from math import ceil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from vla_tcs2.hardware import MappingSpec
from vla_tcs2.hardware.modeling import Architecture, ExecutionPlan, MatmulPlanner
from vla_tcs2.hardware.trace import read_trace


def export(run_dir, output_dir=None):
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir.parent / "timeline"
    if output_dir.exists():
        raise FileExistsError(f"Choose a new export directory: {output_dir}")
    records = read_trace(run_dir / "trace.jsonl")
    manifest = next(records)
    if manifest.get("backend") != "component_sim":
        raise ValueError("Only component_sim results can be exported")
    from vla_tcs2.hardware.modeling import default_architecture
    cfg = manifest.get("backend_config", {})
    architecture = Architecture(cfg["architecture"]) if cfg.get("architecture") else default_architecture()
    clock = architecture.clock_hz
    planner = MatmulPlanner(double_buffer=cfg.get("double_buffer", True),
                            max_plan_events=cfg.get("max_plan_events", 200000))
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary["status"] != "completed" or summary["capture_truncated"] or summary["unsupported_reasons"]:
        raise ValueError("Source capture is failed, truncated or contains unsupported operators")
    if architecture.components[architecture.roles["array"]].spec.mode != "dense":
        raise ValueError("This experiment exports dense GEMM runs only")
    output_dir.mkdir(parents=True)
    resources = list(architecture.resources)
    tids = {name: i + 1 for i, name in enumerate(resources)}
    busy, service, counters = Counter(), Counter(), Counter()
    union_busy, union_end = Counter(), defaultdict(int)
    by_stage = defaultdict(lambda: {"cycles": 0, "busy": Counter(), "service": Counter()})
    offset = count = 0
    representatives = {}
    # Compact interval data are lossless; Chrome Trace duplicates each event across resource lanes.
    with gzip.open(output_dir / "events.csv.gz", "wt", newline="", compresslevel=3) as csv_stream, \
         gzip.open(output_dir / "chrome_trace.json.gz", "wt", compresslevel=3) as chrome, \
         (output_dir / "operators.csv").open("w", newline="") as operators, \
         (run_dir / "simulations.jsonl").open() as simulations:
        fields = ["operator_id", "module_id", "component", "phase", "flow_step", "generation_id",
                  "tile_event", "event_phase", "operation", "resources", "depends_on", "ready_cycle", "start_cycle", "end_cycle",
                  "resource_wait_cycles", "fixed_cycles", "resource_service_cycles"]
        writer = csv.DictWriter(csv_stream, fieldnames=fields)
        writer.writeheader()
        op_fields = ["operator_id", "module_id", "component", "phase", "flow_step", "m", "n", "k", "batch",
                     "a_bits", "b_bits", "output_bits", "dataflow", "tile_m", "tile_n", "tile_k",
                     "start_cycle", "end_cycle", "latency_seconds", "dense_macs", "array_service_cycles",
                     "dram_read_bytes", "dram_write_bytes", "dram_reserved_cycles", "dram_service_cycles"]
        op_writer = csv.DictWriter(operators, fieldnames=op_fields)
        op_writer.writeheader()
        chrome.write('{"displayTimeUnit":"ns","traceEvents":[')
        first = True
        def emit(item):
            nonlocal first
            chrome.write(("" if first else ",") + json.dumps(item, separators=(",", ":")))
            first = False
        for resource, tid in tids.items():
            emit({"ph": "M", "pid": 1, "tid": tid, "name": "thread_name", "args": {"name": resource}})
        for event in records:
            line = simulations.readline()
            if not line:
                raise ValueError("Missing simulation record")
            record = json.loads(line)
            if record["event_id"] != event.event_id or record["status"] != "estimated":
                raise ValueError("Trace/simulation ID or status mismatch")
            sim = record["simulation"]
            mapping = MappingSpec(**record["mapping"])
            plan = (ExecutionPlan.from_dict(record["plan"]) if "plan" in record else
                    planner.lower(event, mapping, architecture))
            models = {e.id: architecture.lower_event(e) for e in plan.events}
            dependencies = {e.id: e.depends_on for e in plan.events}
            expected_counts = Counter()
            for model in models.values():
                expected_counts.update(model.counters)
            if dict(expected_counts) != sim["counters"] or plan.reservations != sim["reservations"]:
                raise ValueError("Reconstructed plan differs from stored simulation; use matching code/config")
            phase_service = {}
            op_service = Counter()
            for key, model in models.items():
                for i, phase in enumerate(model.phases):
                    demands = {d.resource: ceil(d.amount / architecture.resources[d.resource].units_per_cycle)
                               for d in phase.demands}
                    phase_service[key, i] = (phase.fixed_cycles, demands)
                    op_service.update(demands)
            stage = f"{event.phase}:{event.flow_step}"
            by_stage[stage]["cycles"] += sim["total_cycles"]
            by_stage[stage]["busy"].update(sim["resource_busy_cycles"])
            by_stage[stage]["service"].update(op_service)
            busy.update(sim["resource_busy_cycles"])
            service.update(op_service)
            counters.update(sim["counters"])
            for timing in sim["timeline"]:
                fixed, demands = phase_service[timing["event_id"], timing["phase"]]
                start, end = offset + timing["start_cycle"], offset + timing["end_cycle"]
                operation = timing["event_id"].rsplit("/", 1)[-1]
                row = dict(operator_id=event.event_id, module_id=event.module_id, component=event.component,
                           phase=event.phase, flow_step=event.flow_step, generation_id=event.generation_id,
                           tile_event=timing["event_id"], event_phase=timing["phase"], operation=operation,
                           resources="|".join(timing["resources"]), depends_on="|".join(dependencies[timing["event_id"]]),
                           ready_cycle=offset + timing["ready_cycle"],
                           start_cycle=start, end_cycle=end, resource_wait_cycles=timing["start_cycle"] - timing["ready_cycle"],
                           fixed_cycles=fixed, resource_service_cycles=json.dumps(demands, separators=(",", ":")))
                writer.writerow(row)
                for component in {r.split(".")[0] for r in timing["resources"]}:
                    union_busy[component] += max(0, end - max(start, union_end[component]))
                    union_end[component] = max(union_end[component], end)
                for resource in timing["resources"]:
                    emit({"ph": "X", "pid": 1, "tid": tids[resource], "name": f"{event.module_id}/{operation}",
                          "cat": f"{event.component}.{event.phase}", "ts": start / clock * 1e6,
                          "dur": (end - start) / clock * 1e6,
                          "args": {"operator_id": event.event_id, "tile_event": timing["event_id"],
                                   "flow_step": event.flow_step, "start_cycle": start, "end_cycle": end,
                                   "ready_cycle": row["ready_cycle"], "resource_service_cycles": demands.get(resource, 0)}})
            dram, array = architecture.roles["dram"], architecture.roles["array"]
            def subtotal(values, prefix):
                return sum(v for k, v in values.items() if k.startswith(prefix + "."))
            op_writer.writerow(dict(operator_id=event.event_id, module_id=event.module_id, component=event.component,
                                   phase=event.phase, flow_step=event.flow_step, m=event.m, n=event.n, k=event.k, batch=event.batch,
                                   a_bits=event.a.bits, b_bits=event.b.bits, output_bits=event.output.bits,
                                   **record["mapping"], start_cycle=offset, end_cycle=offset + sim["total_cycles"],
                                   latency_seconds=sim["latency_seconds"], dense_macs=event.macs,
                                   array_service_cycles=sim["counters"].get(array + ".compute_cycles", 0),
                                   dram_read_bytes=sim["counters"].get(dram + ".read_bytes", 0),
                                   dram_write_bytes=sim["counters"].get(dram + ".write_bytes", 0),
                                   dram_reserved_cycles=subtotal(sim["resource_busy_cycles"], dram),
                                   dram_service_cycles=subtotal(op_service, dram)))
            # Keep a representative nontrivial GEMM per model component for exact local Gantt plots.
            if event.component not in representatives or event.macs > representatives[event.component]["macs"]:
                representatives[event.component] = {"module_id": event.module_id, "operator_id": event.event_id,
                    "macs": event.macs, "offset": offset, "total_cycles": sim["total_cycles"], "timeline": sim["timeline"]}
            offset += sim["total_cycles"]
            count += 1
        if simulations.readline():
            raise ValueError("Extra simulation records")
        chrome.write("]}\n")
    records.close()
    if count != summary["totals"]["estimated_calls"]:
        raise ValueError("Summary operator count mismatch")
    with (output_dir / "resources.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["resource", "reserved_cycles", "service_cycles", "reserved_fraction", "service_fraction"])
        writer.writeheader()
        for resource in resources:
            if not 0 <= service[resource] <= busy[resource] <= offset:
                raise ValueError(f"Invalid physical resource occupancy: {resource}")
            writer.writerow(dict(resource=resource, reserved_cycles=busy[resource], service_cycles=service[resource],
                                 reserved_fraction=busy[resource] / offset if offset else 0,
                                 service_fraction=service[resource] / offset if offset else 0))
    with (output_dir / "components.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["component", "any_port_reserved_cycles", "reserved_union_fraction"])
        writer.writeheader()
        for component in sorted(union_busy):
            writer.writerow(dict(component=component, any_port_reserved_cycles=union_busy[component],
                                 reserved_union_fraction=union_busy[component] / offset if offset else 0))
    stage_result = {key: {"cycles": values["cycles"], "seconds": values["cycles"] / clock,
                         "resource_busy_cycles": dict(values["busy"]), "resource_service_cycles": dict(values["service"])}
                    for key, values in by_stage.items()}
    spec = architecture.components[architecture.roles["array"]].spec
    executed = counters.get(architecture.roles["array"] + ".executed_macs", 0)
    result = {"clock_hz": clock, "total_cycles": offset, "gemm_only_seconds": offset / clock,
              "operators": count, "resources": resources, "counters": dict(counters), "stages": stage_result,
              "effective_pe_utilization": executed / (spec.rows * spec.cols * offset) if offset else 0,
              "scope": "GEMM-only serial concatenation in capture order; non-GEMM work has no simulated duration",
              "utilization_definition": "reserved = phase resource lock duration; service = summed ceil(demand/rate), excludes fixed latency; service placement within lock is unspecified",
              "wait_definition": "per-event ready-to-start queue wait; overlapping waits must not be summed as critical-path stalls"}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    dram, array = architecture.roles["dram"], architecture.roles["array"]
    def resource_sum(values, component):
        return sum(v for k, v in values.items() if k.startswith(component + "."))
    report = ["# 单次推理 GEMM-only 硬件结果", "", result["scope"], "",
              "| 指标 | 值 |", "|---|---:|",
              f"| GEMM 数 | {count} |", f"| GEMM-only 总延迟 | {offset / clock * 1000:.6f} ms |",
              f"| DRAM 读取 | {counters.get(dram + '.read_bytes', 0)} B |",
              f"| DRAM 写入 | {counters.get(dram + '.write_bytes', 0)} B |",
              f"| 阵列计算服务周期折算 | {resource_sum(service, array) / clock * 1000:.6f} ms |",
              f"| DRAM 传输服务周期折算（不含固定延迟） | {resource_sum(service, dram) / clock * 1000:.6f} ms |",
              f"| DRAM 资源占用周期折算（含固定延迟） | {resource_sum(busy, dram) / clock * 1000:.6f} ms |",
              f"| 有效 PE 利用率 | {result['effective_pe_utilization']:.3%} |", "",
              "计算、访存周期可能重叠，不能相加当作总延迟；资源占用不等于有效服务。", "",
              "## 各阶段", "", "| phase:flow_step | GEMM 时间 (ms) |", "|---|---:|"]
    report.extend(f"| {key} | {v['seconds'] * 1000:.6f} |" for key, v in stage_result.items())
    report += ["", "## 资源利用率", "", "| 资源 | 占用率 | 服务需求/总时间 |", "|---|---:|---:|"]
    report.extend(f"| {r} | {busy[r] / offset if offset else 0:.3%} | {service[r] / offset if offset else 0:.3%} |"
                  for r in sorted(resources, key=lambda r: service[r], reverse=True))
    report += ["", "高服务需求比例是瓶颈候选；高锁定但低服务需求可能来自固定延迟或同步事务等待。",
               "逐事件等待时间可同时发生，求和不代表关键路径停顿。详见 events.csv.gz、operators.csv 和 Perfetto 时间线。", ""]
    (output_dir / "report.md").write_text("\n".join(report))
    with gzip.open(output_dir / "representatives.json.gz", "wt") as stream:
        json.dump(representatives, stream)
    print(output_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="HardwareManager output directory")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    export(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
