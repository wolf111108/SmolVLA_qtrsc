"""Output-stationary temporal tiling with explicit GB/local ping-pong slots."""
from dataclasses import asdict
from math import ceil
from ..backends.dense import packed_bytes
from .protocol import Action, ExecutionPlan, PlanEvent, UnsupportedModel, integer
from .runtime import RuntimeStats

ASSUMPTIONS = [
    "GEMM cores only; sequential calls, no inter-op or cross-batch residency; not end-to-end latency.",
    "Output-stationary temporal M/N/K loops; A/B reload for each output tile and broadcast batch.",
    "Explicit GB/local operand slots and one accumulator tile; output tiles execute sequentially.",
    "Each transfer reserves source/link/destination atomically; fixed latency holds all resources.",
    "Exclusive nonpreemptive resource service; aggregate SRAM/NoC, no bank conflicts or DRAM command timing.",
    "Local operands feed a register-equipped array once per K tile; PE register traffic/energy is not modeled.",
    "Accumulator read/write counted once per K tile; intra-tile accumulation is internal to the array.",
    "Dense packed storage regardless of sparsity; no compression, metadata, codec, energy or area model.",
    "ideal_mac_skip is a pooled-work upper-bound model without load imbalance or sparse scheduling overhead.",
    "Bias, quantization, scales, conversion arithmetic, normalization and other non-GEMM work are excluded.",
    "Mappings selected using measured runtime statistics are oracle choices, not an implemented online scheduler.",
]


class MatmulPlanner:
    def __init__(self, *, double_buffer=True, max_plan_events=200000):
        if type(double_buffer) is not bool:
            raise ValueError("double_buffer must be bool")
        integer(max_plan_events, "max_plan_events", 1)
        self.double_buffer = double_buffer
        self.max_plan_events = max_plan_events

    def lower(self, op, mapping, architecture, stats=None):
        if mapping.dataflow != "output_stationary":
            raise UnsupportedModel("Component planner v1 supports output_stationary only")
        if op.metadata.get("is_bitnet") or op.metadata.get("mixed_precision"):
            raise UnsupportedModel("BitNet/mixed precision requires a dedicated planner")
        if op.mode not in {"raw", "quant_forward"} or (op.mode == "quant_forward" and
                op.method not in {"per_tensor", "pot_fp8_per_tensor"}):
            raise UnsupportedModel("Unsupported quantization execution path")
        if max(op.a.bits, op.b.bits, op.output.bits) > 32:
            raise UnsupportedModel("Operand/output widths above 32 bits require a dedicated planner")
        stats = stats or RuntimeStats(op.event_id, source="estimated")
        if stats.event_id != op.event_id:
            raise ValueError("Runtime statistics belong to a different event")
        roles = architecture.roles
        required = {"dram", "global", "local", "accumulator", "array"}
        if not required <= roles.keys() or len({roles[r] for r in required}) != len(required):
            raise ValueError("Planner requires distinct dram/global/local/accumulator/array roles")
        dram, gb, local, acc, array = (roles[r] for r in ("dram", "global", "local", "accumulator", "array"))
        array_spec = architecture.components[array].spec
        # Validate compute capability before expanding a potentially large plan.
        architecture.components[array].validate(Action(array, "matmul", {
            "m": 1, "n": 1, "k": 1, "a_format": op.a.logical_format,
            "b_format": op.b.logical_format,
            "active_macs": 1 if array_spec.mode == "ideal_mac_skip" else None}))
        mmax, nmax, kmax = min(op.m, mapping.tile_m), min(op.n, mapping.tile_n), min(op.k, mapping.tile_k)
        slots = 2 if self.double_buffer else 1
        operands = packed_bytes(mmax * kmax, op.a.bits) + packed_bytes(kmax * nmax, op.b.bits)
        reservations = {gb: slots * operands + packed_bytes(mmax * nmax, op.output.bits),
                        local: slots * operands,
                        acc: packed_bytes(mmax * nmax, array_spec.accumulator_bits)}
        architecture.validate_reservations(reservations)
        outputs = op.batch * ceil(op.m / mapping.tile_m) * ceil(op.n / mapping.tile_n)
        expected_events = outputs * (5 * ceil(op.k / mapping.tile_k) + 2)
        if expected_events > self.max_plan_events:
            raise UnsupportedModel(f"Plan needs {expected_events} events; max_plan_events={self.max_plan_events}")
        events, stat_sources, used_keys = [], set(), set()
        previous_output = None

        def add(name, actions, dependencies):
            events.append(PlanEvent(name, tuple(actions), tuple(dict.fromkeys(d for d in dependencies if d))))
            return name

        for batch in range(op.batch):
            for mo in range(0, op.m, mapping.tile_m):
                m = min(mapping.tile_m, op.m - mo)
                for no in range(0, op.n, mapping.tile_n):
                    n = min(mapping.tile_n, op.n - no)
                    gb_free = [previous_output] * slots
                    local_free = [previous_output] * slots
                    previous_compute = previous_output
                    acc_bytes = packed_bytes(m * n, array_spec.accumulator_bits)
                    output_bytes = packed_bytes(m * n, op.output.bits)
                    for ki, ko in enumerate(range(0, op.k, mapping.tile_k)):
                        k = min(mapping.tile_k, op.k - ko)
                        slot = ki % slots
                        key = f"{batch}:{mo}:{no}:{ko}:{m}:{n}:{k}"
                        prefix = f"op{op.event_id}/{key}"
                        a_bytes, b_bytes = packed_bytes(m * k, op.a.bits), packed_bytes(k * n, op.b.bits)
                        gate = None if self.double_buffer else previous_compute
                        la = add(prefix + "/load_a", architecture.transfer(dram, gb, a_bytes),
                                 (gb_free[slot], gate))
                        lb = add(prefix + "/load_b", architecture.transfer(dram, gb, b_bytes),
                                 (gb_free[slot], gate))
                        fa = add(prefix + "/feed_a", architecture.transfer(gb, local, a_bytes),
                                 (la, local_free[slot]))
                        fb = add(prefix + "/feed_b", architecture.transfer(gb, local, b_bytes),
                                 (lb, fa, local_free[slot]))
                        active = None
                        if array_spec.mode == "ideal_mac_skip":
                            active, source = stats.active_macs(key, m * n * k)
                            stat_sources.add(source)
                            used_keys.add(key)
                        actions = [Action(local, "read", {"bytes": a_bytes + b_bytes}),
                                   architecture.link_action(local, array, a_bytes + b_bytes),
                                   Action(array, "matmul", {"m": m, "n": n, "k": k,
                                          "a_format": op.a.logical_format, "b_format": op.b.logical_format,
                                          "active_macs": active})]
                        if ko:
                            actions.append(Action(acc, "read", {"bytes": acc_bytes}))
                            actions.append(architecture.link_action(acc, array, acc_bytes))
                        actions.append(Action(acc, "write", {"bytes": acc_bytes}))
                        actions.append(architecture.link_action(array, acc, acc_bytes))
                        compute = add(prefix + "/compute", actions, (fb, previous_compute))
                        gb_free[slot], local_free[slot] = fb, compute
                        previous_compute = compute
                    prefix = f"op{op.event_id}/{batch}:{mo}:{no}"
                    drain = add(prefix + "/drain", architecture.transfer(acc, gb, acc_bytes,
                                destination_bytes=output_bytes), (previous_compute,))
                    previous_output = add(prefix + "/store", architecture.transfer(gb, dram, output_bytes), (drain,))
        if array_spec.mode == "ideal_mac_skip" and set(stats.tile_active_macs) - used_keys:
            raise ValueError("Unused tile statistics: mapping/shape mismatch")
        return ExecutionPlan(tuple(events), reservations, {
            "operator": op.to_dict(), "mapping": asdict(mapping),
            "double_buffer": self.double_buffer, "runtime_stats": stats.to_dict(),
            "compute_statistics": sorted(stat_sources) if stat_sources else ["dense_no_sparse_speedup"],
            "mapping_policy": "fixed", "assumptions": ASSUMPTIONS})
