"""Output-stationary temporal tiling with explicit GB/local ping-pong slots."""
from dataclasses import asdict
from math import ceil
from .protocol import Action, ExecutionPlan, PlanEvent, UnsupportedModel, integer, packed_bytes
from .runtime import RuntimeStats

ASSUMPTIONS = [
    "GEMM cores only; sequential calls, no inter-op or cross-batch residency; not end-to-end latency.",
    "OS/WS/IS temporal loops; WS keeps B over M, IS keeps A over N; each broadcast batch reloads inputs.",
    "Explicit GB/local operand slots and one accumulator tile; output tiles execute sequentially.",
    "WS/IS spill partial sums to DRAM between K passes and currently execute tiles sequentially.",
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
        if mapping.dataflow != "output_stationary":
            return self._lower_stationary(op, mapping, architecture, stats, roles, array_spec)
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

    def _lower_stationary(self, op, mapping, architecture, stats, roles, array_spec):
        """WS/IS: keep one operand across the inner loop; spill psums between K passes."""
        dram, gb, local, acc, array = (roles[r] for r in ("dram", "global", "local", "accumulator", "array"))
        tm, tn, tk = min(op.m, mapping.tile_m), min(op.n, mapping.tile_n), min(op.k, mapping.tile_k)
        a_max, b_max = packed_bytes(tm * tk, op.a.bits), packed_bytes(tk * tn, op.b.bits)
        psum_max = packed_bytes(tm * tn, array_spec.accumulator_bits)
        output_max = packed_bytes(tm * tn, op.output.bits)
        # The stationary operand, the streamed operand, and a psum/output slot
        # occupy independent logical storage regions at both buffer levels.
        reservations = {gb: a_max + b_max + max(psum_max, output_max),
                        local: a_max + b_max, acc: psum_max}
        architecture.validate_reservations(reservations)
        nm, nn, nk = (ceil(op.m / mapping.tile_m), ceil(op.n / mapping.tile_n),
                      ceil(op.k / mapping.tile_k))
        if nk > 1 and (nm if mapping.dataflow == "weight_stationary" else nn) > 1:
            route = (f"{gb}.data", f"{acc}.data")
            if route not in architecture.connections:
                raise UnsupportedModel(f"{mapping.dataflow} needs a {route[0]} to {route[1]} route to restore partial sums")
        output_tiles = op.batch * nm * nn * nk
        stationary_tiles = op.batch * nk * (nn if mapping.dataflow == "weight_stationary" else nm)
        expected_events = output_tiles * 7 + stationary_tiles * 2
        if expected_events > self.max_plan_events:
            raise UnsupportedModel(f"Plan needs at most {expected_events} events; max_plan_events={self.max_plan_events}")
        events, stat_sources, used_keys = [], set(), set()
        previous = None

        def add(name, actions):
            nonlocal previous
            events.append(PlanEvent(name, tuple(actions), (previous,) if previous else ()))
            previous = name

        for batch in range(op.batch):
            outer = range(0, op.n if mapping.dataflow == "weight_stationary" else op.m,
                          mapping.tile_n if mapping.dataflow == "weight_stationary" else mapping.tile_m)
            for outer_offset in outer:
                for ko in range(0, op.k, mapping.tile_k):
                    k = min(mapping.tile_k, op.k - ko)
                    if mapping.dataflow == "weight_stationary":
                        n = min(mapping.tile_n, op.n - outer_offset)
                        stationary_bytes = packed_bytes(k * n, op.b.bits)
                        inner = range(0, op.m, mapping.tile_m)
                    else:
                        m = min(mapping.tile_m, op.m - outer_offset)
                        stationary_bytes = packed_bytes(m * k, op.a.bits)
                        inner = range(0, op.n, mapping.tile_n)
                    contiguous_psum = len(inner) == 1
                    prefix = f"op{op.event_id}/{batch}:{outer_offset}:{ko}"
                    add(prefix + "/load_stationary", architecture.transfer(dram, gb, stationary_bytes))
                    add(prefix + "/feed_stationary", architecture.transfer(gb, local, stationary_bytes))
                    for inner_offset in inner:
                        mo = inner_offset if mapping.dataflow == "weight_stationary" else outer_offset
                        no = outer_offset if mapping.dataflow == "weight_stationary" else inner_offset
                        m, n = min(mapping.tile_m, op.m - mo), min(mapping.tile_n, op.n - no)
                        a_bytes, b_bytes = packed_bytes(m * k, op.a.bits), packed_bytes(k * n, op.b.bits)
                        streamed_bytes = a_bytes if mapping.dataflow == "weight_stationary" else b_bytes
                        psum_bytes = packed_bytes(m * n, array_spec.accumulator_bits)
                        output_bytes = packed_bytes(m * n, op.output.bits)
                        prefix = f"op{op.event_id}/{batch}:{mo}:{no}:{ko}"
                        add(prefix + "/load_streamed", architecture.transfer(dram, gb, streamed_bytes))
                        add(prefix + "/feed_streamed", architecture.transfer(gb, local, streamed_bytes))
                        if ko and not contiguous_psum:
                            add(prefix + "/restore_psum", architecture.transfer(dram, gb, psum_bytes))
                            add(prefix + "/feed_psum", architecture.transfer(gb, acc, psum_bytes))
                        key = f"{batch}:{mo}:{no}:{ko}:{m}:{n}:{k}"
                        active = None
                        if array_spec.mode == "ideal_mac_skip":
                            active, source = stats.active_macs(key, m * n * k)
                            stat_sources.add(source)
                            used_keys.add(key)
                        actions = [Action(local, "read", {"bytes": a_bytes + b_bytes}),
                                   architecture.link_action(local, array, a_bytes + b_bytes),
                                   Action(array, "matmul", {"m": m, "n": n, "k": k,
                                                            "a_format": op.a.logical_format,
                                                            "b_format": op.b.logical_format,
                                                            "active_macs": active})]
                        if ko:
                            actions.extend((Action(acc, "read", {"bytes": psum_bytes}),
                                            architecture.link_action(acc, array, psum_bytes)))
                        actions.extend((Action(acc, "write", {"bytes": psum_bytes}),
                                        architecture.link_action(array, acc, psum_bytes)))
                        add(prefix + "/compute", actions)
                        final = ko + k == op.k
                        if final or not contiguous_psum:
                            size = output_bytes if final else psum_bytes
                            add(prefix + ("/drain" if final else "/spill"),
                                architecture.transfer(acc, gb, psum_bytes, destination_bytes=size))
                            add(prefix + ("/store" if final else "/store_psum"),
                                architecture.transfer(gb, dram, size))
        if array_spec.mode == "ideal_mac_skip" and set(stats.tile_active_macs) - used_keys:
            raise ValueError("Unused tile statistics: mapping/shape mismatch")
        return ExecutionPlan(tuple(events), reservations, {
            "operator": op.to_dict(), "mapping": asdict(mapping), "double_buffer": False,
            "runtime_stats": stats.to_dict(),
            "compute_statistics": sorted(stat_sources) if stat_sources else ["dense_no_sparse_speedup"],
            "mapping_policy": "fixed", "assumptions": ASSUMPTIONS})
