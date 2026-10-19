"""OS BG-major planner, finite psums, SSS pre-scan and bounded streaming runs.

The 64x256 logical macro is a scheduling choice, not a paper requirement.
Mode-dependent microtiles are remapped through row-major INT32 backing psums.
A microtile keeps its accumulator over all K and weight digits for one BG;
BG boundaries explicitly spill/restore. No fictitious 64-KiB on-chip accumulator.
"""

from collections import Counter
from math import ceil, prod, log2
import numpy as np
from .schedule import work, Completion
from .memory import span
from .sparsity import effective_mask, digits
from .workload import read_code

# Operations supported by the auxiliary analytical model. Unknown operators fail
# closed rather than disappearing from a complete-inference report.
ELEMENTWISE = {
    "add",
    "add_",
    "sub",
    "sub_",
    "mul",
    "mul_",
    "div",
    "div_",
    "pow",
    "neg",
    "abs",
    "sqrt",
    "rsqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tanh",
    "sigmoid",
    "silu",
    "gelu",
    "relu",
    "clamp",
    "clamp_",
    "clamp_min",
    "clamp_max",
    "where",
    "masked_fill",
    "masked_fill_",
    "eq",
    "ne",
    "gt",
    "ge",
    "lt",
    "le",
    "logical_and",
    "logical_or",
    "bitwise_and",
    "bitwise_or",
    "bitwise_not",
    "logical_not",
    "floor",
    "round",
    "reciprocal",
    "_to_copy",
    "copy_",
    "clone",
    "contiguous",
    "cat",
    "stack",
    "repeat",
    "repeat_interleave",
    "index",
    "index_select",
    "gather",
    "embedding",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "empty",
    "empty_like",
    "empty_strided",
    "arange",
    "scalar_tensor",
    "lift_fresh",
    "lift_fresh_copy",
    "fill_",
    "zero_",
    "triu",
    "tril",
    "cumsum",
    "cumprod",
    "view_copy",
    "alias_copy",
    "detach_copy",
}
REDUCTIONS = {
    "sum",
    "mean",
    "amax",
    "amin",
    "max",
    "min",
    "var",
    "var_mean",
    "std",
    "norm",
    "linalg_vector_norm",
    "all",
    "any",
}
SPECIAL = {
    "exp",
    "log",
    "rsqrt",
    "reciprocal",
    "sin",
    "cos",
    "tanh",
    "sigmoid",
    "silu",
    "gelu",
}
FUSED = {
    "_softmax",
    "softmax",
    "_safe_softmax",
    "native_layer_norm",
    "layer_norm",
    "rms_norm",
}


class PaperPlanner:
    def __init__(self, scheduler, memory, config, workload):
        self.s, self.mem, self.c, self.root = scheduler, memory, config, workload
        self.stats = Counter()
        self.bsc_free = [Completion(), Completion()]
        self.code_ready = {}
        self.sparsity_rows = []
        self.coordinates = {}
        self.resident = None

    def vector(
        self, kind, elements, deps=(), *, special=False, factor=1, reduction=False
    ):
        width = self.c.auxiliary_elements_per_cycle
        cycles = ceil(elements / width) * factor + (
            ceil(log2(width)) if reduction else 0
        )
        cycles += self.c.auxiliary_special_latency if special else 1
        cycles = ceil(cycles / self.c.auxiliary_scale)
        return self.s.issue(
            kind, "auxiliary", (work("aux", kind, vector=cycles),), deps
        )

    def auxiliary(self, rec):
        name = rec["op"].split(".")[1]
        if name not in ELEMENTWISE | REDUCTIONS | FUSED:
            raise ValueError(
                f'Unmodeled auxiliary operator {rec["op"]}; extend cost model explicitly'
            )
        if name in {"embedding", "index_select", "gather", "index"}:
            return self.lookup(rec)
        outs = rec["outputs"]
        if name in {"empty", "empty_like", "empty_strided"}:
            for d in outs:
                self.mem.raw(d)
            self.stats["auxiliary_operators"] += 1
            return self.s.floor  # Allocation has no initialization/data transfer.
        inputs = rec["inputs"][1:] if name == "copy_" else rec["inputs"]
        if name in {"fill_", "zero_", "zeros_like", "ones_like", "full_like"}:
            inputs = []
        size = max([d["numel"] for d in outs] + [d["numel"] for d in inputs] + [1])
        # Tiled multi-pass service. Reduction scalar registers are bounded by a
        # row, softmax/normalization re-read operands on each required pass.
        passes = 3 if name in FUSED else 1
        chunk = max(1, self.c.memory_block_bytes // 4)
        tail = self.s.floor
        for pass_idx in range(passes):
            for base in range(0, size, chunk):
                n = min(chunk, size - base)
                deps = [tail]
                for d in inputs:
                    if not d["numel"]:
                        continue
                    b = self.mem.raw(d)
                    off, length = span(d)
                    # View span overfetch is conservative and appears in EMA.
                    start = off + (length * base // size)
                    end = off + ceil(length * (base + n) / size)
                    deps.append(self.mem.access("input", b, start, end - start))
                tail = self.vector(
                    rec["op"],
                    n,
                    deps,
                    special=name in SPECIAL | FUSED,
                    reduction=name in REDUCTIONS | FUSED,
                )
                if pass_idx == passes - 1:
                    for d in outs:
                        b = self.mem.raw(d)
                        off, length = span(d)
                        start = off + length * base // size
                        end = off + ceil(length * (base + n) / size)
                        if end > start:
                            tail = self.mem.access(
                                "input", b, start, end - start, write=True, deps=(tail,)
                            )
        self.stats["auxiliary_operators"] += 1
        return tail

    def lookup(self, rec):
        if "lookup_file" not in rec:
            raise ValueError("Lookup requires captured selected storage ranges")
        source = self.mem.raw(rec["inputs"][0])
        tail = self.s.floor
        for d in rec["inputs"][1:]:
            off, length = span(d)
            tail = self.mem.access("input", self.mem.raw(d), off, length, deps=(tail,))
        with np.load(self.root / rec["lookup_file"], allow_pickle=False) as data:
            selected_bytes = int(data["lengths"].sum())
            consumed = 0
            for off, length in zip(data["offsets"], data["lengths"]):
                for start in range(
                    int(off), int(off + length), self.c.memory_block_bytes
                ):
                    amount = min(self.c.memory_block_bytes, int(off + length) - start)
                    read = self.mem.access("input", source, start, amount, deps=(tail,))
                    tail = self.vector(
                        rec["op"],
                        ceil(amount / rec["inputs"][0]["element_bytes"]),
                        (read,),
                    )
                    for d in rec["outputs"]:
                        outoff, outlen = span(d)
                        begin = outoff + outlen * consumed // max(selected_bytes, 1)
                        end = outoff + outlen * (consumed + amount) // max(
                            selected_bytes, 1
                        )
                        tail = self.mem.access(
                            "input",
                            self.mem.raw(d),
                            begin,
                            end - begin,
                            write=True,
                            deps=(tail,),
                        )
                    consumed += amount
        self.stats["auxiliary_operators"] += 1
        return tail

    def prepare_code(self, rec, role, pool):
        desc = rec[role]
        raw = rec["inputs"][0 if role == "a" else 1]
        scale = rec["scales"][0 if role == "a" else 1]
        ident = f'q:{role}:{raw["id"]}:{raw.get("version",0)}:{raw["offset"]}:{raw["shape"]}:{raw["stride"]}:{scale}:{desc["sha256"]}'
        category = "activation"
        if role == "b":
            phases = raw.get("source_phases", [])
            category = (
                "vlm_kv"
                if phases == ["prefill"]
                else (
                    "mixed_kv"
                    if "prefill" in phases
                    else "expert_kv" if rec["component"] == "expert" else "activation"
                )
            )
        b = self.mem.buffer(
            ident,
            ceil(prod(desc["shape"]) * desc["bits"] / 8),
            category,
            initialized=False,
        )
        if role == "a":
            signs = self.mem.buffer(
                "sign:" + ident,
                ceil(prod(desc["shape"]) / 8),
                "metadata",
                initialized=False,
            )
        if ident in self.code_ready:
            self.stats["quantization_reuses"] += 1
            return b
        source = self.mem.raw(raw, category)
        roff, rlen = span(raw)
        block = self.c.memory_block_bytes
        for off in range(0, b.size, block):
            count = min(block, b.size - off)
            # Proportional source span handles transposed and broadcast views.
            start = roff + rlen * off // max(b.size, 1)
            end = roff + ceil(rlen * (off + count) / max(b.size, 1))
            loaded = self.mem.access("input", source, start, end - start)
            q = self.vector("quantize_input", count * 8 // desc["bits"], (loaded,))
            self.mem.access(pool, b, off, count, write=True, deps=(q,))
            if role == "a":
                so = off // 8
                se = ceil((off + count) / 8)
                self.mem.access("metadata", signs, so, se - so, write=True, deps=(q,))
        self.code_ready[ident] = True
        return b

    def load_matrix_tile(
        self, b, pool, batch, rows, cols, r0, rn, c0, cn, bits, deps=(), service=True
    ):
        done = []
        # Contiguous N tiles in row-major B; no artificial per-element DRAM request.
        for r in range(r0, r0 + rn):
            firstbit = ((batch * rows + r) * cols + c0) * bits
            off = firstbit // 8
            size = ceil((firstbit % 8 + cn * bits) / 8)
            done.append(self.mem.access(pool, b, off, size, deps=deps, service=service))
        return max([self.s.floor, *done, *deps], key=lambda x: x.end)

    def gemm(self, rec):
        a = read_code(self.root, rec["a"])
        params = self.mem.buffer(
            "params:" + rec["module_id"] + ":" + str(rec["scales"]),
            12 + 4 * rec.get("bias_elements", 0),
            "auxiliary",
        )
        self.mem.access("input", params, 0, 12)
        if rec["linear"]:
            a = a.reshape(1, -1, a.shape[-1])
            n = rec["b"]["shape"][0]
            k = rec["b"]["shape"][1]
            a_batches = b_batches = np.zeros(1, dtype=np.int64)
        else:
            n = rec["b"]["shape"][-1]
            k = a.shape[-1]
            batch_shape = np.broadcast_shapes(
                a.shape[:-2], tuple(rec["b"]["shape"][:-2])
            )

            def batch_indices(shape):
                return np.broadcast_to(
                    np.arange(prod(shape)).reshape(shape), batch_shape
                ).reshape(-1)

            a_batches = batch_indices(a.shape[:-2])
            b_batches = batch_indices(tuple(rec["b"]["shape"][:-2]))
            a = np.broadcast_to(a, (*batch_shape, *a.shape[-2:])).reshape(
                -1, *a.shape[-2:]
            )
        batches, m, _ = a.shape
        if k * (1 << (rec["a"]["bits"] - 1)) * (1 << (rec["b"]["bits"] - 1)) >= (
            1 << 31
        ):
            raise ValueError("Worst-case INT32 accumulation bound exceeded")
        if m == 0 or n == 0 or k == 0:
            raise ValueError("Empty GEMM not supported")
        if rec["linear"]:
            # Stored artifact is [N,K]; simulated external layout is offline packed [K,N].
            b = self.mem.buffer(
                "weight:" + rec["module_id"] + ":" + rec["b"]["sha256"],
                ceil(k * n * rec["b"]["bits"] / 8),
                "weight",
            )
        else:
            b = self.prepare_code(rec, "b", "weight")
        aq = self.prepare_code(rec, "a", "input")
        self.s.barrier()  # quantized representations exist before BG processing
        self.mem.prune()
        psum = self.mem.buffer(
            f'psum:{rec["id"]}', batches * m * n * 4, "partial_sum", initialized=False
        )
        out = self.mem.raw(rec["outputs"][0])
        bw = rec["b"]["bits"] // 4
        self.stats["dense_macs"] += batches * m * n * k
        for batch in range(batches):
            for ms in range(0, m, 64):
                slab = a[batch, ms : ms + 64]
                # Decisions depend on first 8 tokens exactly as in the paper.
                dense = self.c.group == "D0"
                for lv in range(3, -1, -1):
                    mask8 = effective_mask(slab, lv, 8)[0]
                    eg = float(mask8.any(axis=-1).mean())
                    pre = self.c.group in {"D2", "D3"} and not dense
                    if pre:
                        self.prescan(
                            rec,
                            aq,
                            batch,
                            int(a_batches[batch]),
                            m,
                            k,
                            ms,
                            min(8, len(slab)),
                            lv,
                            mask8,
                        )
                        dense = eg > self.c.sss_threshold
                        if not dense:
                            for subm in range(8, len(slab), 8):
                                self.prescan(
                                    rec,
                                    aq,
                                    batch,
                                    int(a_batches[batch]),
                                    m,
                                    k,
                                    ms + subm,
                                    min(8, len(slab) - subm),
                                    lv,
                                    effective_mask(slab[subm : subm + 8], lv, 8)[0],
                                )
                    p, bc, prescanned = 64 if dense else 8, 32 if dense else 256, pre
                    self.sparsity_rows.append(
                        dict(
                            operator=rec["id"],
                            batch=batch,
                            m=ms,
                            bg=lv,
                            eg=eg,
                            mode="dense" if dense else "sparse",
                            prescan=pre,
                        )
                    )
                    masks = effective_mask(slab, lv, p)
                    for mm in range(0, len(slab), p):
                        pm = min(p, len(slab) - mm)
                        for nn in range(0, n, bc):
                            bn = min(bc, n - nn)
                            coord = {
                                "batch": batch,
                                "input_batch": int(a_batches[batch]),
                                "weight_batch": int(b_batches[batch]),
                                "m": ms + mm,
                                "n": nn,
                                "bg": lv,
                                "p": p,
                                "b": bc,
                            }
                            identity = (batch, ms + mm, pm, nn, bn)
                            restore = self.s.floor
                            if (
                                self.resident is not None
                                and self.resident[0] != identity
                            ):
                                old, done = self.resident
                                ob, om, op, on, oc = old
                                self.psum_io(
                                    psum, ob, m, n, om, op, on, oc, "write", (done,)
                                )
                                self.s.barrier()
                                self.mem.prune()
                                self.resident = None
                            if self.resident is not None:
                                restore = self.resident[1]
                                self.stats["psum_retained_tiles"] += 1
                            elif lv != 3:
                                restore = self.psum_io(
                                    psum, batch, m, n, ms + mm, pm, nn, bn, "read"
                                )
                            last = restore
                            for kt in range(0, k, self.c.k_tile):
                                kn = min(self.c.k_tile, k - kt)
                                # Two BSC local buffers cover preparation/consume overlap.
                                for ti in range(kt // 16, ceil((kt + kn) / 16)):
                                    valid = min(16, k - ti * 16)
                                    mask = masks[mm // p, ti, :valid]
                                    te = (
                                        valid
                                        if self.c.group == "D0"
                                        else int(mask.sum())
                                    )
                                    self.stats["candidate_packs"] += valid * bw
                                    self.stats["executed_packs"] += te * bw
                                    self.stats["skipped_packs"] += (valid - te) * bw
                                    if prescanned and p == 8 and not te:
                                        self.stats["prescan_skipped_tiles"] += 1
                                        continue
                                    coordinate_ready = self.s.floor
                                    if prescanned and p == 8:
                                        coordbuf = self.coordinates[
                                            (rec["id"], batch, ms + mm, lv)
                                        ]
                                        coordinate_ready = self.mem.access(
                                            "metadata", coordbuf, ti * 8, 8
                                        )
                                    which = ti % 2
                                    bsc = f"bsc{which}"
                                    free = self.bsc_free[which]
                                    abatch = int(a_batches[batch])
                                    signbuf = self.mem.buffers["sign:" + aq.id]
                                    detects = []
                                    # Consume one p8 plane at a time. Staging all p64 rows
                                    # first can exceed SRAM capacity for large K strides.
                                    for sub in range(ceil(p / 8)):
                                        first = ms + mm + sub * 8
                                        count = max(0, min(8, pm - sub * 8))
                                        a_load = self.load_matrix_tile(
                                            aq,
                                            "input",
                                            abatch,
                                            m,
                                            k,
                                            first,
                                            count,
                                            ti * 16,
                                            valid,
                                            8,
                                            (free,),
                                            service=False,
                                        )
                                        signload = self.load_matrix_tile(
                                            signbuf,
                                            "metadata",
                                            abatch,
                                            m,
                                            k,
                                            first,
                                            count,
                                            ti * 16,
                                            valid,
                                            1,
                                            (free,),
                                            service=False,
                                        )
                                        ranges = [
                                            ((abatch * m + row) * k + ti * 16, valid)
                                            for row in range(first, first + count)
                                        ]
                                        input_banks = self.mem.bank_demands(
                                            "input", aq, ranges
                                        )
                                        load = self.s.issue(
                                            "bsc.C1",
                                            "bsc",
                                            (
                                                work(
                                                    "input",
                                                    "bg_read",
                                                    read=32,
                                                    **input_banks,
                                                ),
                                                work("metadata", "sign_read", read=16),
                                            )
                                            + (
                                                ()
                                                if self.c.ideal_bsc
                                                else (work(bsc, "load", load=1),)
                                            ),
                                            (a_load, signload, free, coordinate_ready),
                                            {
                                                **coord,
                                                "k": ti * 16,
                                                "Te": te,
                                                "sub": sub,
                                            },
                                        )
                                        self.mem.hold("input", load)
                                        self.mem.hold("metadata", load)
                                        detect = self.s.issue(
                                            "bsc.C2",
                                            "bsc",
                                            (
                                                ()
                                                if self.c.ideal_bsc
                                                or self.c.group == "D0"
                                                else (work(bsc, "detect", detect=1),)
                                            ),
                                            (load,),
                                            coord,
                                        )
                                        detects.append(detect)
                                    compact = self.s.issue(
                                        "bsc.C3",
                                        "bsc",
                                        (
                                            ()
                                            if self.c.ideal_bsc or self.c.group == "D0"
                                            else (work(bsc, "compact", compact=1),)
                                        ),
                                        tuple(detects),
                                        coord,
                                    )
                                    if (
                                        self.c.group == "D3"
                                        and rec.get("iag_file")
                                        and nn == 0
                                    ):
                                        self.s.issue(
                                            "iag.score_only",
                                            "iag",
                                            (
                                                work(
                                                    "iag",
                                                    "score",
                                                    score=ceil(pm * valid / 128),
                                                ),
                                            ),
                                            (detect,),
                                            coord,
                                        )
                                    if te:
                                        # Fill contiguous external W tile first; only SRAM reads
                                        # feeding the array are sparse-addressed (no DRAM skipping claim).
                                        wb_batch = int(b_batches[batch])
                                        wt = self.load_matrix_tile(
                                            b,
                                            "weight",
                                            wb_batch,
                                            k,
                                            n,
                                            ti * 16,
                                            valid,
                                            nn,
                                            bn,
                                            rec["b"]["bits"],
                                            service=False,
                                        )
                                        indices = (
                                            np.flatnonzero(mask)
                                            if self.c.group != "D0"
                                            else np.arange(valid)
                                        )
                                        wranges = []
                                        for j in indices:
                                            bit = (
                                                (wb_batch * k + ti * 16 + int(j)) * n
                                                + nn
                                            ) * rec["b"]["bits"]
                                            wranges.append(
                                                (
                                                    bit // 8,
                                                    ceil(
                                                        (
                                                            bit % 8
                                                            + bn * rec["b"]["bits"]
                                                        )
                                                        / 8
                                                    ),
                                                )
                                            )
                                        weight_banks = self.mem.bank_demands(
                                            "weight", b, wranges
                                        )
                                        compute = self.s.issue(
                                            "array.digit_run",
                                            "compute",
                                            (
                                                work("array", "compute", issue=te * bw),
                                                work(bsc, "emit", emit=te * bw),
                                                work(
                                                    "weight",
                                                    "selected_read",
                                                    read=ceil(
                                                        te * bn * rec["b"]["bits"] / 8
                                                    ),
                                                    **weight_banks,
                                                ),
                                            ),
                                            (compact, wt, last),
                                            {
                                                **coord,
                                                "k": ti * 16,
                                                "Te": te,
                                                "weight_digits": bw,
                                                "effective_addresses": (
                                                    np.flatnonzero(mask).tolist()
                                                    if self.c.group != "D0"
                                                    else list(range(valid))
                                                ),
                                                "weight_sram_service_bytes": ceil(
                                                    te * bn * rec["b"]["bits"] / 8
                                                ),
                                            },
                                        )
                                        self.stats["digit_mac_slots"] += (
                                            pm * bn * te * bw
                                        )
                                        # Effective nonzero individual contributions distinguish pack waste.
                                        dd = digits(
                                            slab[
                                                mm : mm + pm, ti * 16 : ti * 16 + valid
                                            ]
                                        )[lv]
                                        self.stats["nonzero_digit_macs"] += (
                                            int(np.count_nonzero(dd)) * bn * bw
                                        )
                                        self.mem.hold("weight", compute)
                                        last = compute
                                    else:
                                        last = self.s.issue(
                                            "empty_tile", "control", (), (last, compact)
                                        )
                                    self.bsc_free[which] = last
                            if lv:
                                self.resident = (identity, last)
                            else:
                                self.resident = None
                                bias = (
                                    self.mem.access(
                                        "input", params, 12 + nn * 4, bn * 4
                                    )
                                    if rec.get("bias_elements")
                                    else self.s.floor
                                )
                                q = self.vector(
                                    "requantize_output", pm * bn, (last, bias)
                                )
                                for row in range(ms + mm, ms + mm + pm):
                                    offset = ((batch * m + row) * n + nn) * rec[
                                        "outputs"
                                    ][0]["element_bytes"]
                                    self.mem.access(
                                        "input",
                                        out,
                                        offset,
                                        bn * rec["outputs"][0]["element_bytes"],
                                        write=True,
                                        deps=(q,),
                                    )
                            # One accumulator context; next microtile cannot overwrite it early.
                            self.s.barrier()
                            self.mem.prune()
                            self.bsc_free = [self.s.floor, self.s.floor]
        return self.s.last

    def psum_io(self, b, batch, m, n, ms, pm, ns, bn, direction, deps=()):
        out = []
        for row in range(ms, ms + pm):
            off = ((batch * m + row) * n + ns) * 4
            if direction == "read":
                transfer = self.mem.dma(b, off, bn * 4, "read", deps)
                token = self.s.issue(
                    "acc.restore",
                    "partial_sum",
                    (work("acc", "write", write=bn * 4),),
                    (transfer,),
                )
            else:
                token = self.s.issue(
                    "acc.spill",
                    "partial_sum",
                    (work("acc", "read", read=bn * 4),),
                    deps,
                )
                token = self.mem.dma(b, off, bn * 4, "write", (token,))
            out.append(token)
        return max([self.s.floor, *out, *deps], key=lambda x: x.end)

    def prescan(self, rec, a, batch, abatch, m, k, ms, pm, lv, mask):
        # Metadata scoped to resident scan batches. No free full-layer coordinate cache.
        per_entry = 8  # coordinate + count, declared system assumption
        coordbuf = self.mem.buffer(
            f'coords:{rec["id"]}:{batch}:{ms}:{lv}',
            len(mask) * per_entry,
            "metadata",
            initialized=False,
        )
        self.coordinates[(rec["id"], batch, ms, lv)] = coordbuf
        max_entries = max(1, (self.c.metadata_bytes - 1024) // per_entry)
        for base in range(0, len(mask), max_entries):
            chunk = mask[base : base + max_entries]
            for j, values in enumerate(chunk, base):
                free = self.bsc_free[j % 2]
                load = self.load_matrix_tile(
                    a,
                    "input",
                    abatch,
                    m,
                    k,
                    ms,
                    pm,
                    j * 16,
                    min(16, k - j * 16),
                    8,
                    (free,),
                    service=False,
                )
                signbuf = self.mem.buffers["sign:" + a.id]
                signload = self.load_matrix_tile(
                    signbuf,
                    "metadata",
                    abatch,
                    m,
                    k,
                    ms,
                    pm,
                    j * 16,
                    min(16, k - j * 16),
                    1,
                    (free,),
                    service=False,
                )
                banks = self.mem.bank_demands(
                    "input",
                    a,
                    [
                        ((abatch * m + row) * k + j * 16, min(16, k - j * 16))
                        for row in range(ms, ms + pm)
                    ],
                )
                c1 = self.s.issue(
                    "sss.prescan.C1",
                    "sss",
                    (
                        work("input", "bg_read", read=32, **banks),
                        work("metadata", "sign_read", read=16),
                    )
                    + (
                        () if self.c.ideal_bsc else (work(f"bsc{j%2}", "load", load=1),)
                    ),
                    (load, signload, free),
                )
                self.mem.hold("input", c1)
                self.mem.hold("metadata", c1)
                c2 = self.s.issue(
                    "sss.prescan.C2",
                    "sss",
                    (
                        ()
                        if self.c.ideal_bsc
                        else (work(f"bsc{j%2}", "detect", detect=1),)
                    ),
                    (c1,),
                )
                self.bsc_free[j % 2] = c2
                self.stats["prescan_tiles"] += 1
            end = max([self.s.floor, *self.bsc_free], key=lambda x: x.end)
            written = self.mem.access(
                "metadata",
                coordbuf,
                base * per_entry,
                len(chunk) * per_entry,
                write=True,
                deps=(end,),
            )
            self.s.issue(
                "sss.mode_select",
                "sss",
                (work("sss", "select", control=1),),
                (written,),
            )
        self.s.barrier()
        self.mem.prune()
