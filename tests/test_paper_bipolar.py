"""Numerical, byte-conservation, scheduling and real quant-forward capture checks."""

import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_tcs2.hardware.paper.sparsity import (
    digits,
    digit_gemm,
    effective_mask,
    pack_int4,
    unpack_int4,
)
from vla_tcs2.hardware.paper.architecture import PaperConfig, paper_architecture
from vla_tcs2.hardware.paper.schedule import StreamingEventSimulator, work
from vla_tcs2.hardware.paper.memory import MemorySystem
from vla_tcs2.hardware.modeling.protocol import Action


class ArithmeticTests(unittest.TestCase):
    def test_exhaustive_signed_products(self):
        a = np.arange(-128, 128, dtype=np.int64).reshape(-1, 1)
        for bits in (4, 8):
            b = np.arange(-(1 << (bits - 1)), 1 << (bits - 1), dtype=np.int64).reshape(
                1, -1
            )
            np.testing.assert_array_equal(digit_gemm(a, b, b_bits=bits), a @ b)
        np.testing.assert_array_equal(
            sum(d * (4**lv) for lv, d in enumerate(digits(a))), a
        )

    def test_negative_lsb_and_joint_pack(self):
        a = np.zeros((8, 16), dtype=np.int8)
        a[3, 5] = -1
        self.assertTrue(effective_mask(a, 0, 8)[0, 0, 5])
        self.assertFalse(effective_mask(a, 1, 8).any())
        a[0, 7] = 4
        self.assertTrue(effective_mask(a, 1, 8)[0, 0, 7])
        self.assertEqual(int(effective_mask(a, 1, 64).sum()), 1)

    def test_random_gemm_tails(self):
        rng = np.random.default_rng(8)
        a = rng.integers(-128, 128, (11, 19))
        b = rng.integers(-8, 8, (19, 7))
        np.testing.assert_array_equal(digit_gemm(a, b, b_bits=4), a @ b)
        z = np.arange(-8, 8, dtype=np.int8)[:15]
        np.testing.assert_array_equal(unpack_int4(pack_int4(z), z.shape), z)


class TimingTests(unittest.TestCase):
    def test_gap_filling(self):
        s = StreamingEventSimulator(paper_architecture(PaperConfig()))
        long = s.issue(
            "latency", "memory", (Action("dram", "wait", {"fixed_cycles": 100}),)
        )
        late = s.issue(
            "late", "compute", (work("array", "compute", issue=10),), (long,)
        )
        early = s.issue("early", "compute", (work("array", "compute", issue=5),))
        self.assertEqual((late.end, early.end), (110, 5))

    def test_cache_and_burst_conservation(self):
        c = PaperConfig(memory_block_bytes=64)
        s = StreamingEventSimulator(paper_architecture(c))
        m = MemorySystem(s, c)
        b = m.buffer("x", 128, "weight")
        m.access("weight", b, 1, 3)
        m.access("weight", b, 2, 1)
        self.assertEqual(m.counts["read.weight.bus_bytes"], 64)
        self.assertEqual(m.counts["read.weight.payload_bytes"], 64)
        self.assertEqual(m.unique_bytes["weight"], 64)
        self.assertEqual(m.counts["weight.hit_bytes"], 1)

    def test_dma_latency_does_not_lock_bus(self):
        c = PaperConfig()
        s = StreamingEventSimulator(paper_architecture(c))
        m = MemorySystem(s, c)
        b = m.buffer("x", 2048, "weight")
        x = m.dma(b, 0, 1024, "read")
        y = m.dma(b, 1024, 1024, "read")
        service = __import__("math").ceil(1024 / (12e9 / 390e6))
        self.assertEqual(y.end, 39 + 2 * service)
        self.assertEqual(s.busy["dram.bus"], 2 * service)
        self.assertEqual(sum(y.critical.values()), y.end)

    def test_finite_cache_dirty_eviction(self):
        c = PaperConfig()
        s = StreamingEventSimulator(paper_architecture(c))
        m = MemorySystem(s, c)
        b = m.buffer("activation", c.input_bytes * 2, "activation")
        m.access("input", b, 0, b.size, write=True)
        self.assertLessEqual(m.peak["input"], c.input_bytes)
        self.assertGreater(m.counts["write.activation.bus_bytes"], 0)


def toy_capture(root, complete=False):
    import torch
    from torch import nn
    from vla_tcs2.quant_linear import QuantizedLinear
    from vla_tcs2.quant_matmul import QuantizedMatMul
    from vla_tcs2.hardware.paper.workload import WorkloadCapture
    from vla_tcs2.runtime_context import (
        CURRENT_PHASE,
        CURRENT_FLOW_STEP,
        CURRENT_GENERATION_ID,
    )

    class Transformer(nn.Module):
        def __init__(self, component="vlm"):
            super().__init__()
            self.fc = QuantizedLinear(
                16,
                8,
                bias=False,
                mode="quant_forward",
                a_bit=8,
                w_bit=4 if component == "expert" else 8,
                o_bit=8,
                scale_root_str=str(root / "scales"),
            )
            self.fc.set_layer_info("fc", 0, module_id=f"{component}.layer.0.fc")
            self.fc.a_interval = self.fc.w_interval = self.fc.o_interval = 0.125
            self.fc.save_scales()
            self.mm = QuantizedMatMul(
                mode="quant_forward",
                A_bit=8,
                B_bit=8,
                O_bit=8,
                scale_root_str=str(root / "scales"),
            )
            self.mm.set_layer_info("qk", 0, module_id=f"{component}.layer.0.qk")
            self.mm.A_interval = self.mm.B_interval = self.mm.O_interval = 0.125
            self.mm.save_scales()

        def forward(self, x):
            y = self.fc(x)
            q = self.mm(y, y.transpose(-2, -1))
            return torch.softmax(q * 0.25, dim=-1)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm_with_expert = Transformer()
            if complete:
                self.vlm_with_expert_expert = Transformer("expert")

        def forward(self, x):
            y = self.vlm_with_expert(x)
            if complete:
                token = CURRENT_PHASE.set("denoise")
                try:
                    for step in range(10):
                        st = CURRENT_FLOW_STEP.set(step)
                        try:
                            y = self.vlm_with_expert_expert(x)
                        finally:
                            CURRENT_FLOW_STEP.reset(st)
                finally:
                    CURRENT_PHASE.reset(token)
            return y

    torch.manual_seed(42)
    model = Model().eval()
    x = torch.randn(8, 16)
    tokens = [
        (CURRENT_PHASE, CURRENT_PHASE.set("prefill")),
        (CURRENT_FLOW_STEP, CURRENT_FLOW_STEP.set(-1)),
        (CURRENT_GENERATION_ID, CURRENT_GENERATION_ID.set(0)),
    ]
    try:
        with torch.no_grad():
            before = model(x)
            with WorkloadCapture(root / "workload")(model):
                after = model(x)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
    return root / "workload"


class IntegrationTests(unittest.TestCase):
    def test_real_quantized_capture_replay(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable; numerical and scheduler checks still run")
        from vla_tcs2.hardware.paper.workload import read_records, read_code
        from vla_tcs2.hardware.paper.replay import replay

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workload = toy_capture(root)
            recs = list(read_records(workload))
            gemms = [r for r in recs if r["kind"] == "gemm"]
            self.assertEqual(len(gemms), 2)
            self.assertTrue(any("softmax" in r.get("op", "") for r in recs))
            self.assertFalse(any("mm." in r.get("op", "") for r in recs))
            a = read_code(workload, gemms[0]["a"])
            self.assertEqual(a.dtype, np.int8)
            result = replay(workload, root / "replay", complete=False)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(
                sum(result["groups"]["vlm_prefill"]["critical"].values()),
                result["total_cycles"],
            )
            self.assertLessEqual(result["peak_sram_bytes"]["input"], 16384)
            with gzip.open(root / "replay/timeline/chrome_trace.json.gz", "rt") as f:
                json.load(f)
            with gzip.open(root / "replay/timeline/events.csv.gz", "rt") as f:
                rows = list(csv.DictReader(f))
            by_id = {int(r["id"]): r for r in rows}
            by_resource = {}
            for r in rows:
                for dep in json.loads(r["depends_on"]):
                    self.assertLessEqual(int(by_id[dep]["end"]), int(r["start"]))
                for resource in json.loads(r["resources"]):
                    by_resource.setdefault(resource, []).append(
                        (int(r["start"]), int(r["end"]))
                    )
            for intervals in by_resource.values():
                intervals.sort()
                self.assertTrue(
                    all(e <= s for (_, e), (s, _) in zip(intervals, intervals[1:]))
                )
            for group in ("D0", "D1", "D2"):
                other = replay(
                    workload,
                    root / group,
                    replace(PaperConfig(), group=group),
                    complete=False,
                    timeline=False,
                )
                self.assertEqual(
                    other["work"]["dense_macs"], result["work"]["dense_macs"]
                )


class MappingTests(unittest.TestCase):
    def test_sss_switch_and_tail_transactions_reconstruct_gemm(self):
        from vla_tcs2.hardware.paper.planner import PaperPlanner

        class Sink:
            def __init__(self):
                self.rows = []

            def write(self, row):
                self.rows.append(row)

        rng = np.random.default_rng(13)
        a = rng.integers(-8, 8, (67, 19), dtype=np.int8)
        b = rng.integers(-8, 8, (19, 41), dtype=np.int8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez_compressed(root / "a.npz", data=a)
            np.savez_compressed(root / "b.npz", data=pack_int4(b.T))
            desc = lambda ident, shape: {
                "id": ident,
                "storage_bytes": int(np.prod(shape)) * 4,
                "numel": int(np.prod(shape)),
                "element_bytes": 4,
                "shape": list(shape),
                "stride": [shape[-1], 1],
                "offset": 0,
                "external": ident == "input",
            }
            rec = {
                "id": 0,
                "linear": True,
                "module_id": "expert.fc",
                "component": "expert",
                "scales": [1.0, 1.0, 1.0],
                "a": {
                    "file": "a.npz",
                    "shape": list(a.shape),
                    "bits": 8,
                    "sha256": "a",
                },
                "b": {
                    "file": "b.npz",
                    "shape": list(b.T.shape),
                    "bits": 4,
                    "sha256": "b",
                },
                "inputs": [desc("input", a.shape)],
                "outputs": [desc("output", (67, 41))],
            }
            c = PaperConfig()
            sink = Sink()
            s = StreamingEventSimulator(paper_architecture(c), sink)
            mem = MemorySystem(s, c)
            planner = PaperPlanner(s, mem, c, root)
            planner.gemm(rec)
            s.barrier()
            result = np.zeros((67, 41), dtype=np.int64)
            ds = digits(a)
            modes = {}
            for row in sink.rows:
                if row["kind"] != "array.digit_run":
                    continue
                x = row["metadata"]
                ms, ns, lv = x["m"], x["n"], x["bg"]
                pm, bn = min(x["p"], 67 - ms), min(x["b"], 41 - ns)
                cols = np.asarray(x["effective_addresses"]) + x["k"]
                result[ms : ms + pm, ns : ns + bn] += (
                    ds[lv][ms : ms + pm][:, cols]
                    @ b[cols, ns : ns + bn].astype(np.int64)
                ) * (4**lv)
                modes[lv] = x["p"]
            np.testing.assert_array_equal(
                result, a.astype(np.int64) @ b.astype(np.int64)
            )
            self.assertEqual(modes[0], 64)
            self.assertTrue(any(r["mode"] == "sparse" for r in planner.sparsity_rows))
            self.assertGreater(mem.counts["read.partial_sum.bus_bytes"], 0)
            self.assertLessEqual(mem.peak["input.bank0"], c.input_bytes // 2)
            self.assertLessEqual(mem.peak["weight.bank1"], c.weight_bytes // 2)


class RegressionTests(unittest.TestCase):
    def test_staged_working_set_cannot_be_silently_evicted(self):
        c = PaperConfig()
        s = StreamingEventSimulator(paper_architecture(c))
        mem = MemorySystem(s, c)
        b = mem.buffer("large", 64 * 960, "activation")
        with self.assertRaisesRegex(ValueError, "working set"):
            for row in range(64):
                mem.access("input", b, row * 960, 16, service=False)

    def test_partial_write_preserves_existing_backing(self):
        c = PaperConfig(memory_block_bytes=64)
        s = StreamingEventSimulator(paper_architecture(c))
        mem = MemorySystem(s, c)
        existing = mem.buffer("external", 64, "activation")
        mem.access("input", existing, 0, 1, write=True)
        self.assertEqual(mem.counts["read.activation.payload_bytes"], 64)
        fresh = mem.buffer("fresh", 64, "activation", initialized=False)
        mem.access("input", fresh, 0, 1, write=True)
        self.assertEqual(mem.counts["read.activation.payload_bytes"], 64)

    def test_embedding_reads_selected_rows(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from torch import nn
        from vla_tcs2.hardware.paper.workload import WorkloadCapture, read_records
        from vla_tcs2.hardware.paper.planner import PaperPlanner
        from vla_tcs2.runtime_context import CURRENT_PHASE

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.vlm_with_expert = nn.Embedding(1000, 16)

            def forward(self, x):
                return self.vlm_with_expert(x)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = Model()
            token = CURRENT_PHASE.set("prefill")
            try:
                with torch.no_grad(), WorkloadCapture(root / "workload")(model):
                    model(torch.tensor([1, 999]))
            finally:
                CURRENT_PHASE.reset(token)
            rec = next(
                r
                for r in read_records(root / "workload")
                if "embedding" in r.get("op", "")
            )
            c = PaperConfig(memory_block_bytes=64)
            s = StreamingEventSimulator(paper_architecture(c))
            mem = MemorySystem(s, c)
            PaperPlanner(s, mem, c, root / "workload").auxiliary(rec)
            self.assertEqual(mem.counts["read.boundary.payload_bytes"], 128 + 16)
            self.assertEqual(mem.counts["read.boundary.bus_bytes"], 128 + 64)

    def test_large_stride_staging_and_broadcast_addresses(self):
        from vla_tcs2.hardware.paper.planner import PaperPlanner

        class Sink:
            def __init__(self):
                self.rows = []

            def write(self, row):
                if row["kind"] == "array.digit_run":
                    self.rows.append(row)

        def desc(ident, shape):
            strides = [int(np.prod(shape[i + 1 :])) for i in range(len(shape))]
            return dict(
                id=ident,
                shape=list(shape),
                stride=strides,
                numel=int(np.prod(shape)),
                storage_bytes=int(np.prod(shape)) * 4,
                element_bytes=4,
                offset=0,
                external=True,
            )

        cases = [
            (np.ones((64, 960), dtype=np.int8), np.ones((1, 960), dtype=np.int8), True),
            (
                np.ones((2, 1, 8, 3), dtype=np.int8),
                np.ones((1, 3, 3, 4), dtype=np.int8),
                False,
            ),
        ]
        for a, b, linear in cases:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                np.savez(root / "a.npz", data=a)
                np.savez(root / "b.npz", data=b)
                shape = (64, 1) if linear else (2, 3, 8, 4)
                rec = dict(
                    id=0,
                    linear=linear,
                    module_id="vlm.fc",
                    component="vlm",
                    scales=[1, 1, 1],
                    a=dict(file="a.npz", sha256="a", shape=list(a.shape), bits=8),
                    b=dict(file="b.npz", sha256="b", shape=list(b.shape), bits=8),
                    inputs=[desc("a", a.shape)]
                    + ([] if linear else [desc("b", b.shape)]),
                    outputs=[desc("out", shape)],
                )
                c = PaperConfig(group="D0")
                sink = Sink()
                sim = StreamingEventSimulator(paper_architecture(c), sink)
                mem = MemorySystem(sim, c)
                PaperPlanner(sim, mem, c, root).gemm(rec)
                self.assertTrue(all(not pins for pins in mem.pinned.values()))
                self.assertLessEqual(mem.peak["input.bank0"], 8192)
                if not linear:
                    batches = {
                        r["metadata"]["batch"]: (
                            r["metadata"]["input_batch"],
                            r["metadata"]["weight_batch"],
                        )
                        for r in sink.rows
                    }
                    self.assertEqual(batches, {i: (i // 3, i % 3) for i in range(6)})

    def test_full_phase_accounting(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from collections import Counter
        from vla_tcs2.hardware.paper.replay import replay, validate_workload

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workload = toy_capture(root, complete=True)
            validate_workload(workload)
            result = replay(workload, root / "replay", timeline=False)
            phases = [
                g
                for name, g in result["groups"].items()
                if name != "expert_denoise_total"
            ]
            self.assertEqual(sum(g["cycles"] for g in phases), result["total_cycles"])
            busy = Counter()
            for g in phases:
                busy.update(g["busy"])
            self.assertEqual(dict(busy), result["resource_busy_cycles"])
            self.assertEqual(len(phases), 11)


if __name__ == "__main__":
    unittest.main()
