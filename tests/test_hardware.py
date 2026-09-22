"""Hand-calculated hardware contracts; run without torch via unittest discovery."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_tcs2.hardware import HardwareManager, HardwareSpec, MappingSpec, OperatorEvent, TensorDesc
from vla_tcs2.hardware.backends.dense import DenseAnalytic
from vla_tcs2.hardware.capture import make_event, describe
from vla_tcs2.hardware.trace import read_trace


def event(index=0, **overrides):
    values = dict(event_id=index, module_id="vision.layer.0.fc1", component="vision", op_type="linear",
                  a=TensorDesc((3, 7), "float32", "int8", 8),
                  b=TensorDesc((5, 7), "float32", "int8", 8),
                  output=TensorDesc((3, 5), "float32", "int8", 8), m=3, n=5, k=7)
    values.update(overrides)
    return OperatorEvent(**values)


class FakeTensor:
    dtype = "torch.float32"
    def __init__(self, shape): self.shape = shape
    def element_size(self): return 4


class HardwareTests(unittest.TestCase):
    def test_tail_tiles_by_hand(self):
        e = event()
        h = HardwareSpec(pe_rows=2, pe_cols=4, frequency_hz=100, dram_bytes_per_second=10)
        t = MappingSpec(tile_m=2, tile_n=4, tile_k=4)
        r = DenseAnalytic().estimate(e, h, t)
        self.assertEqual((r.dense_macs, r.compute_cycles), (105, 28))
        self.assertEqual((r.dram_read_bytes, r.dram_write_bytes), (112, 15))
        self.assertEqual(r.peak_buffer_bytes, 56)
        self.assertAlmostEqual(r.latency_seconds, .28 + 12.7)
        self.assertAlmostEqual(r.pe_utilization, 105 / 224)

    def test_overlap_double_buffers(self):
        r = DenseAnalytic().estimate(event(), HardwareSpec(pe_rows=2, pe_cols=4,
            frequency_hz=100, dram_bytes_per_second=10, overlap_compute_memory=True),
            MappingSpec(tile_m=2, tile_n=4, tile_k=4))
        self.assertEqual(r.peak_buffer_bytes, 80)
        self.assertEqual(r.latency_seconds, 12.7)

    def test_sram_overflow_is_not_zero_cost(self):
        r = DenseAnalytic().estimate(event(), HardwareSpec(sram_bytes=1), MappingSpec())
        self.assertEqual(r.status, "unsupported")
        self.assertIsNone(r.latency_seconds)

    def test_outlier_and_mixed_paths_have_no_fake_latency(self):
        for e in (event(mode="quant_forward", method="pot_fp8_outlier"),
                  event(metadata={"mixed_precision": True}), event(metadata={"is_bitnet": True}),
                  event(mode="test_forward")):
            r = DenseAnalytic().estimate(e, HardwareSpec(), MappingSpec())
            self.assertEqual(r.dense_macs, 105)
            self.assertEqual(r.status, "unsupported")
            self.assertIsNone(r.latency_seconds)

    def test_quantized_per_tensor_core(self):
        r = DenseAnalytic().estimate(event(mode="quant_forward", method="per_tensor"),
                                    HardwareSpec(), MappingSpec())
        self.assertEqual(r.status, "estimated")

    def test_bad_specs(self):
        for config in ({"frequency_hz": 0}, {"frequency_hz": float("nan")},
                       {"dram_bytes_per_second": float("inf")}, {"pe_rows": True},
                       {"pipeline_cycles": -1}, {"overlap_compute_memory": "false"}):
            with self.assertRaises(ValueError): HardwareSpec(**config)
        with self.assertRaises(ValueError): MappingSpec(tile_k=0)
        with self.assertRaises(ValueError): MappingSpec(dataflow="unknown")

    def test_schema_roundtrip_and_version(self):
        self.assertEqual(OperatorEvent.from_dict(json.loads(json.dumps(event().to_dict()))), event())
        for version in (None, 2):
            data = event().to_dict()
            data["schema_version"] = version
            with self.assertRaises(ValueError): OperatorEvent.from_dict(data)
        with self.assertRaises(ValueError): event(k=9)
        with self.assertRaises(ValueError): TensorDesc((0, 3), "float32", "int8", 8)

    def test_storage_and_logical_dtype_differ(self):
        desc = describe(FakeTensor((3, 7)), SimpleNamespace(enabled=True, kind="int", bits=4))
        self.assertEqual((desc.storage_dtype, desc.logical_format, desc.bits), ("float32", "int4", 4))
        desc = describe(FakeTensor((3, 7)), SimpleNamespace(enabled=False, kind="fp", fmt="e5m10"))
        self.assertEqual(desc.bits, 32)  # Actual pass-through dtype, not nominal disabled spec.

    def test_broadcast_matmul_not_linear_flatten(self):
        e = make_event(0, "expert.layer.0.qk", "matmul", SimpleNamespace(mode="raw"),
            (FakeTensor((2, 1, 3, 7)), FakeTensor((1, 4, 7, 5))), {}, FakeTensor((2, 4, 3, 5)),
            {"generation_id": 8, "flow_step": 2, "phase": "denoise"})
        self.assertEqual((e.batch, e.m, e.n, e.k, e.macs), (8, 3, 5, 7, 840))
        self.assertEqual(e.batch_shape, (2, 4))
        self.assertEqual(e.flow_step, 2)
        self.assertEqual(e.generation_id, 8)

    def test_keyword_linear_adapter(self):
        layer = SimpleNamespace(weight=FakeTensor((5, 7)), bias=None, mode="raw")
        e = make_event(0, "vision.layer.0.fc1", "linear", layer, (),
                       {"input": FakeTensor((2, 3, 7))}, FakeTensor((2, 3, 5)), {})
        self.assertEqual((e.m, e.n, e.k), (6, 5, 7))

    def test_invalid_broadcast_and_output(self):
        with self.assertRaises(ValueError):
            make_event(0, "mm", "matmul", SimpleNamespace(),
                       (FakeTensor((2, 3, 7)), FakeTensor((4, 7, 5))), {}, FakeTensor((4, 3, 5)), {})
        with self.assertRaises(ValueError): event(output=TensorDesc((3, 6), "float32", "int8", 8))

    def test_stream_limits_summary_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            with HardwareManager(d, max_events=2) as hw:
                hw.submit(event(0, generation_id=1))
                hw.submit(event(1, generation_id=1, mode="quant_forward", method="outlier"))
                hw.submit(event(2))
            summary = json.loads((Path(d) / "summary.json").read_text())
            self.assertEqual(summary["totals"]["dense_macs"], 210)
            self.assertEqual(summary["totals"]["unsupported_calls"], 1)
            self.assertEqual(summary["totals"]["estimated_core_macs"], 105)
            self.assertEqual(summary["captured_generation_count"], 1)
            self.assertTrue(summary["capture_truncated"])
            self.assertEqual(len(list(read_trace(Path(d) / "trace.jsonl"))), 3)
            with self.assertRaises(FileExistsError):
                with HardwareManager(d): pass

    def test_failure_reports_and_closes(self):
        with tempfile.TemporaryDirectory() as d:
            hw = HardwareManager(d)
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                with hw:
                    hw.submit(event())
                    raise RuntimeError("deliberate")
            self.assertTrue(all(s.closed for s in hw._streams))
            self.assertEqual(json.loads((Path(d) / "summary.json").read_text())["status"], "failed")
            with self.assertRaises(RuntimeError): hw.submit(event(1))

    def test_offline_replay_same_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "src", Path(d) / "dst"
            with HardwareManager(src, hardware={"pe_rows": 3}, mapping={"tile_k": 2}) as hw:
                hw.submit(event())
                hw.submit(event(1, mode="quant_forward", method="outlier"))
            import os
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            subprocess.run([sys.executable, "-m", "vla_tcs2.hardware", "--trace", str(src / "trace.jsonl"),
                            "--output-dir", str(dst)], check=True, env=env, capture_output=True)
            a, b = [json.loads((p / "summary.json").read_text()) for p in (src, dst)]
            self.assertEqual(a["totals"], b["totals"])

    def test_disabled_and_unknown_backend(self):
        self.assertIsNone(HardwareManager.from_config({}, "unused"))
        with self.assertRaises(ValueError): HardwareManager("unused", backend="missing")
        with self.assertRaises(ValueError): HardwareManager.from_config({"enabled": "false"}, "unused")


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable; run these integration tests in smolvla_eval")
class TorchCaptureTests(unittest.TestCase):
    def test_plain_linear_transparency_and_cleanup(self):
        from vla_tcs2.runtime_context import CURRENT_GENERATION_ID
        layer = torch.nn.Linear(7, 5)
        x = torch.randn(3, 7)
        expected = layer(x)
        token = CURRENT_GENERATION_ID.set(9)
        try:
            with tempfile.TemporaryDirectory() as d:
                with HardwareManager(d).capture_model(layer) as hw:
                    actual = layer(input=x)
                    self.assertTrue(torch.equal(expected, actual))
                    self.assertEqual(hw.totals["calls"], 1)
                    self.assertEqual(hw.generations, {9})
                self.assertEqual(len(layer._forward_hooks), 0)
        finally:
            CURRENT_GENERATION_ID.reset(token)

    def test_quant_linear_and_matmul_modes(self):
        from vla_tcs2.quant_linear import QuantizedLinear
        from vla_tcs2.quant_matmul import QuantizedMatMul
        class Pair(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = QuantizedLinear(7, 5, bias=False, a_bit=8, w_bit=8, o_bit=8)
                self.mm = QuantizedMatMul(A_bit=8, B_bit=8, O_bit=8)
        model = Pair()
        x, a, b = torch.randn(3, 7), torch.randn(2, 3, 7), torch.randn(2, 7, 5)
        with tempfile.TemporaryDirectory() as d:
            with HardwareManager(d).capture_model(model) as hw:
                torch.testing.assert_close(model.linear(x), torch.nn.functional.linear(x, model.linear.weight))
                torch.testing.assert_close(model.mm(a, b), a @ b)
                self.assertEqual(hw.totals["dense_macs"], 315)
                for layer in (model.linear, model.mm):
                    layer.mode = "quant_forward"
                    layer._load_scales = lambda: None
                model.linear.a_interval = model.linear.w_interval = model.linear.o_interval = .1
                model.mm.A_interval = model.mm.B_interval = model.mm.O_interval = .1
                model.linear(x)
                model.mm(a, b)
                self.assertEqual(hw.totals["calls"], 4)
                self.assertEqual(hw.totals["unsupported_calls"], 0)

    def test_cleanup_on_model_error(self):
        layer = torch.nn.Linear(7, 5)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                with HardwareManager(d).capture_model(layer):
                    layer(torch.randn(3, 6))
            self.assertEqual(len(layer._forward_hooks), 0)
            self.assertEqual(json.loads((Path(d) / "summary.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
