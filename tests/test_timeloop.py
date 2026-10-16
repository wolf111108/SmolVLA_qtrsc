"""Mapping traffic, constrained optimality and automatic persistence contracts."""
import csv
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_tcs2.hardware import HardwareManager, HardwareSpec, MappingSpec, OperatorEvent, TensorDesc
from vla_tcs2.hardware.backends.dense import DenseAnalytic
from vla_tcs2.hardware.timeloop import DATAFLOWS, SearchSpec, TimeloopMapper


def event(m=3, k=7, n=5, bits=8, output_bits=8, **overrides):
    values = dict(event_id=0, module_id="expert.fc", component="expert", op_type="linear",
                  a=TensorDesc((m, k), "logical", "logical", bits),
                  b=TensorDesc((n, k), "logical", "logical", bits),
                  output=TensorDesc((m, n), "logical", "logical", output_bits), m=m, n=n, k=k)
    values.update(overrides)
    return OperatorEvent(**values)


class TrafficTests(unittest.TestCase):
    def test_three_flows_and_accumulator_spills_by_hand(self):
        h = HardwareSpec(pe_rows=2, pe_cols=4, frequency_hz=100, dram_bytes_per_second=10)
        expected = {
            "output_stationary": (42, 70, 0, 0, 112, 15),
            "weight_stationary": (42, 35, 60, 60, 137, 75),
            "input_stationary": (21, 70, 60, 60, 151, 75),
        }
        for flow, traffic in expected.items():
            with self.subTest(flow=flow):
                r = DenseAnalytic().estimate(event(), h, MappingSpec(2, 4, 4, flow))
                self.assertEqual((r.activation_read_bytes, r.weight_read_bytes, r.partial_sum_read_bytes,
                                  r.partial_sum_write_bytes, r.dram_read_bytes, r.dram_write_bytes), traffic)
                self.assertEqual((r.compute_cycles, r.peak_buffer_bytes), (28, 56))
                self.assertAlmostEqual(r.memory_seconds, (traffic[-2] + traffic[-1]) / 10)

    def test_previous_large_example(self):
        e = event(256, 1024, 4096, bits=16, output_bits=32)
        h = HardwareSpec(sram_bytes=1024 * 1024)
        for t, mib in ((MappingSpec(64, 256, 1024, "weight_stationary"), 20),
                       (MappingSpec(64, 256, 1024, "input_stationary"), 36.5),
                       (MappingSpec(256, 512, 128), 16)):
            r = DenseAnalytic().estimate(e, h, t)
            self.assertEqual(r.peak_buffer_bytes, 704 * 1024)
            self.assertEqual(r.dram_read_bytes + r.dram_write_bytes, mib * 1024 * 1024)

    def test_no_spill_when_output_reduction_is_contiguous(self):
        for t in (MappingSpec(3, 4, 4, "weight_stationary"),
                  MappingSpec(2, 5, 4, "input_stationary")):
            r = DenseAnalytic().estimate(event(), HardwareSpec(), t)
            self.assertEqual(r.partial_sum_read_bytes, 0)
            self.assertEqual(r.dram_write_bytes, 15)

    def test_subbyte_tail_packing_and_batched_replication(self):
        e = event(3, 3, 3, bits=3, output_bits=3)
        # Transfer sizes: M/N/K groups are [2,1]; sum of packed
        # tile bytes per matrix is ceil(12/8)+ceil(6/8)*2+ceil(3/8)=5.
        # OS rereads each matrix for two output blocks: 10+10 bytes.
        r = DenseAnalytic().estimate(e, HardwareSpec(), MappingSpec(2, 2, 2))
        self.assertEqual((r.activation_read_bytes, r.weight_read_bytes, r.dram_write_bytes), (10, 10, 5))
        b = event(3, 3, 3, bits=3, output_bits=3, op_type="matmul", batch=2, batch_shape=(2,),
                  a=TensorDesc((2, 3, 3), "logical", "logical", 3),
                  b=TensorDesc((1, 3, 3), "logical", "logical", 3),
                  output=TensorDesc((2, 3, 3), "logical", "logical", 3))
        for flow in DATAFLOWS:
            t = MappingSpec(2, 2, 2, flow)
            one = DenseAnalytic().estimate(e, HardwareSpec(), t)
            two = DenseAnalytic().estimate(b, HardwareSpec(), t)
            self.assertEqual(two.dram_read_bytes, 2 * one.dram_read_bytes)
            self.assertEqual(two.dram_write_bytes, 2 * one.dram_write_bytes)

    def test_shared_and_independent_bandwidth(self):
        h = HardwareSpec(dram_bytes_per_second=10, activation_read_bytes_per_second=2,
                         weight_read_bytes_per_second=5)
        t = MappingSpec(2, 4, 4)
        r = DenseAnalytic().estimate(event(), h, t)
        self.assertAlmostEqual(r.memory_seconds, 42/2 + 70/5 + 15/10)
        r = DenseAnalytic().estimate(event(), replace(h, dram_mode="independent"), t)
        self.assertEqual(r.memory_seconds, 21)


class SearchTests(unittest.TestCase):
    def test_all_candidates_match_independent_brute_force_minimum(self):
        e = event()
        h = HardwareSpec(sram_bytes=32, pipeline_cycles=2)
        for objective in ("memory_time", "latency", "dram_bytes"):
            mapper = TimeloopMapper(h, SearchSpec(candidate_policy="all", objective=objective))
            result = mapper.map_event(e)
            costs = []
            for m, n, k, flow in itertools.product(range(1, 4), range(1, 6), range(1, 8), DATAFLOWS):
                r = DenseAnalytic().estimate(e, h, MappingSpec(m, n, k, flow))
                if r.status == "estimated":
                    costs.append(r.memory_seconds if objective == "memory_time" else r.latency_seconds
                                 if objective == "latency" else r.dram_read_bytes + r.dram_write_bytes)
            self.assertEqual(result.objective_value, min(costs))
            self.assertEqual(result.feasible_candidates, len(costs))
            self.assertEqual(result.evaluated_candidates + result.capacity_pruned_candidates, 3*5*7*3)

    def test_bandwidth_changes_selected_tile(self):
        s = SearchSpec(tile_m_candidates=(4, 8, 16), tile_n_candidates=(4, 8, 16),
                       tile_k_candidates=(1,), dataflows=("output_stationary",))
        base = HardwareSpec(sram_bytes=512, dram_bytes_per_second=1000)
        weight_slow = TimeloopMapper(replace(base, weight_read_bytes_per_second=1), s).map_event(event(64, 64, 64))
        activation_slow = TimeloopMapper(replace(base, activation_read_bytes_per_second=1), s).map_event(event(64, 64, 64))
        self.assertEqual((weight_slow.mapping.tile_m, weight_slow.mapping.tile_n), (16, 4))
        self.assertEqual((activation_slow.mapping.tile_m, activation_slow.mapping.tile_n), (4, 16))

    def test_cache_rebinds_identity_but_invalidates_workload_and_hardware(self):
        mapper = TimeloopMapper()
        first = mapper.map_event(event())
        second = mapper.map_event(event(event_id=1, module_id="expert.other", flow_step=9))
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.estimate.event_id, 1)
        self.assertEqual(second.mapping, first.mapping)
        self.assertFalse(mapper.map_event(event(bits=16)).cache_hit)
        mapper.hardware = replace(mapper.hardware, sram_bytes=64)
        self.assertFalse(mapper.map_event(event()).cache_hit)
        mapper.search = replace(mapper.search, objective="latency")
        self.assertFalse(mapper.map_event(event()).cache_hit)

    def test_overflow_unsupported_paths_and_empty_candidates(self):
        for mapper in (TimeloopMapper(HardwareSpec(sram_bytes=1)),
                       TimeloopMapper(search=SearchSpec(tile_m_candidates=(999,)))):
            r = mapper.map_event(event())
            self.assertIsNone(r.mapping)
            self.assertEqual(r.estimate.status, "unsupported")
            self.assertIsNone(r.objective_value)
        for e in (event(mode="quant_forward", method="outlier"), event(metadata={"is_bitnet": True})):
            r = TimeloopMapper().map_event(e)
            self.assertEqual(r.evaluated_candidates, 0)
            self.assertEqual(r.estimate.status, "unsupported")

    def test_double_buffer_capacity_and_alignment(self):
        # 2x4 output: 32 bytes, A+B: 24 bytes, total 56 or 80 double buffered.
        s = SearchSpec(tile_m_candidates=(2,), tile_n_candidates=(4,), tile_k_candidates=(4,))
        self.assertIsNotNone(TimeloopMapper(HardwareSpec(sram_bytes=56), s).map_event(event()).mapping)
        self.assertIsNone(TimeloopMapper(HardwareSpec(sram_bytes=56, overlap_compute_memory=True), s)
                          .map_event(event()).mapping)
        r = TimeloopMapper(search=SearchSpec(candidate_policy="all", alignment_m=2,
                                             alignment_n=4, alignment_k=4)).map_event(event())
        self.assertEqual(r.candidate_space["tiles"], {"m": (2, 3), "n": (4, 5), "k": (4, 7)})

    def test_invalid_configuration(self):
        for kwargs in ({"enabled": "yes"}, {"objective": "missing"}, {"candidate_policy": "missing"},
                       {"tile_m_candidates": []}, {"tile_k_candidates": [True]},
                       {"alignment_n": 0}, {"dataflows": []}, {"dataflows": ["missing"]}):
            with self.assertRaises(ValueError): SearchSpec(**kwargs)
        for kwargs in ({"dram_mode": "missing"}, {"dram_mode": "independent"},
                       {"activation_read_bytes_per_second": float("nan")},
                       {"weight_read_bytes_per_second": True}):
            with self.assertRaises(ValueError): HardwareSpec(**kwargs)

    def test_direct_api_saves_reviewable_inputs_and_result(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mapping.json"
            r = TimeloopMapper().map_gemm(3, 7, 5, save_path=path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["engine"], "project_analytic")
            self.assertEqual(saved["mapping"]["tile_m"], r.mapping.tile_m)
            self.assertEqual(saved["event"]["k"], 7)
            with self.assertRaises(FileExistsError):
                TimeloopMapper().map_gemm(3, 7, 5, save_path=path)


class PersistenceTests(unittest.TestCase):
    def test_manager_persists_per_event_and_replay_inherits_search(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "src", Path(d) / "dst"
            with HardwareManager(src, mapping_search={"enabled": True}, hardware={"sram_bytes": 64}) as hw:
                hw.submit(event())
                hw.submit(event(event_id=1, module_id="expert.other", flow_step=1))
                hw.submit(event(event_id=2, mode="quant_forward", method="outlier"))
            records = [json.loads(line) for line in (src / "mappings.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertEqual(records[1]["module_id"], "expert.other")
            self.assertEqual(records[1]["estimate"]["event_id"], 1)
            self.assertTrue(records[1]["cache_hit"])
            self.assertIsNone(records[2]["mapping"])
            with (src / "operators.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["dataflow"], records[0]["mapping"]["dataflow"])
            self.assertEqual(rows[0]["memory_seconds"], str(records[0]["estimate"]["memory_seconds"]))
            self.assertEqual(rows[2]["tile_m"], "")
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            subprocess.run([sys.executable, "-m", "vla_tcs2.hardware", "--trace", str(src / "trace.jsonl"),
                            "--output-dir", str(dst)], check=True, env=env, capture_output=True)
            summaries = [json.loads((p / "summary.json").read_text()) for p in (src, dst)]
            self.assertEqual(summaries[0]["totals"], summaries[1]["totals"])
            self.assertEqual(summaries[0]["mapping_search"], summaries[1]["mapping_search"])
            self.assertEqual((src / "mappings.jsonl").read_text(), (dst / "mappings.jsonl").read_text())

    def test_mapping_output_is_not_overwritten_and_streams_close_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mappings.jsonl"
            path.write_text("existing")
            with self.assertRaises(FileExistsError):
                with HardwareManager(d, mapping_search={"enabled": True}): pass
            self.assertEqual(path.read_text(), "existing")
        with tempfile.TemporaryDirectory() as d:
            hw = HardwareManager(d, mapping_search={"enabled": True})
            with self.assertRaises(RuntimeError):
                with hw:
                    hw.submit(event())
                    raise RuntimeError("failure")
            self.assertTrue(all(s.closed for s in hw._streams))
            self.assertEqual(len((Path(d) / "mappings.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
