#!/usr/bin/env python3
"""Deterministic exporter checks; never loads a model or downloads a dataset."""
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))
from export_timeline import export
from vla_tcs2.hardware import HardwareManager, OperatorEvent, TensorDesc
from vla_tcs2.hardware.modeling import default_architecture


class ExportChecks(unittest.TestCase):
    def test_global_time_service_work_and_plot(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = default_architecture().config
            cfg["clock_hz"] = 1000
            cfg["components"]["dram"].update(bandwidth_bytes_per_second=1e6, efficiency=1, latency_cycles=0, burst_bytes=1)
            cfg["components"]["array"].update(rows=2, cols=2)
            for name in ("gb", "local", "acc"):
                cfg["components"][name].update(read_bytes_per_cycle=1000, write_bytes_per_cycle=1000)
            for name in ("dma", "noc"):
                cfg["components"][name]["bytes_per_cycle"] = 1000
            with HardwareManager(root / "hardware", max_events=0, backend_config={"architecture": cfg},
                                 mapping={"tile_m": 2, "tile_n": 2, "tile_k": 2}) as manager:
                for i in range(2):
                    manager.submit(OperatorEvent(i, f"expert.layer.{i}.q_proj", "expert", "linear",
                        TensorDesc((2, 4), "float32", "int8", 8), TensorDesc((2, 4), "float32", "int8", 8),
                        TensorDesc((2, 2), "float32", "int8", 8), 2, 2, 4,
                        phase="denoise", flow_step=i, generation_id=0))
            result = export(root / "hardware")
            self.assertEqual(result["total_cycles"], 18)
            self.assertEqual(result["operators"], 2)
            with gzip.open(root / "timeline/events.csv.gz", "rt") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(min(int(x["start_cycle"]) for x in rows if x["operator_id"] == "1"), 9)
            with (root / "timeline/resources.csv").open() as stream:
                resources = {r["resource"]: r for r in csv.DictReader(stream)}
            self.assertEqual(int(resources["local.read"]["reserved_cycles"]), 8)
            self.assertEqual(int(resources["local.read"]["service_cycles"]), 4)
            with gzip.open(root / "timeline/chrome_trace.json.gz", "rt") as stream:
                trace = json.load(stream)
            slices = [e for e in trace["traceEvents"] if e["ph"] == "X"]
            self.assertTrue(all(e["ts"] == e["args"]["start_cycle"] * 1000 for e in slices))
            for tid in {e["tid"] for e in slices}:
                intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in slices if e["tid"] == tid)
                self.assertTrue(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])))
            with self.assertRaises(FileExistsError):
                export(root / "hardware")
            subprocess.run([sys.executable, str(SCRIPTS / "plot_timeline.py"), "--timeline-dir", str(root / "timeline"),
                            "--bins", "5"], check=True)
            import numpy as np
            with np.load(root / "timeline/occupancy_bins.npz") as bins:
                for resource, values in zip(bins["resources"], bins["reserved_fraction"]):
                    self.assertAlmostEqual(values.sum() * 18 / 5, int(resources[resource]["reserved_cycles"]))
            self.assertTrue((root / "timeline/figures/detail_expert.png").exists())


if __name__ == "__main__":
    unittest.main()
