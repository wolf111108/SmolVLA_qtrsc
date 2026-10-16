"""Hand-checkable component protocol, flow conservation and resource tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_tcs2.hardware import HardwareManager, HardwareSpec, MappingSpec, OperatorEvent, TensorDesc
from vla_tcs2.hardware.modeling import (
    Action, ActionModel, Architecture, ComponentMapper, EventSimulator, ExecutionPlan,
    MatmulPlanner, Phase, PlanEvent, ResourceDemand, ResourceSpec, RuntimeStats,
    UnsupportedModel, default_architecture, register_component)
from vla_tcs2.hardware.timeloop import SearchSpec
from vla_tcs2.hardware.trace import read_trace


def op(m=2, n=2, k=2, **extra):
    return OperatorEvent(event_id=0, module_id="expert.fc", component="expert", op_type="linear",
                         a=TensorDesc((m, k), "float32", "int8", 8),
                         b=TensorDesc((n, k), "float32", "int8", 8),
                         output=TensorDesc((m, n), "float32", "int8", 8), m=m, n=n, k=k, **extra)


def arch(**changes):
    config = default_architecture(HardwareSpec(pe_rows=2, pe_cols=2, frequency_hz=1000,
                                  dram_bytes_per_second=1000000)).config
    for name in ("gb", "local", "acc"):
        config["components"][name].update(read_bytes_per_cycle=1000, write_bytes_per_cycle=1000)
    for name in ("dma", "noc"):
        config["components"][name]["bytes_per_cycle"] = 1000
    config["components"]["array"].update(changes)
    return Architecture(config)


class ComponentTests(unittest.TestCase):
    def test_hand_timeline_single_tile(self):
        a = arch()
        p = MatmulPlanner().lower(op(), MappingSpec(2, 2, 2), a)
        r = EventSimulator().run(p, a)
        self.assertEqual(r.total_cycles, 7)
        self.assertEqual((r.counters['dram.read_bytes'], r.counters['dram.write_bytes']), (8, 4))
        self.assertEqual(r.counters['array.dense_macs'], 8)
        self.assertEqual(r.counters['acc.read_bytes'], 16)  # drain only
        self.assertEqual(r.counters['acc.write_bytes'], 16)
        self.assertEqual(r.reservations, {'gb': 20, 'local': 16, 'acc': 16})
        self.assertEqual(r, EventSimulator().run(p, a))  # fresh simulator state

    def test_ping_pong_improves_overlap_with_same_traffic(self):
        a, e, t = arch(), op(k=4), MappingSpec(2, 2, 2)
        single = EventSimulator().run(MatmulPlanner(double_buffer=False).lower(e, t, a), a)
        double = EventSimulator().run(MatmulPlanner(double_buffer=True).lower(e, t, a), a)
        self.assertEqual((single.total_cycles, double.total_cycles), (12, 9))
        self.assertEqual(single.counters, double.counters)
        self.assertGreater(double.reservations['gb'], single.reservations['gb'])

    def test_tails_and_conservation(self):
        a, e = arch(cols=4), op(3, 5, 7)
        p = MatmulPlanner().lower(e, MappingSpec(2, 4, 4), a)
        r = EventSimulator().run(p, a)
        self.assertEqual((r.counters['array.dense_macs'], r.counters['array.compute_cycles']), (105, 28))
        self.assertEqual((r.counters['dram.read_bytes'], r.counters['dram.write_bytes']), (112, 15))
        self.assertEqual(r.counters['local.write_bytes'], 112)
        self.assertEqual(r.counters['local.read_bytes'], 112)
        self.assertEqual(r.counters['gb.read_bytes'], 127)
        for resource in a.resources:
            uses = sorted((t.start_cycle, t.end_cycle) for t in r.timeline if resource in t.resources)
            self.assertTrue(all(end <= start for (_, end), (start, _) in zip(uses, uses[1:])))
        ends = {t.event_id: t.end_cycle for t in r.timeline}
        for event in p.events:
            start = next(t.start_cycle for t in r.timeline if t.event_id == event.id)
            self.assertTrue(all(ends[dep] <= start for dep in event.depends_on))

    def test_shared_dram_reads_and_writes_serialize(self):
        a = arch()
        config = a.config
        config['components']['dram']['bandwidth_bytes_per_second'] = 1000
        a = Architecture(config)
        p = ExecutionPlan((PlanEvent('read', (Action('dram', 'read', {'bytes': 5}),)),
                           PlanEvent('write', (Action('dram', 'write', {'bytes': 7}),))))
        r = EventSimulator().run(p, a)
        self.assertEqual(r.total_cycles, 12)
        self.assertEqual(r.resource_busy_cycles['dram.bus'], 12)

    def test_ready_events_do_not_reserve_future_resources(self):
        a = arch()
        p = ExecutionPlan((
            PlanEvent('late', (Action('dma', 'transfer', {'bytes': 1000}),), ('long',)),
            PlanEvent('long', (Action('noc', 'transfer', {'bytes': 10000}),)),
            PlanEvent('now', (Action('dma', 'transfer', {'bytes': 1000}),))))
        r = EventSimulator().run(p, a)
        times = {t.event_id: (t.start_cycle, t.end_cycle) for t in r.timeline}
        self.assertEqual(times, {'long': (0, 10), 'now': (0, 1), 'late': (10, 11)})

    def test_shared_demands_sum_not_max(self):
        a = arch()
        p = ExecutionPlan((PlanEvent('both', (Action('dram', 'read', {'bytes': 1000}),
                                                Action('dram', 'write', {'bytes': 2000}))),))
        self.assertEqual(EventSimulator().run(p, a).total_cycles, 3)

    def test_multiphase_plugin(self):
        class Delayed:
            def __init__(self, name): self.id = name
            def ports(self): return ()
            def resources(self): return (ResourceSpec(self.id + '.r', 'bytes', 2),)
            def validate(self, action): pass
            def lower(self, action):
                return ActionModel((Phase(fixed_cycles=3), Phase((ResourceDemand(self.id+'.r', 4, 'bytes'),))))
        register_component('test_delayed', lambda name, params, clock: Delayed(name))
        c = arch().config
        c['components']['test'] = {'type': 'test_delayed'}
        a = Architecture(c)
        p = ExecutionPlan((PlanEvent('test', (Action('test', 'run'),)),))
        r = EventSimulator().run(p, a)
        self.assertEqual(r.total_cycles, 5)
        self.assertEqual(r.resource_busy_cycles['test.r'], 2)
        with self.assertRaises(ValueError): register_component('test_delayed', lambda: None)

    def test_bad_plans_and_architectures_fail(self):
        a = arch()
        action = Action('dma', 'transfer', {'bytes': 1})
        for events in ((PlanEvent('a', (action,), ('missing',)),),
                       (PlanEvent('a', (action,), ('b',)), PlanEvent('b', (action,), ('a',))),
                       (PlanEvent('a', (action,)), PlanEvent('a', (action,)))):
            with self.assertRaises(ValueError): EventSimulator().run(ExecutionPlan(events), a)
        c = a.config
        c['connections'][0]['to'] = 'gb.missing'
        with self.assertRaises(ValueError): Architecture(c)
        with self.assertRaises(ValueError):
            EventSimulator().run(ExecutionPlan((PlanEvent('a', (Action('dma', 'transfer', {'bytes': -1}),)),)), a)
        with self.assertRaises(ValueError): RuntimeStats(0, activation_zero_fraction=float('nan'))
        with self.assertRaises(ValueError): RuntimeStats(0, allow_uniform_estimate='yes')

    def test_capacity_event_budget_and_formats(self):
        a = arch()
        c = a.config
        c['components']['local']['capacity_bytes'] = 15
        with self.assertRaises(UnsupportedModel): MatmulPlanner().lower(op(), MappingSpec(2, 2, 2), Architecture(c))
        with self.assertRaises(UnsupportedModel): MatmulPlanner(max_plan_events=1).lower(op(), MappingSpec(), arch())
        with self.assertRaises(UnsupportedModel): MatmulPlanner().lower(op(), MappingSpec(), arch(supported_formats=['e4m3']))
        with self.assertRaises(UnsupportedModel): MatmulPlanner().lower(op(), MappingSpec(dataflow='weight_stationary'), arch())

    def test_burst_padding_is_physical_traffic(self):
        c = arch().config
        c['components']['dram']['burst_bytes'] = 64
        a = Architecture(c)
        r = EventSimulator().run(MatmulPlanner().lower(op(), MappingSpec(2, 2, 2), a), a)
        self.assertEqual(r.counters['dram.read_logical_bytes'], 8)
        self.assertEqual(r.counters['dram.read_bytes'], 128)
        self.assertEqual(r.counters['dram.write_bytes'], 64)
        self.assertEqual(r.counters['dma.transfer_bytes'], 192)

    def test_sparse_dense_and_explicit_estimate(self):
        e, t = op(k=16), MappingSpec(2, 2, 16)
        stats = RuntimeStats(0, activation_zero_fraction=.75, sample_count=32, allow_uniform_estimate=True)
        a = arch()
        dense = EventSimulator().run(MatmulPlanner().lower(e, t, a, stats), a)
        a = arch(mode='ideal_mac_skip')
        sparse = EventSimulator().run(MatmulPlanner().lower(e, t, a, stats), a)
        self.assertEqual((dense.counters['array.compute_cycles'], sparse.counters['array.compute_cycles']), (16, 4))
        self.assertEqual(dense.counters['dram.read_bytes'], sparse.counters['dram.read_bytes'])
        self.assertEqual(sparse.metadata['compute_statistics'], ['estimated_uniform'])
        with self.assertRaises(UnsupportedModel): MatmulPlanner().lower(e, t, a, RuntimeStats(0))
        with self.assertRaises(ValueError): MatmulPlanner().lower(e, t, a, replace(stats, event_id=1))
        exact = RuntimeStats(0, scope='tile', tile_active_macs={'0:0:0:0:2:2:16': 0})
        result = EventSimulator().run(MatmulPlanner().lower(e, t, a, exact), a)
        self.assertEqual(result.counters['array.compute_cycles'], 0)
        self.assertGreater(result.total_cycles, 0)  # zero MACs still move dense data

    def test_broadcast_batches_reload(self):
        e = OperatorEvent(event_id=0, module_id='expert.qk', component='expert', op_type='matmul',
                          a=TensorDesc((2, 1, 2, 2), 'float32', 'int8', 8),
                          b=TensorDesc((1, 3, 2, 2), 'float32', 'int8', 8),
                          output=TensorDesc((2, 3, 2, 2), 'float32', 'int8', 8),
                          m=2, n=2, k=2, batch=6, batch_shape=(2, 3))
        a = arch()
        r = EventSimulator().run(MatmulPlanner().lower(e, MappingSpec(2, 2, 2), a), a)
        self.assertEqual((r.total_cycles, r.counters['dram.read_bytes'], r.counters['array.dense_macs']), (42, 48, 48))

    def test_serialization_and_search(self):
        a = arch()
        search = SearchSpec(enabled=True, objective='latency', dataflows=('output_stationary',),
                            tile_m_candidates=(1, 2), tile_n_candidates=(1, 2), tile_k_candidates=(1, 2))
        choice = ComponentMapper(a, search).map_event(op())
        self.assertEqual(choice.evaluated_candidates, 8)
        self.assertEqual(choice.simulation.total_cycles, 7)
        restored = ExecutionPlan.from_dict(json.loads(json.dumps(choice.plan.to_dict())))
        self.assertEqual(json.dumps(EventSimulator().run(restored, a).to_dict(), sort_keys=True),
                         json.dumps(choice.simulation.to_dict(), sort_keys=True))
        with self.assertRaises(UnsupportedModel): ComponentMapper(a, search, max_candidates=1).map_event(op())

    def test_manager_replay_and_unsupported_report(self):
        with tempfile.TemporaryDirectory() as d:
            source, target = Path(d)/'source', Path(d)/'replay'
            with HardwareManager(source, backend='component_sim', mapping={'tile_m': 2, 'tile_n': 2, 'tile_k': 2},
                                 backend_config={'architecture': arch().config, 'save_plan': True}) as manager:
                manager.submit(op())
                manager.submit(replace(op(), event_id=1, mode='test_forward'))
            records = [json.loads(x) for x in (source/'simulations.jsonl').read_text().splitlines()]
            self.assertEqual(records[0]['simulation']['total_cycles'], 7)
            self.assertIn('plan', records[0])
            self.assertEqual(records[1]['status'], 'unsupported')
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
            subprocess.run([sys.executable, '-m', 'vla_tcs2.hardware', '--trace', str(source/'trace.jsonl'),
                            '--output-dir', str(target)], check=True, env=env, capture_output=True)
            self.assertEqual((source/'simulations.jsonl').read_text(), (target/'simulations.jsonl').read_text())
            with self.assertRaises(FileExistsError):
                with HardwareManager(source, backend='component_sim'): pass

    def test_manager_component_search_uses_component_costs(self):
        with tempfile.TemporaryDirectory() as d:
            with HardwareManager(d, backend='component_sim', backend_config={'architecture': arch().config},
                                 mapping_search={'enabled': True, 'objective': 'latency',
                                     'dataflows': ['output_stationary'], 'tile_m_candidates': [2],
                                     'tile_n_candidates': [2], 'tile_k_candidates': [1, 2]}) as manager:
                manager.submit(op())
            record = json.loads((Path(d)/'simulations.jsonl').read_text())
            self.assertEqual(record['mapping']['tile_k'], 2)
            self.assertEqual(record['simulation']['metadata']['search']['evaluated_candidates'], 2)


    def test_replay_backend_switch_does_not_inherit_incompatible_settings(self):
        with tempfile.TemporaryDirectory() as d:
            source, target = Path(d)/"source", Path(d)/"target"
            with HardwareManager(source, mapping_search={"enabled": True}) as manager:
                manager.submit(op())
            config = Path(d)/"config.json"
            config.write_text(json.dumps({"backend": "component_sim", "backend_config": {"architecture": arch().config},
                                          "mapping": {"tile_m": 2, "tile_n": 2, "tile_k": 2}}))
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/"src"))
            subprocess.run([sys.executable, "-m", "vla_tcs2.hardware", "--trace", str(source/"trace.jsonl"),
                            "--config", str(config), "--output-dir", str(target)], check=True, env=env, capture_output=True)
            summary = json.loads((target/"summary.json").read_text())
            self.assertFalse(summary["manifest"]["mapping_search"]["enabled"])
            self.assertEqual(summary["totals"]["estimated_calls"], 1)

    def test_stats_roundtrip_keeps_oracle_and_estimation_labels(self):
        with tempfile.TemporaryDirectory() as d:
            stats = RuntimeStats(0, activation_zero_fraction=.5, allow_uniform_estimate=True)
            event = replace(op(k=8), metadata={"runtime_stats": stats.to_dict()})
            with HardwareManager(d, backend="component_sim", backend_config={"architecture": arch(mode="ideal_mac_skip").config},
                                 mapping_search={"enabled": True, "objective": "latency", "dataflows": ["output_stationary"],
                                                 "tile_m_candidates": [2], "tile_n_candidates": [2], "tile_k_candidates": [4, 8]}) as manager:
                manager.submit(event)
            recovered = list(read_trace(Path(d)/"trace.jsonl"))[1]
            self.assertEqual(recovered.metadata["runtime_stats"], stats.to_dict())
            result = json.loads((Path(d)/"simulations.jsonl").read_text())["simulation"]
            self.assertEqual(result["metadata"]["mapping_policy"], "oracle_runtime_stats")
            self.assertEqual(result["metadata"]["compute_statistics"], ["estimated_uniform"])

    def test_missing_connection_and_stale_tile_statistics(self):
        c = arch().config
        c["connections"] = [x for x in c["connections"] if x["from"] != "local.data"]
        with self.assertRaisesRegex(ValueError, "No route"):
            MatmulPlanner().lower(op(), MappingSpec(), Architecture(c))
        a = arch(mode="ideal_mac_skip")
        stats = RuntimeStats(0, scope="tile", tile_active_macs={"0:0:0:0:2:2:2": 9})
        with self.assertRaisesRegex(ValueError, "exceed"):
            MatmulPlanner().lower(op(), MappingSpec(2, 2, 2), a, stats)
        with self.assertRaisesRegex(ValueError, "fixed mapping"):
            ComponentMapper(a, SearchSpec(dataflows=("output_stationary",))).map_event(op(), stats)

    def test_slow_bandwidth_stalls_without_changing_work(self):
        a = arch()
        p = MatmulPlanner().lower(op(k=16), MappingSpec(2, 2, 4), a)
        fast = EventSimulator().run(p, a)
        c = a.config
        c["components"]["dram"]["bandwidth_bytes_per_second"] = 1000
        slow = EventSimulator().run(p, Architecture(c))
        self.assertGreater(slow.total_cycles, fast.total_cycles)
        self.assertEqual(slow.counters, fast.counters)
        self.assertEqual(EventSimulator().run(ExecutionPlan(()), a).total_cycles, 0)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable; run hook integration in smolvla_eval")
class ComponentHookTests(unittest.TestCase):
    def test_raw_online_statistics_preserve_model_output(self):
        from vla_tcs2.hardware.modeling import capture_raw_activation_zeros
        layer = torch.nn.Linear(2, 2, bias=False)
        x = torch.tensor([[0., 1.], [0., 2.]])
        expected = layer(x)
        with tempfile.TemporaryDirectory() as d:
            with HardwareManager(d, backend="component_sim", runtime_stats_provider=capture_raw_activation_zeros,
                                 backend_config={"architecture": arch(mode="ideal_mac_skip").config}) as manager:
                manager.attach(layer)
                self.assertTrue(torch.equal(layer(input=x), expected))
            e = list(read_trace(Path(d)/"trace.jsonl"))[1]
            self.assertEqual(e.metadata["runtime_stats"]["activation_zero_fraction"], .5)
            self.assertEqual(len(layer._forward_hooks), 0)


if __name__ == '__main__':
    unittest.main()
